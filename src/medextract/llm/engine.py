"""Self-hosted LLM engine via transformers — no external API, no network.

Loads an open instruct model (default Qwen3-8B, 8.2B ≤ 9B) on a GPU, checks the
≤9B parameter budget at startup, and does batched chat generation. Used only to
rerank retrieved candidates (never for NER — the spans come from GLiNER). Set
``load_in_4bit`` to fit the 8B model in ~6 GB on a free Colab T4.
"""
from __future__ import annotations

import logging
from typing import List, Optional

log = logging.getLogger("medextract.llm.engine")

MAX_PARAMS = 9_000_000_000  # competition cap


class LLMEngine:
    def __init__(
        self,
        model_name: str = "/mnt/pretrained_fm/Qwen_Qwen3-8B",
        device: str = "wait",
        dtype: str = "bfloat16",
        max_new_tokens: int = 128,
        min_free_gb: float = 18.0,
        enforce_param_cap: bool = True,
        enable_thinking: bool = False,
        load_in_4bit: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ..utils.gpu import resolve_device

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.device = resolve_device(device, min_free_gb=min_free_gb)
        torch_dtype = getattr(torch, dtype, torch.bfloat16)

        log.info("loading LLM %s on %s (%s%s)", model_name, self.device, dtype,
                 ", 4-bit" if load_in_4bit else "")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if load_in_4bit:
            # nf4 quantization: fits the 8B model in ~6 GB so it runs on a free
            # Colab T4. device_map pins it to the resolved GPU (no extra .to()).
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb, device_map={"": self.device})
            self.model = model.eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
            self.model = model.to(self.device).eval()
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        n_params = sum(p.numel() for p in self.model.parameters())
        log.info("LLM parameter count: %.2fB", n_params / 1e9)
        if enforce_param_cap and n_params > MAX_PARAMS:
            raise ValueError(
                f"{model_name} has {n_params/1e9:.2f}B params > 9B competition cap"
            )
        self._torch = torch

    def chat(self, messages_batch: List[List[dict]], max_new_tokens: Optional[int] = None,
             temperature: float = 0.0) -> List[str]:
        """Generate a completion for each chat (list of {role, content}) message list."""
        torch = self._torch

        def _tmpl(m):
            try:  # Qwen3 supports enable_thinking; harmless kwarg elsewhere
                return self.tok.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True,
                    enable_thinking=self.enable_thinking)
            except TypeError:
                return self.tok.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True)

        prompts = [_tmpl(m) for m in messages_batch]
        enc = self.tok(prompts, return_tensors="pt", padding=True,
                       padding_side="left", truncation=True, max_length=4096)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        return [self.tok.decode(g, skip_special_tokens=True).strip() for g in gen]

    def complete(self, prompt: str, system: Optional[str] = None, **kw) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        return self.chat([msgs], **kw)[0]
