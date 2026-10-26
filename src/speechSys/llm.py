"""Gemma 3 (4-bit) wrapper: one question in, one short same-language answer out.

Greedy decode (temperature 0) so a rerun reproduces the record. Latency and token
count are returned raw; tok/s and TTFT belong to evaluate.py, not here.
"""
import time

SYSTEM_PROMPT = (
    "You are a voice assistant for Indian users. Answer in the SAME language and "
    "script as the question. Reply with only the answer itself - a name, a number, "
    "a phrase. No explanation, no preamble, no restating the question."
)


def load(repo, device, token=None, dtype="float16"):
    import torch
    from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    tok = AutoTokenizer.from_pretrained(repo, token=token)
    kw = {}
    q = getattr(AutoConfig.from_pretrained(repo, token=token), "quantization_config", None)
    if q:
        # prequantized checkpoints ship compute_dtype=bfloat16; Turing sm75 has no bf16
        q = {k: v for k, v in dict(q).items() if not k.startswith("_")}
        q["bnb_4bit_compute_dtype"] = dtype
        kw["quantization_config"] = BitsAndBytesConfig(**q)
    model = AutoModelForCausalLM.from_pretrained(
        repo, device_map=device, dtype=getattr(torch, dtype), token=token, **kw).eval()
    return tok, model


def respond(tok, model, question, max_new_tokens=48, system=SYSTEM_PROMPT):
    """(text, latency_s, n_new_tokens)"""
    import torch
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(model.device)
    t = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    dt = time.perf_counter() - t
    new = out[0][ids["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip(), dt, int(new.shape[0])
