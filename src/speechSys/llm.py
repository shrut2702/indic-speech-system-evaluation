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
    """(text, latency_s, n_new_tokens, ttft_s, attempts)"""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _FirstToken(StoppingCriteria):
        """Never stops; called once per new token, so the first call marks TTFT."""
        def __init__(self):
            self.at = None

        def __call__(self, input_ids, scores, **kw):
            if self.at is None:
                self.at = time.perf_counter()
            return False

    msgs = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(model.device)
    first = _FirstToken()
    t = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id,
                             stopping_criteria=StoppingCriteriaList([first]))
    dt = time.perf_counter() - t
    new = out[0][ids["input_ids"].shape[1]:]
    ttft = (first.at - t) if first.at else None
    return tok.decode(new, skip_special_tokens=True).strip(), dt, int(new.shape[0]), ttft, 1


# --- Sarvam API backend -------------------------------------------------------
# Selected by a model name starting with "sarvam" in params.yaml, so the swap is
# config-only. No VRAM, no commit memory - it sidesteps the local 6 GB limit.
SARVAM_URL = "https://api.sarvam.ai/v1/chat/completions"


def _sarvam_key(root):
    import os
    if os.environ.get("SARVAM_API_KEY"):
        return os.environ["SARVAM_API_KEY"]
    f = root / ".env"
    for line in f.read_text(encoding="utf-8").splitlines() if f.exists() else []:
        k, _, v = line.partition("=")
        if k.strip() == "SARVAM_API_KEY":
            return v.strip().strip("\"'")
    raise SystemExit("SARVAM_API_KEY not found in environment or .env")


def _call_api(model, question, key, system=SYSTEM_PROMPT, stream=True,
              reasoning="default"):
    """One request: (text, latency_s, n_completion_tokens, ttft_s).

    `reasoning` is "default" to leave reasoning_effort unset (the API defaults to
    "low"), "off" to disable reasoning, or a level the API accepts. Disabling it cuts
    tokens and latency by two orders of magnitude but loses multi-step items.
    """
    import json, urllib.request, urllib.error
    # No completion cap is sent. The parameter is max_tokens (not
    # max_completion_tokens), but with reasoning enabled a small cap is spent thinking
    # and returns content: null, so capping is only safe with reasoning off.
    # max_new_tokens in params.yaml is honoured by the local path only.
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": question}],
        "temperature": 0,
    }
    if reasoning != "default":
        # the API disables reasoning on a JSON null, not on a string
        body["reasoning_effort"] = None if reasoning in ("off", "none") else reasoning
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    for attempt in range(6):
        try:
            return _one_call(body, key, stream)
        except urllib.error.HTTPError as e:
            # with reasoning off a call takes ~0.2s, so 125 items trip the rate limit
            # in seconds. Back off rather than lose the run.
            if (e.code == 429 or e.code >= 500) and attempt < 5:
                time.sleep(2 ** attempt)
                continue
            raise SystemExit(
                f"Sarvam HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:400]}")
    raise SystemExit("unreachable")


def _one_call(body, key, stream):
    """One HTTP request: (text, latency_s, n_completion_tokens, ttft_s)."""
    import json, urllib.request
    req = urllib.request.Request(SARVAM_URL, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json", "api-subscription-key": key,
        "Authorization": f"Bearer {key}"})
    parts, ttft, n = [], None, 0
    t = time.perf_counter()
    if not stream:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.loads(r.read())
        dt = time.perf_counter() - t
        n = (d.get("usage") or {}).get("completion_tokens", 0)
        msg = (d["choices"][0].get("message") or {})
        return ((msg.get("content") or "").strip(), dt, int(n), None)
    with urllib.request.urlopen(req, timeout=180) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                n = d["usage"].get("completion_tokens", n)
            for ch in d.get("choices") or []:
                delta = (ch.get("delta") or {}).get("content") or ""
                if delta:
                    if ttft is None:
                        ttft = time.perf_counter() - t
                    parts.append(delta)
    return "".join(parts).strip(), time.perf_counter() - t, int(n), ttft


def respond_api(model, question, key, max_new_tokens=48, system=SYSTEM_PROMPT,
                stream=True, retries=2, reasoning="default"):
    """(text, latency_s, n_completion_tokens, ttft_s, attempts) - as respond().

    sarvam-105b returns content: "" on roughly a third of hard items: its reasoning
    runs to the server's 2048-token ceiling and never reaches an answer. It is
    non-deterministic even at temperature 0, so a retry usually succeeds. Latency and
    tokens are summed across attempts, because that is what the call really cost.
    """
    total_s = total_n = 0
    ttft = None
    for attempt in range(1, retries + 2):
        txt, dt, n, tf = _call_api(model, question, key, system=system, stream=stream,
                                   reasoning=reasoning)
        total_s += dt
        total_n += n
        if ttft is None:
            ttft = tf
        if txt:
            return txt, total_s, total_n, ttft, attempt
    return "", total_s, total_n, ttft, retries + 1
