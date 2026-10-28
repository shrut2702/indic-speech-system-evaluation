# Indic Speech System Evaluation

Indic speech-to-text-response pipeline and its evaluation harness.

Speech clip in, text answer out: **ASR → LLM**.

## What it measures

- **Transcript accuracy** — raw WER, normalized WER, normalized CER (all micro), plus a
  separate entity error rate.
- **Response correctness** — whether the LLM's answer contains the known answer, scored
  by string matching, no LLM judge.
- **Latency** — p50/p90/p95 per stage: ASR, time to first token, LLM, end-to-end, and
  ASR real-time factor per item.
- **Where failures come from** — a control run feeds the LLM the correct transcript, so
  ASR-caused failures can be separated from the LLM's own limits.

## Benchmark

25 questions × 5 languages (en, gu, hi, pa, ta) = 125 clips per condition. The same 25
questions in every language, so language is the only variable.

Two acoustic conditions: `clean` (24 kHz, as synthesized) and `noisy_0db` (the same
clips mixed with real cafeteria babble at 0 dB SNR).

Five categories: entity/number, normal, code-switch, disfluent, long-form.

The manifests at `artifacts/bench/items_<lang>.jsonl` are the source of truth for every
item — its text, reference transcript, reference answer and aliases.

## Setup

Needs Python 3.12. How much you install depends on what you want to do.

### 1. Create and activate a virtual environment

```bash
python -m venv venv
```

| Platform      | Activate with              |
| ------------- | -------------------------- |
| Windows       | `./venv/scripts/activate`  |
| macOS / Linux | `source venv/bin/activate` |

### 2. Install dependencies

Pick the row that matches what you plan to run.

**A. Recompute the reported metrics from the logged records** — no models, no keys, any OS:

```bash
pip install PyYAML
```

**B. Run the pipeline with the hosted models** (Saaras ASR + Sarvam LLM) — no GPU, any OS:

```bash
pip install PyYAML soundfile numpy torch==2.6.0
```

**C. Run the local ASR (SraVaani)** — needs an NVIDIA GPU, so Windows or Linux only:

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124 torchaudio==2.6.0+cu124
pip install -r requirements.txt
```

`requirements.txt` pins the CUDA builds of torch, so it will not install on macOS. On a
Mac, use option B and pass `--asr-model saaras:v4`.

### 3. API keys — only for B and C

Create a file named `.env` in the repo root:

```
SARVAM_API_KEY=...      # the Sarvam LLM, and the Saaras ASR
HF_TOKEN=...            # only for C: SraVaani is gated, accept its terms on HuggingFace first
```

Model downloads go to `artifacts/hf`, not your system drive.

### 4. Audio clips

The clips are not in the repo. Download them here:

**https://drive.google.com/drive/folders/1kL4Sszz3vkkoKl6IMzXeHjki2MRBiUUY**

Unzip into `artifacts/bench/` so the two condition folders sit beside the manifests:

```
artifacts/bench/
  items_en.jsonl  items_gu.jsonl  items_hi.jsonl  items_pa.jsonl  items_ta.jsonl
  clean/<lang>/<category>/<item_id>.wav
  noisy_0db/<lang>/<category>/<item_id>.wav
```

Use these rather than regenerating them: speaker assignment is seeded
and deterministic, but the TTS is a hosted model, so regenerated audio is not
bit-identical and the WER numbers would shift. The clips are part of the dataset.

Only needed for options B and C — option A reads the logged records, not the audio.

## Running it

`main.py` is the only entry point. It has four modes, and they split into two jobs:
**inference** writes records, **scoring** reads them.

| Mode        | What it does                                                   | Needs       |
| ----------- | -------------------------------------------------------------- | ----------- |
| `pipeline`  | runs ASR → LLM over the clips, writes one record per clip      | clips, keys |
| `llm-only`  | the control: runs the LLM on the correct transcripts, no audio | keys        |
| `eval`      | transcript accuracy and latency, from records                  | nothing     |
| `responses` | answer correctness, from records                               | nothing     |

### Just score the existing records

Nothing to install beyond PyYAML, no keys, no audio:

```bash
# clean condition baseline (sravaani)
python main.py eval artifacts/runs/pipeline_clean_sravaani1.0_sarvam105b.jsonl
python main.py responses artifacts/runs/pipeline_clean_sravaani1.0_sarvam105b.jsonl

# noisy condition baseline (sravaani)
python main.py eval artifacts/runs/pipeline_noisy_0db_sravaani1.0_sarvam105b.jsonl
python main.py responses artifacts/runs/pipeline_noisy_0db_sravaani1.0_sarvam105b.jsonl

# clean condition fix (saaras)
python main.py eval artifacts/runs/pipeline_clean_saaras_v4_sarvam105b.jsonl
python main.py responses artifacts/runs/pipeline_clean_saaras_v4_sarvam105b.jsonl

# noisy condition fix (saaras)
python main.py eval artifacts/runs/pipeline_noisy_0db_saaras_v4_sarvam105b.jsonl
python main.py responses artifacts/runs/pipeline_noisy_0db_saaras_v4_sarvam105b.jsonl
```

Both print their tables and write files under `artifacts/results/<run_id>/`:
`per_sample.csv`, `summary.json`, `responses.csv`.

### Run the pipeline yourself

Start with a 2-clip smoke test so a typo costs you nothing:

```bash
python main.py pipeline --conditions clean --langs hi --limit 2 --run-id smoke
```

Then the full run. `--run-id` names the output file, so give it something you will
recognise later:

```bash
python main.py pipeline --conditions clean     --run-id my_clean_run
python main.py pipeline --conditions noisy_0db --run-id my_noisy_run
```

Each writes `artifacts/runs/<run-id>.jsonl`. Score it with `eval` and `responses` as above.

A full 125-clip run takes roughly 15 minutes and is dominated by the LLM, not the ASR.

### Run the control

```bash
python main.py llm-only --run-id my_control
```

This reads `reference_transcript` straight from the manifests and never touches the
audio, so it shows how well the LLM does when the ASR is perfect.

### Compare a pipeline run against the control

Pass both files to `responses`. It prints each one, then a paired table that splits the
failures into ASR-caused and LLM-caused:

```bash
python main.py responses artifacts/runs/my_control.jsonl artifacts/runs/my_clean_run.jsonl
```

### Swap models without editing any file

```bash
python main.py pipeline --conditions noisy_0db --asr-model saaras:v4   # hosted ASR
python main.py llm-only --llm-reasoning off                            # reasoning off
```

### Flags worth knowing

| Flag                  | Effect                                       |
| --------------------- | -------------------------------------------- |
| `--langs hi ta`       | only these languages (default: all five)     |
| `--limit 5`           | only the first 5 items, for a smoke test     |
| `--run-id <name>`     | names the output file (default: a timestamp) |
| `--conditions clean`  | `clean` or `noisy_0db` (default: both)       |
| `--asr-model <name>`  | `saaras:v4` for the hosted ASR               |
| `--llm-reasoning off` | disables the LLM's reasoning                 |

## Reproducing the reported numbers

Inference and scoring are decoupled on purpose. Every run writes one JSON record per
item to `artifacts/runs/`, carrying the transcript, the response, per-stage latencies and
the model identifiers. `eval` and `responses` read those records and **never call a
model**, so every metric can be recomputed from the logs with no API key and no GPU.

Re-running inference will not reproduce the responses exactly — the hosted LLM is not
deterministic and offers no seed. Two identical control runs agreed on 108 of 125
answers. The logged records are the record. ASR output _is_ deterministic and reproduces
exactly.

## Notes on the numbers

- Raw and normalized WER are reported side by side. The normalizer folds spoken numbers
  to digits, so the two have different denominators — compare the rates, not the error
  counts.
- Entity error rate covers the entity/number and code-switch categories only; the other
  categories carry no entities.
- ASR latency and RTF are **not comparable** between the local and hosted ASR: one is
  compute on a GTX 1660 Ti, the other a network round trip. The LLM stage is comparable.
