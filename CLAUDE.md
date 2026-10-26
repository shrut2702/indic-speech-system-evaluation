# Vaani

Indic speech-to-text-response pipeline + evaluation harness, built for a Speech AI
research assignment (Task 2). Deliverable is **working code + evaluation results +
ablation analysis**, reproducible from a README.

The graded signal is the **evaluation infrastructure**, not the pipeline. The pipeline
is three pretrained components glued together; the harness is the actual work. Prefer
rigour in measurement over features.

## Pipeline

`Speech input -> ASR (Indic) -> LLM -> text response`

## Layout

```
main.py              single CLI entry point (pipeline | eval)
app.py               optional demo UI. Build LAST, only if numbers are already in.
config/config.yaml   paths, dataset + model registry
params.yaml          model choices, decode params, generation params
requirements.txt     finalized runtime deps ONLY (see rules below)
src/speechSys/       library code (FLAT - see rules below)
research/            experiments, demos, one-off scripts (see rules below)
artifacts/           all generated + downloaded data (see rules below)
template.py          original scaffold generator. Do not edit, do not re-run.
```

## Code style

- Keep it short. No verbose comments, no docstring essays, no narration of obvious
  lines. Write what the surrounding code reads like.
- If the standard library already does it, use the standard library. Do not
  reimplement path handling, JSON, hashing, CSV, statistics, argparse, dataclasses,
  timing, or anything else already in `stdlib`.
- Reuse existing modules and functions in this repo before writing a new one. Check
  `src/speechSys/` first.
- Do not overcomplicate. No abstraction layers, factories, registries, plugin systems
  or base classes until something concrete needs them twice. Plain functions and
  dataclasses over classes; flat over clever.

## Hard rules

### `src/speechSys/` is flat

No nested packages, no `src/speechSys/asr/whisper/...`. A handful of sibling modules
at one level. Confirmed modules:

- `pipeline.py` - ASR -> LLM, emits one record per example
- `evaluate.py` - eval harness: reads logged records, computes all metrics

Add further flat modules only when a real need appears (likely: `asr.py`, `llm.py`,
`dataset.py`, `normalize.py`, `metrics.py`, `judge.py`, `config.py`). Do not create
them speculatively, and do not create a subfolder to "organize" them.

### `research/` is for throwaway work

Experiments, demos, running an ASR or LLM in isolation to understand its setup,
audio augmentation scripts (prepending silence, mixing noise, telephone-band
filtering), scratch benchmarks, notebooks.

- The two trees never import each other, in either direction. `research/` is
  self-contained, and `src/speechSys/` never reaches into it.
- Code moves one way only: written and verified in `research/`, then rewritten as a
  flat module under `src/speechSys/`. Copy it across - do not import it from its old
  home, and do not leave the pipeline depending on a research script.
- Research scripts are allowed to be messy, duplicated, and abandoned.

### `requirements.txt` holds finalized deps only

Only packages needed to run the pipeline, the eval harness, or anything else
non-experimental. Never add a dep just because a research script tried it. If a model
or library was explored and discarded, its dep does not belong here. Install
experimental packages into the venv ad hoc; add to `requirements.txt` only once the
code that needs it lands under `src/speechSys/`.

Pin versions for anything that affects numbers (ASR backends, decoding libs, metric
libs). `ai4bharat/indic-conformer-600m-multilingual` needs `onnxruntime==1.20.1`.

### `main.py` is the only entry point

Two modes, selected by CLI args:

- run the pipeline over a dataset -> writes raw per-example records
- run eval over already-logged records -> writes metrics + tables

Inference and scoring are **decoupled on purpose**: the normalizer and the rubric will
change, and re-scoring must never require re-running inference. Scoring reads the
logged records file, never calls a model.

### Audio + data live under `artifacts/`

Clips: `artifacts/bench/<acoustic_condition>/<lang>/<category>/`

e.g. `artifacts/bench/clean/hindi/entity_number/`,
`artifacts/bench/noisy_10db/tamil/code_switch/`

A JSON/JSONL manifest accompanies the clips and is the single source of truth: clip
path, language, category, acoustic condition, reference transcript, reference answer,
true audio duration.

Run outputs also go under `artifacts/` (e.g. `artifacts/runs/<run_id>.jsonl`,
`artifacts/results/`). Do not write generated files to the repo root.

## Benchmark design

25 items x 4 languages = 100 examples. The same 25 questions are translated across
languages (a **parallel set**), so language is the only variable.

Items are split across categories (named entity/number, normal, code-switch,
disfluent, long-form). The split is not frozen yet - the manifest is the source of
truth for it, not this file.

Two acoustic conditions, generated from the same clips so the comparison stays
controlled and the item count stays at 25:

- `clean` - 24 kHz mono 16-bit, as synthesized by Bulbul v4 Flash.
- `noisy_5db` - real cafeteria babble (DEMAND `PCAFETER`) mixed to **5 dB SNR**,
  seeded (`--seed 0`) so every clip gets a different noise window and reruns are
  identical. Produced by `research/add_noise.py`.

Both conditions are 24 kHz: the noise bed is upsampled to 24 kHz and mixed there, so
sample rate is never a confound between them. Downsampling to 16 kHz for the ASR
happens in the pipeline, identically for both conditions.

5 dB was chosen on evidence, not taste: across 6 trial clips, 10 dB changed nothing on
half of them and scored *better* than clean on two, so it does not discriminate. 5 dB
moves WER on most items without flooring any.

**Effective n is 25, not 100** - the per-language versions are not independent samples.
Report it as "25 items x 4 languages" and never claim 100 independent examples.

Every item carries a known **reference answer**, written in advance. Without it
correctness is unmeasurable. Questions are closed-form (answer derivable from the
clip's content), not open-ended chat.

## Measurement rules

- **RTF applies to the ASR stage only** (`processing_time / audio_duration`). The LLM
  stage has no audio duration - report absolute latency, TTFT, tokens/sec. Never
  report a single RTF for the whole pipeline.
- **Log latency per stage**, not just end-to-end. The ablation depends on it.
- **Report both raw and normalized WER/CER.** Indic normalization (Unicode NFC,
  ZWJ/ZWNJ, danda, native vs ASCII numerals) swings WER by several points; the delta
  is itself a finding. Use CER alongside WER for agglutinative Dravidian languages.
- **ASR hallucination is out of scope.** Only the LLM response is assessed for
  hallucination. Its definition is still open - do not harden a metric or a threshold
  for it yet.
- **Response rubric labels are mutually exclusive:** `correct`, `incorrect`,
  `hallucinated`, `refused`, plus a separate `language_drift` flag. The
  `hallucinated` label is provisional until its criteria are settled.
- **LLM-as-judge runs at temperature 0** against a fixed rubric, and must be validated
  against ~30 hand-labelled examples with Cohen's kappa reported. An unvalidated judge
  is not a measurement.
- **Cache inference** keyed on `(model, audio, params)` hash. Without it the iteration
  loop is unworkable on one GPU.
- **Swap models by config, not by editing code.** The ablation runs the same harness
  twice and diffs; if switching backends needs a code change, the comparison isn't
  trustworthy.
- **RTF needs pinned hardware.** Warm up, run each config >=3 times, report median and
  spread, and record the machine spec with the results.

## WER rubric and normalization scope

Implemented in `src/speechSys/normalize.py`. Any change to it invalidates previously
reported numbers - re-score from the logged records, and say which version produced a
table.

### Metrics

- `WER = (S + D + I) / N` over whitespace tokens; `CER` the same over characters.
  Word match is exact string equality, so one wrong character fails the whole word.
- Report **CER alongside WER for Tamil** (and any Dravidian language added later):
  agglutination packs morphemes into single long words, so WER overstates the damage.
- Report **raw and normalized side by side, plus the delta**. Raw alone misrepresents
  the model (SraVaani emits no punctuation, so every item takes a fixed penalty);
  normalized alone looks like hiding. The gap is itself a result.
- **Entity error rate is separate** from WER: whether each `reference_entities` token
  survives verbatim. It predicts response correctness better than WER does.

### The normalizer does exactly this, in order

1. Unicode NFC.
2. Native digits to ASCII (Devanagari, Gujarati, Tamil, Gurmukhi).
3. Number words to digits, per language. A run of bare 0-9 words concatenates
   (`चार आठ सात` -> `487`); a run containing a tens word or a multiplier accumulates
   (`दो सौ उनचास` -> `249`).
4. Lowercase, English only - Indic scripts are caseless.
5. Drop punctuation: danda, double danda, `? ! , . ; : " ' ( ) [ ] -`, ZWJ, ZWNJ.
6. Collapse whitespace.
7. Fold loanwords from Latin into **the spoken language's own script**, so a Hindi
   transcript stays Hindi: `USB` -> `यूएसबी`, `ਯੂਐਸਬੀ` in Punjabi, `usb` in English.
   An ASR that Latinizes a term therefore costs no WER. `TERMS` holds one row per
   term with its form in each language.

Applied identically to reference and hypothesis, always with the same `lang`.

Step 3 **changes N**: `एक हज़ार सात सौ अट्ठासी` is 5 tokens, `1788` is 1. Normalized WER
therefore has a different denominator from raw WER. Expected - state it rather than
reconciling the two.

### Explicitly out of scope

Each of these counts as an error today. Decide per item; do not quietly widen the
normalizer to make a number look better.

- **Filler words.** `umm`, `uh` and stutter repetitions stay in both strings. Report
  the `disfluent` category with and without filler removal as a separate row.
- **Clock times.** `six forty five` / `छह बजकर पैंतालीस` does not become `6:45`; a
  number mapper cannot know it is a time. `ent_05` is scored via `answer_aliases`.
- **Two-unit digit runs.** `चार आठ` sums to `12`, not `48` - deliberate, so that
  "three four books" is not mangled. Nothing in the current bench hits it.
- **Loanwords outside the lexicon.** `TERMS` covers acronyms, brands and bare tech
  terms. General loans (`ऑफिस`, `क्विज़`, `पोस्ट`) are not folded. Add an observed
  term to `TERMS` with a self-check case in the same change.
- **Inflected native forms.** Only the Latin spelling is folded. A native form that
  carries a suffix (Gujarati `યુએસબીનું`, Tamil `யூஎஸ்பியின்`) is left alone - it already
  matches the reference, and prefix matching to strip suffixes is unsafe, since Tamil
  `வை` begins many native words.
- **`एक` / `ਇੱਕ` as an article.** These mean both "one" and "a", so step 3 turns the
  article into `1`. Identical on both sides, so WER is unaffected - but normalized
  transcripts will read oddly.
- **Synonyms and lexical variants.** `मुंबई`/`बॉम्बे` are handled by
  `answer_aliases` at answer-scoring time, never by the transcript normalizer.

## Record format

One JSONL record per example from the pipeline, carrying at minimum: example id,
language, category, acoustic condition, clip path, true audio duration, reference
transcript, reference answer, ASR hypothesis, LLM response, per-stage latencies, model
+ param identifiers, run id. Metrics are derived from this file - never computed inline
during inference.

## Environment

Windows, PowerShell. `venv/` is in-repo and is not a package - never import from it and
never add it to the layout docs. Not a git repository yet.
