# ViCiRAG — CiRAG for Vietnamese Multi-hop QA

A reimplementation of **CIRAG: Construction-Integration Retrieval and Adaptive Generation for
Multi-hop Question Answering** ([arXiv:2601.06799](https://arxiv.org/abs/2601.06799), ACL 2026),
evaluated on the Vietnamese multi-hop QA dataset **ViMQA**, together with a replacement stopping
test for the answer generation module.

Final project for the Natural Language Processing course, theory track.

## Headline results

| | EM | F1 | Δ vs IRCoT |
|---|---|---|---|
| **2WikiMultihopQA, Qwen2.5-7B, 100 questions** | | | |
| IRCoT | 40.00 | 44.63 | |
| CIRAG | 54.00 | 57.97 | **+14.0** |
| *(original paper reports +13.4)* | | | |
| **ViMQA, Gemini 2.5 Flash, 500 questions** | | | |
| NativeRAG | 55.20 | 67.95 | −3.8 |
| IRCoT | 59.00 | 73.27 | |
| CIRAG, original stopping test | 50.80 | 66.66 | −8.2 |
| + fallback level | 57.00 | 73.05 | −2.0 |
| **+ new stopping test** | **60.40** | **75.08** | **+1.4** |

Three findings:

1. Our code **reproduces** the original paper on its own dataset, to within 0.6 EM points.
2. That margin **does not appear on ViMQA**. Four controlled experiments show the cause lies
   neither in our code, the Vietnamese prompts, a broken pipeline stage, nor backbone capability.
3. The cause is the **stopping test** in ACMG. It uses the model's refusal string as its signal
   that the context is sufficient, and a strong backbone almost never refuses. Replacing that
   signal with a test on the evidence itself gains **9.6 EM points** at no extra model calls.

Full details in [`report/main.pdf`](report/main.pdf) (Vietnamese) or
[`report/main_en.pdf`](report/main_en.pdf). All raw numbers in [`RESULTS.md`](RESULTS.md).

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # then fill in OPENROUTER_API_KEY

# 10 public sample questions, no dataset download needed
PYTHONPATH=src .venv/bin/python -m vicirag.pipeline \
    --dataset vimqa_sample --split dev --it 2
```

Works without a GPU: the embedding model falls back to CPU.

## Full run

```bash
# 1. Prepare the data (see Data below)
.venv/bin/python scripts/prepare_vimqa.py --hf nguyenlab/vimqa --limit 500

# 2. Run every experiment in the report
bash scripts/run_experiments.sh vimqa validation

# 3. Collect the result tables and draw the figures
PYTHONPATH=src .venv/bin/python scripts/collect_results.py
.venv/bin/python scripts/make_figures.py
```

`run_experiments.sh` skips runs that already have results in `outputs/`, so it is safe to rerun.

### Flags worth knowing

| Flag | Effect |
|---|---|
| `--method {cirag,native,ircot}` | Pick the method or a baseline |
| `--min-source-docs 2` | **The proposed stopping test**: require the core triple set to come from at least 2 documents |
| `--no-passage-fallback` | Drop the fallback level, giving the exact behaviour of Eq. 6 in the original paper |
| `--ab {fact,sentence,passage,fact-sentence,fact-passage}` | Context-level ablation (Table 2) |
| `--it N` | Maximum ICI rounds (default 4) |
| `--llm-name`, `--openie-llm-name` | Change the reasoning model and the triple extraction model |
| `--lang {vi,en,en-vi}` | Prompt language, used for the reproduction run on 2WikiMQA |

## Notebook

[`notebooks/CiRAG_Vietnamese.ipynb`](notebooks/CiRAG_Vietnamese.ipynb) runs on Google Colab or
Kaggle and depends on no local paths. It has the eight sections the project requires: install
libraries, load the data, explore and preprocess, build the model, evaluate, compare against
baselines, analyse errors, and demo on new Vietnamese data.

## Data

ViMQA \[[Le et al., 2022](https://aclanthology.org/2022.lrec-1.700/)\] is a Vietnamese multi-hop
QA dataset written by hand from Vietnamese Wikipedia, laid out like HotpotQA.

```bash
# Full release on HuggingFace, no EULA needed
.venv/bin/python scripts/prepare_vimqa.py --hf nguyenlab/vimqa --limit 500

# 2WikiMultihopQA, used for the reproduction run on the original paper's dataset
.venv/bin/python scripts/prepare_2wiki.py --limit 100
```

Each script writes two files under `data/<dataset>/processed/`: one with the questions and one
with the corpus. Following IRCoT, the corpus merges the passages of every question and removes
duplicates by title, so the system has to find two evidence passages among nearly two thousand
rather than among the ten that come with each question.

## Repository layout

```
src/vicirag/
  pipeline.py       wires ICI and ACMG together, runs batches, logs trajectories
  ici.py            the Construction-Integration loop
  acmg.py           the triple → sentence → passage cascade and the proposed stopping test
  openie.py         offline triple extraction over the corpus, cached by content hash
  retriever.py      dense index and the two provenance maps M^D, M^S
  eval/vi_metrics.py  EM and F1 with normalisation for Vietnamese
  baselines/        NativeRAG and IRCoT, sharing the retriever with CIRAG
  prompts_vi/       Vietnamese prompts for OpenIE, Integration and ACMG
  llm/openrouter.py OpenRouter client with a per-model SQLite cache

scripts/            data preparation, experiment runner, result tables, figures
tests/              38 tests, no GPU and no API calls
report/             the two report PDFs, Vietnamese and English
```

## Tests

```bash
.venv/bin/python tests/test_core.py              # answer normalisation, LLM output parsing
.venv/bin/python tests/test_pipeline_offline.py  # ICI and ACMG against a scripted LLM
```

Both finish in seconds, need no GPU and make no API calls.

## Cost

Every experiment in the report cost about **8 USD** through OpenRouter. The SQLite cache is keyed
by model and prompt, so rerunning is close to free. Triple extraction accounts for most of the
calls, so it uses a cheaper model.

## What is not implemented

The original paper has a third module, **Trajectory Distillation**, which trains a small model to
copy the reasoning steps of a large one and so cut cost. This repository does not implement it,
so every number in the report is compared against the *w/o TD* setting the paper reports.

## Credits

Prompts and output formats are reused from the official repository
[52566rz/CIRAG](https://github.com/52566rz/CIRAG). The reimplementation of ICI and ACMG, all
Vietnamese prompts, the evaluation module for Vietnamese, and the proposed stopping test are ours.
