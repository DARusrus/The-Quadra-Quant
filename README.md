<!-- README.md — paste this entire file into your GitHub repo root -->

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12&height=200&section=header&text=LLM%20Quantization%20Benchmark&fontSize=40&fontColor=fff&fontAlignY=38&desc=Q4_K_M%20vs%20IQ4_XS%20%E2%80%94%20Phi-3-mini%20%26%20Mistral-7B&descAlignY=58&descColor=d8b4fe" width="100%"/>

<br/>

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![llama-cpp](https://img.shields.io/badge/llama--cpp--python-CUDA%2012.1-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-6.13-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-GPU%20T4-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

<br/>

### 🧠 Compare quantized LLMs live — memory, reasoning, and response quality at your fingertips

[**🚀 Try the Live Demo**](#) &nbsp;·&nbsp; [**📓 Open in Kaggle**](#) &nbsp;·&nbsp; [**📊 View Results**](#-metrics-captured) &nbsp;·&nbsp; [**🛠 How It Works**](#%EF%B8%8F-architecture)

<br/>

</div>

---

## ✨ What Is This?

> **A fully interactive benchmarking suite** that pits two quantization strategies — `Q4_K_M` (standard 4-bit) vs `IQ4_XS` (importance-matrix optimized 4-bit) — against each other across **three cognitive tasks**, on two state-of-the-art open-source LLMs.
>
> No API keys. No paid services. 100% open-source, running on free Kaggle / HuggingFace GPUs.

<br/>

---

## 🎯 Models & Variants

<div align="center">

| Model | Variant | Quantization | Size | Strategy |
|:------|:--------|:-------------|:-----|:---------|
| 🔵 **Phi-3-mini-4k** | `phi_q4km` | Q4_K_M | ~2.3 GB | Standard K-quant medium |
| 🟣 **Phi-3-mini-4k** | `phi_iq4xs` | IQ4_XS | ~2.0 GB | Importance-matrix optimized |
| 🔴 **Mistral-7B-v0.3** | `mistral_q4km` | Q4_K_M | ~5.2 GB | Standard K-quant medium |
| 🟠 **Mistral-7B-v0.3** | `mistral_iq4xs` | IQ4_XS | ~4.1 GB | Importance-matrix optimized |

</div>

<br/>

---

## 🧪 Three Benchmark Tests

<div align="center">
<table>
<tr>
<td align="center" width="33%">

### 🧠 Buffer Window Memory

![](https://img.shields.io/badge/Type-Short--term%20Recall-7c3aed?style=flat-square)

Maintains a **sliding window** of the last 3 conversation turns in context. At the end, asks the model recall questions.

**What it tests:** Can the model remember facts stated a few turns ago?

</td>
<td align="center" width="33%">

### 📝 Summary Memory

![](https://img.shields.io/badge/Type-Long--term%20Retention-0891b2?style=flat-square)

Compresses the entire conversation history into a **running summary string** — updated after every turn, no second LLM call.

**What it tests:** Does the model retain facts across many turns without a full context window?

</td>
<td align="center" width="33%">

### 🔗 Chain-of-Thought

![](https://img.shields.io/badge/Type-Analytical%20Reasoning-059669?style=flat-square)

Presents **math and logic problems**, requesting explicit step-by-step reasoning with a clearly stated final answer.

**What it tests:** Is the reasoning process correct, coherent, and extractable?

</td>
</tr>
</table>
</div>

<br/>

---

## 🏗️ Architecture

```mermaid
flowchart TD
    A([🚀 Start]) --> B[Cell 1\nDependency Install\nllama-cpp CUDA 12.1]
    B --> C[Cell 2\nConfig + Seeds\nModel Downloads]
    C --> D[Cell 3\nInference Engine\n+ Test Suite]
    D --> E[Cell 4\nBenchmark Loop\nPlots + Persistence]
    E --> F[Cell 5\nGradio Live Demo\nHF Spaces Deploy]

    D --> G[(📁 GGUF Models\n4 variants)]
    E --> H[(💾 results.json\nmetrics.csv)]
    F --> I([🌐 Public URL\nshare=True])

    style A fill:#7c3aed,color:#fff,stroke:none
    style F fill:#059669,color:#fff,stroke:none
    style I fill:#0891b2,color:#fff,stroke:none
    style G fill:#1e293b,color:#94a3b8,stroke:#334155
    style H fill:#1e293b,color:#94a3b8,stroke:#334155
```

<br/>

---

## 📐 Pipeline — 5 Cells, One Notebook

<div align="center">
<table>
<tr>
<td>

**`CELL 1`** &nbsp; 🔧 &nbsp; **ENVIRONMENT**

</td>
<td>

- Single clean `llama-cpp-python` install (CUDA 12.1 wheel) — no double install
- Gradio 6.13, matplotlib, seaborn, tqdm
- GPU hardware check + all-imports verification before proceeding

</td>
</tr>

<tr>
<td>

**`CELL 2`** &nbsp; ⚙️ &nbsp; **CONFIG + DOWNLOADS**

</td>
<td>

- Full seed coverage: `random` + `numpy` + `PYTHONHASHSEED=42`
- Kaggle-compatible paths via `MODEL_DIR` env-var fallback (not hardcoded `/content/`)
- Resume-capable downloads — 64KB chunks, `.part` file kept on interruption
- All 4 GGUF models verified on disk before proceeding

</td>
</tr>

<tr>
<td>

**`CELL 3`** &nbsp; 🧠 &nbsp; **INFERENCE ENGINE**

</td>
<td>

- Model-specific chat templates: Phi-3 uses `<|system|>` · Mistral uses `[INST]`
- True token counting via `llm.tokenize()` — not crude `split()`
- Buffer window test → sliding 3-turn memory
- Summary memory test → string-based, eliminates double inference bug
- Chain-of-thought → regex final answer extraction

</td>
</tr>

<tr>
<td>

**`CELL 4`** &nbsp; 🏁 &nbsp; **BENCHMARK + PLOTS**

</td>
<td>

- Per-model `try/except` — one failure never aborts the full run
- Incremental JSON checkpoint saved after every model completes
- Tidy `pd.DataFrame` → `metrics.csv` export
- 4 seaborn comparison plots: latency by test, token distribution, stacked time, scatter

</td>
</tr>

<tr>
<td>

**`CELL 5`** &nbsp; 🎨 &nbsp; **GRADIO LIVE DEMO**

</td>
<td>

- Stateful `DemoSession` — model loads once, reused across all messages
- 4 interactive test modes switchable mid-conversation
- Real-time metrics panel: latency · tokens · tokens/sec
- Pre-run benchmark results and plots embedded in accordion
- Auto port scanner (7860–7870) + `share=True` → instant public URL

</td>
</tr>
</table>
</div>

<br/>

---

## 📊 Metrics Captured

Every inference records:

| Metric | Method | Notes |
|:-------|:-------|:------|
| `latency_seconds` | `time.time()` wall clock | Full round-trip including tokenization |
| `prompt_tokens` | `llm.tokenize(prompt)` | True subword count, not word split |
| `response_tokens` | `llm.tokenize(response)` | True subword count |
| `tokens_per_sec` | `response_tokens / latency` | Derived at display time |
| `extracted_answer` | Regex on CoT response | Numeric/fraction extraction |
| `buffer_depth` | `len(history) // 2` | Active turns in sliding window |
| `summary_length` | `len(summary_string)` | Characters of running summary |

<br/>

---

## 🔬 Bugs Fixed From Original Notebook

<div align="center">

| # | Original Bug | Fix Applied |
|:--|:-------------|:------------|
| 1 | `llama-cpp-python` installed **twice** | Single clean install in Cell 1 |
| 2 | `len(text.split())` for token count | `llm.tokenize()` — true subword tokens |
| 3 | Missing `PYTHONHASHSEED` in seed setup | Added alongside `random` + `numpy` in Cell 2 |
| 4 | Hardcoded `/content/models/` (Colab only) | Env-var fallback, Kaggle-compatible |
| 5 | No error handling in model loader | Per-model `try/except`, skips on failure |
| 6 | No download resumption on interruption | Range requests + `.part` file kept |
| 7 | 2× inference per summary turn | String-based summary — no second LLM call |
| 8 | Wrong chat templates for both models | Phi-3 `<\|system\|>` · Mistral `[INST]` |
| 9 | All output to stdout only (lost on restart) | JSON + CSV persisted after every model |
| 10 | Zero visualizations | 4 seaborn/matplotlib comparison charts |

</div>

<br/>

---

## 🚀 Quick Start

### Option A — Kaggle (recommended, free GPU T4)

```bash
# 1. New Kaggle notebook → Accelerator: GPU T4 x2
# 2. Paste Cells 1–5 in order
# 3. Run All
# 4. Cell 5 prints a public share=True link
```

### Option B — HuggingFace Spaces (free, always-on)

```
New Space → SDK: Gradio → paste all 5 cells into app.py
```

```txt
# requirements.txt
llama-cpp-python
gradio>=6.0
numpy
pandas
matplotlib
seaborn
tqdm
```

### Option C — Local (NVIDIA GPU required)

```bash
git clone https://github.com/YOUR_USERNAME/llm-quantization-benchmark
cd llm-quantization-benchmark

pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
pip install gradio numpy pandas matplotlib seaborn tqdm

export PYTHONHASHSEED=42
export MODEL_DIR=./models
python main.py
```

<br/>

---

## 🖥️ Requirements

| Requirement | Value |
|:------------|:------|
| Python | 3.10+ |
| GPU | NVIDIA T4 / P100 (≥ 8 GB VRAM) |
| CUDA | 12.1 |
| Disk | ~14 GB (all 4 models) |
| RAM | 16 GB+ |
| Network | HuggingFace access for model download |

<br/>

---

## 📁 Output Files

```
/kaggle/working/outputs/
├── results_TIMESTAMP.json        ← Full nested benchmark results
├── metrics_TIMESTAMP.csv         ← Tidy DataFrame (one row per inference)
└── benchmark_plots_TIMESTAMP.png ← 4-panel comparison chart
```

<br/>

---

## 🧩 Tech Stack

<div align="center">

![llama-cpp-python](https://img.shields.io/badge/llama--cpp--python-GGUF%20Inference-76B900?style=flat-square&logo=nvidia)
![Gradio](https://img.shields.io/badge/Gradio-Interactive%20UI-FF7C00?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Model%20Hub-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0?style=flat-square)
![NumPy](https://img.shields.io/badge/NumPy-Numerics-013243?style=flat-square&logo=numpy)

</div>

<br/>

---

## 🌱 Reproducibility

- **Seed** — `GLOBAL_SEED = 42` applied to `random`, `numpy`, `PYTHONHASHSEED`, and llama-cpp generation config
- **Sampling note** — `temperature=0.7, top_p=0.9` introduce controlled stochasticity; seed mitigates but doesn't eliminate variance across different hardware
- **Checkpoints** — results saved after every model; a kernel crash loses at most one model's data
- **Downloads** — `.part` files kept on interruption; re-run to resume without re-downloading from zero

<br/>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12&height=120&section=footer" width="100%"/>

**Built for a college ML portfolio &nbsp;·&nbsp; 100% open-source &nbsp;·&nbsp; No API keys required**

<br/>

⭐ **Star this repo if it helped you understand LLM quantization!**

</div>
