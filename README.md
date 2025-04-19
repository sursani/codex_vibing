# AI‑Powered Stock Analysis

**AI Stock Analyzer** is a tiny showcase project that combines the OpenAI
Agents SDK, *yfinance*, and Streamlit to produce quick, human‑readable
financial insights for any publicly‑traded company.

The project ships with **two independent front‑ends** that reuse the exact
same business logic:

1. **Command‑line interface** – ideal for scripts and pipelines.
2. **Streamlit dashboard** – a point‑and‑click web UI complete with price
   charts.

Under the hood the heavy lifting lives in
[`stock_analysis_lib.py`](stock_analysis_lib.py):

* fetch the latest stock price via an OpenAI agent that is allowed to use a
  web‑search tool;
* retrieve trailing & forward P/E ratios with *yfinance*;
* ask an LLM to comment on the company's outlook and return structured JSON;
* **NEW**: Added a quality review section with a verdict on the analysis quality.

---

## Installation

```bash
# 1. Clone the repo and enter it
git clone <this‑repo‑url> stock‑analysis
cd stock‑analysis

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt
```

### OpenAI credentials

Both applications expect `OPENAI_API_KEY` to be available in the environment.
You can either export it directly…

```bash
export OPENAI_API_KEY="sk‑..."
```

…or place it in a local **`.env`** file next to the app entry point – the
`python‑dotenv` helper shipped in `requirements.txt` will load it for you
automatically:

```ini
# .env
OPENAI_API_KEY=sk‑...
```

---

## 1 — Run the CLI

The CLI prints a rich JSON object that you can pipe to `jq`, store in a file,
etc.

```bash
python stock_analysis_app.py AAPL

{
  "price": {
    "ticker": "AAPL",
    "price": 215.23
  },
  "pe": {
    "ticker": "AAPL",
    "trailingPE": 28.10,
    "forwardPE": 25.58
  },
  "analysis": {
    "ticker": "AAPL",
    "price": 215.23,
    "trailingPE": 28.10,
    "forwardPE": 25.58,
    "analysis": "Apple's premium valuation reflects its..."
  },
  "judge": {
    "quality": "excellent",
    "comments": "The analysis is thorough and well-supported by data."
  }
}
```

Optional flags:

* `--model gpt‑4o-mini` – override the default LLM model.

---

## 2 — Run the Streamlit app

```bash
streamlit run streamlit_stock_app.py
```

The dashboard lets you:

* type a ticker symbol and kick off the analysis;
* choose the historical period/interval shown in the line chart;
* optionally override the LLM model.

It displays 🔹 key price & P/E metrics, 🔹 an interactive Plotly chart, 🔹
the LLM's commentary, and 🔹 a quality review of the analysis with a verdict and comments. A "Raw JSON output" expander at the bottom is handy for
debugging.

---

## Folder overview

```
├── stock_analysis_lib.py      # Shared async helpers & agent definitions
├── stock_analysis_app.py      # CLI entry point (uses the library above)
├── streamlit_stock_app.py     # Streamlit UI (also reuses the library)
├── requirements.txt           # Runtime dependencies
└── README.md                  # ← you are here
```

Have fun and feel free to adapt the agent prompts/tools to your own use‑case!
