# Quant Toolkit

A personal quantitative analysis workbench. Built to be extended.

Current tools:
- **Stock Analysis** — historical price, returns, distribution diagnostics, risk metrics, drawdowns, rolling stats for a single ticker.

Coming next:
- Portfolio Optimizer (mean-variance)
- Risk Model (VaR, factor exposures, stress tests)
- Backtesting Engine

---

## One-time setup on macOS

You need Python 3.10+ installed. macOS ships with Python but it's usually old, so use Homebrew.

**1. Install Homebrew if you don't have it.** Paste this in Terminal:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**2. Install Python:**

```bash
brew install python@3.12
```

**3. Put this project somewhere clean.** I'd suggest `~/quant/`:

```bash
mkdir -p ~/quant
# extract the zip you got into ~/quant so files end up at ~/quant/app.py, ~/quant/src/, etc.
cd ~/quant
```

**4. Create a virtual environment and install dependencies.** A virtual env is just an isolated Python install that lives inside the project folder. Avoids polluting your system Python and lets different projects use different package versions.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You should see `(.venv)` at the start of your terminal prompt while it's active. Run `deactivate` to leave it. Any time you reopen Terminal to use this project, run `source .venv/bin/activate` first.

**5. Configure your Alpha Vantage key:**

```bash
cp .env.example .env
# open .env in any editor and paste your AV key
```

Get a fresh key at https://www.alphavantage.co/support/#api-key. The one you pasted in chat should be rotated.

---

## Running

From the project folder with the venv activated:

```bash
streamlit run app.py
```

A browser tab opens at http://localhost:8501. Use the sidebar to navigate between pages. Stop the server with `Ctrl+C` in Terminal.

That's the only terminal command you'll regularly use.

---

## Project layout

```
quant/
├── app.py                          # Streamlit home page (entry point)
├── pages/                          # Each file is a sidebar nav item
│   └── 1_📈_Stock_Analysis.py
├── src/                            # Reusable Python modules
│   ├── data.py                     # Data providers, cache
│   └── analysis.py                 # Returns, risk metrics, drawdowns, VaR
├── tests/                          # Unit tests (pytest)
│   └── test_analysis.py
├── cache/                          # Parquet cache of historical data (gitignored)
├── .streamlit/config.toml          # Theme
├── .env                            # Your API keys (gitignored, you create this)
├── .env.example                    # Template showing required vars
├── requirements.txt
└── .gitignore
```

**Mental model:**
- `src/` is the library. No Streamlit code in there. You could import these functions from a Jupyter notebook, a CLI script, or future tools.
- `pages/` is the UI. Each page is a thin script that imports from `src/` and adds Streamlit widgets.
- This separation means when we build the optimizer next, the data layer and risk metrics are already there to use.

---

## Running tests

```bash
pytest tests/ -v
```

10 tests currently. Run these before changing anything in `src/analysis.py` to make sure you haven't broken existing behavior.

---

## Adding new pages

Drop a new file in `pages/`. The filename convention is `<number>_<emoji>_<Name>.py`:

```
pages/2_⚖️_Portfolio_Optimizer.py
pages/3_📉_Risk_Model.py
pages/4_🔁_Backtester.py
```

The number controls sidebar order. The emoji is optional but nice.

---

## When yfinance breaks

It will. Yahoo periodically changes their endpoints. When it happens:

```bash
pip install --upgrade yfinance
```

Usually fixes it within a few days of the break.

If you need a more reliable data source long-term, look at:
- IEX Cloud (paid, but cheap)
- Polygon.io (paid)
- StFX library subscriptions (Bloomberg, Refinitiv) if available to you as a research assistant

---

## Git

If you want version control (recommended):

```bash
git init
git add .
git commit -m "Initial commit"
# create private repo on github, then:
git remote add origin <your-repo-url>
git push -u origin main
```

`.gitignore` already excludes secrets and cache, so this is safe to push to a private repo.
