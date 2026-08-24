# Phishing Website Detector — Web App

## What's here
```
phishing-app/
├── backend/
│   ├── main.py                # FastAPI app (loads model, exposes /predict)
│   ├── feature_extractor.py   # Computes the 30 UCI-style features from a live URL
│   ├── requirements.txt
│   └── model/                 # <-- Saved model files here
└── frontend/
    └── index.html             # Single-file UI (paste a URL, get a verdict)
```

## Setup

1. **Copy your trained model in.**
   From your notebook, you already ran:
   ```python
   joblib.dump(best_model, "phishing_model.pkl")
   joblib.dump(scaler, "scaler.pkl")
   ```
   Download both files from Colab and place them at:
   ```
   backend/model/phishing_model.pkl
   backend/model/scaler.pkl
   ```

2. **Install backend dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Run the backend**
   ```bash
   uvicorn main:app --reload --port 8000
   ```
   Check it's alive: open http://localhost:8000/health — should show `"model_loaded": true`.

4. **Open the frontend**
   Just open `frontend/index.html` directly in a browser (or serve it with
   `python -m http.server` from the `frontend/` folder). It calls the API at
   `http://localhost:8000` — change `API_BASE` in the `<script>` tag if you
   deploy the backend elsewhere.

5. **Try it**
   Paste a URL into the scan bar and hit Scan. You'll see the 30 checks light
   up, then a verdict card with the phishing/legitimate probability split.

## How the live pipeline differs from the notebook

Your notebook trained on **pre-extracted feature rows** from a static CSV/ARFF —
someone else had already computed `SSLfinal_State`, `URL_of_Anchor`, etc. for
each sample. A real app needs to compute those same 30 features **at request
time** from just a URL. That's what `feature_extractor.py` does:

- Fetches the page (`requests` + `BeautifulSoup`) to check HTML structure —
  anchor tags, forms, iframes, favicon, external resource ratios, etc.
- Uses `python-whois` for domain age / registration length.
- Uses `dnspython` to confirm a DNS record exists.
- Parses the URL string itself for IP-address hosts, `@` symbols, hyphens,
  shortening services, subdomain depth, etc.

**Four features are approximated**, because the free services the original
2015 dataset relied on no longer exist or now require paid APIs:
`web_traffic` (Alexa rank), `Page_Rank` (Google PageRank), `Google_Index`,
and `Links_pointing_to_page` (backlink count). These are defaulted to neutral
values and flagged in the API response's `warnings` field — the model's
verdict leans on the other 26 live-computed signals. If you want real values
for these, you'd wire in something like a backlink API (Ahrefs/Moz) or a
phishing blocklist API (PhishTank, Google Safe Browsing) — happy to help
with that next.

## Suggested next steps
- Add response caching by domain (repeated scans of the same site shouldn't
  re-fetch/re-WHOIS every time).
- Wire in Google Safe Browsing API as a second opinion / fallback signal.
- Deploy the backend (Render/Railway/Fly.io) and update `API_BASE` in the
  frontend, or serve the frontend from FastAPI itself with `StaticFiles`.
