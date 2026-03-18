"""
app.py  —  Malicious URL Detector
===================================
Simple, clean Streamlit frontend.
Links to:  src/feature_extraction.py  |  src/train_model.py  |  data/generate_dataset.py

Run:  streamlit run app.py
"""

import os, sys, time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, accuracy_score
)
from sklearn.model_selection import train_test_split
import streamlit as st

# ── Robust path setup: works locally AND on Streamlit Cloud ──────────────────
# Walks up from __file__ to find whichever ancestor contains the src/ folder
_HERE = os.path.dirname(os.path.abspath(__file__))
for _root in [_HERE, os.path.dirname(_HERE), os.getcwd()]:
    if os.path.isdir(os.path.join(_root, "src")):
        sys.path.insert(0, _root)
        os.chdir(_root)          # also fix relative paths (models/, data/)
        break

from src.feature_extraction import extract_features, FEATURE_COLUMNS   # ← feature_extraction.py

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Malicious URL Detector",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Fonts & base ── */
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

  /* ── Page background ── */
  .stApp { background: #f5f6f8; }

  /* ── Top title bar ── */
  .top-bar {
    background: #1a2236;
    border-radius: 12px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 12px;
  }
  .top-bar h1 {
    color: #ffffff; font-size: 1.4rem; font-weight: 600;
    margin: 0; font-family: 'IBM Plex Sans', sans-serif;
  }
  .top-bar span { color: #64b5f6; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }

  /* ── Cards ── */
  .card {
    background: #ffffff;
    border: 1px solid #e2e6ea;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
  }
  .card-title {
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #7a869a; margin-bottom: 1rem;
    font-family: 'IBM Plex Mono', monospace;
  }

  /* ── Verdict banners ── */
  .verdict {
    border-radius: 10px; padding: 1.1rem 1.4rem;
    margin: 1rem 0; border-left: 5px solid;
    display: flex; align-items: center; gap: 14px;
  }
  .verdict-safe       { background: #f0faf4; border-color: #27ae60; }
  .verdict-suspicious { background: #fffbf0; border-color: #f39c12; }
  .verdict-malicious  { background: #fff5f5; border-color: #e74c3c; }
  .verdict-icon { font-size: 1.8rem; line-height: 1; }
  .verdict-body { flex: 1; }
  .verdict-label {
    font-size: 1.2rem; font-weight: 600;
    font-family: 'IBM Plex Sans', sans-serif;
  }
  .verdict-safe       .verdict-label { color: #1e8449; }
  .verdict-suspicious .verdict-label { color: #b7770d; }
  .verdict-malicious  .verdict-label { color: #c0392b; }
  .verdict-url {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem; color: #7a869a; margin-top: 3px;
    word-break: break-all;
  }
  .verdict-pct {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem; font-weight: 600; text-align: right;
  }
  .verdict-safe       .verdict-pct { color: #27ae60; }
  .verdict-suspicious .verdict-pct { color: #f39c12; }
  .verdict-malicious  .verdict-pct { color: #e74c3c; }

  /* ── Signal chips ── */
  .signals-wrap { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 0.6rem; }
  .sig-chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 12px; border-radius: 20px;
    font-size: 0.82rem; font-family: 'IBM Plex Mono', monospace;
  }
  .sig-bad  { background: #fdedec; color: #c0392b; border: 1px solid #f5b7b1; }
  .sig-good { background: #eafaf1; color: #1e8449; border: 1px solid #a9dfbf; }

  /* ── Prob bars ── */
  .prob-row { margin-bottom: 0.8rem; }
  .prob-label {
    display: flex; justify-content: space-between;
    font-size: 0.82rem; margin-bottom: 4px; color: #4a5568;
    font-family: 'IBM Plex Mono', monospace;
  }
  .bar-track {
    background: #e9ecef; border-radius: 99px; height: 9px; overflow: hidden;
  }
  .bar-fill { height: 100%; border-radius: 99px; }
  .bar-safe      { background: #27ae60; }
  .bar-malicious { background: #e74c3c; }

  /* ── Metric boxes ── */
  .metric-box {
    background: #ffffff; border: 1px solid #e2e6ea;
    border-radius: 10px; padding: 1rem; text-align: center;
  }
  .metric-val { font-size: 1.6rem; font-weight: 600; color: #1a2236; font-family: 'IBM Plex Mono', monospace; }
  .metric-lbl { font-size: 0.72rem; color: #7a869a; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 3px; }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] { background: #1a2236 !important; }
  section[data-testid="stSidebar"] * { color: #c9d6e8 !important; }
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 { color: #ffffff !important; }
  section[data-testid="stSidebar"] hr  { border-color: #2e3f5c !important; }

  /* ── Buttons ── */
  .stButton > button {
    background: #1a2236 !important; color: #ffffff !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important; padding: 0.55rem 1.4rem !important;
  }
  .stButton > button:hover { background: #26344f !important; }

  /* ── Inputs ── */
  .stTextInput > div > input, .stTextArea > div > textarea {
    border: 1px solid #d0d7e2 !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.88rem !important;
    background: #ffffff !important;
  }
  .stTextInput > div > input:focus, .stTextArea > div > textarea:focus {
    border-color: #2196f3 !important;
    box-shadow: 0 0 0 3px rgba(33,150,243,0.12) !important;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #ffffff; border-radius: 10px;
    padding: 4px; border: 1px solid #e2e6ea;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 7px !important; padding: 8px 20px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important; font-size: 0.88rem !important;
    color: #7a869a !important;
  }
  .stTabs [aria-selected="true"] {
    background: #1a2236 !important; color: #ffffff !important;
  }
  .stTabs [data-baseweb="tab-panel"] { padding-top: 1.2rem !important; }

  /* ── Download button ── */
  .stDownloadButton > button {
    background: transparent !important; color: #1a2236 !important;
    border: 1px solid #d0d7e2 !important; font-weight: 500 !important;
  }
  .stDownloadButton > button:hover { background: #f5f6f8 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD MODEL  ←  uses models/best_model.joblib (from train_model.py)
# ─────────────────────────────────────────────────────────────
MODEL_PATH = "models/best_model.joblib"

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Model not found — training now (takes ~30 sec)…"):
            if not os.path.exists("data/urls_dataset.csv"):
                os.system("python data/generate_dataset.py")   # ← generate_dataset.py
            os.system("python src/train_model.py")              # ← train_model.py
    payload = joblib.load(MODEL_PATH)
    return payload["model"], payload["feature_columns"]

model, feat_cols = load_model()


# ─────────────────────────────────────────────────────────────
# PREDICT  ←  uses src/feature_extraction.py
# ─────────────────────────────────────────────────────────────
def predict(url: str) -> dict:
    from urllib.parse import urlparse
    raw   = url.strip()
    feats = extract_features(raw)                              # ← feature_extraction.py
    if feats.get("domain_age_days", -1) == -1:
        feats["domain_age_days"] = 365

    X     = np.array([feats.get(c, 0) for c in feat_cols]).reshape(1, -1)
    prob  = model.predict_proba(X)[0]
    label = int(model.predict(X)[0])

    safe_pct = round(prob[0] * 100, 1)
    mal_pct  = round(prob[1] * 100, 1)
    verdict  = "MALICIOUS" if label == 1 else "SUSPICIOUS" if mal_pct >= 30 else "SAFE"

    try:
        parsed   = urlparse(raw if "://" in raw else "http://" + raw)
        hostname = (parsed.hostname or "").lower()
        path     = parsed.path or ""
        tld      = hostname.split(".")[-1] if "." in hostname else ""
    except Exception:
        hostname = path = tld = ""

    signals = []
    if feats.get("is_https"):        signals.append(("✅ Uses HTTPS", "good"))
    else:                            signals.append(("⚠️ No HTTPS", "bad"))
    if feats.get("is_ip_host"):      signals.append(("⚠️ IP address as host", "bad"))
    if feats.get("suspicious_tld"):  signals.append((f"⚠️ Suspicious TLD (.{tld})", "bad"))
    if feats.get("brand_in_domain"): signals.append(("⚠️ Brand impersonation", "bad"))
    if feats.get("digit_in_word"):   signals.append(("⚠️ Typosquatting detected", "bad"))
    if feats.get("phish_path_kw"):   signals.append(("⚠️ Phishing keywords in path", "bad"))
    if feats.get("is_shortener"):    signals.append(("⚠️ URL shortener used", "bad"))
    if feats.get("at_sign"):         signals.append(("⚠️ @ symbol in URL", "bad"))
    if feats.get("has_punycode"):    signals.append(("⚠️ Punycode / IDN attack", "bad"))
    if feats.get("executable_ext"):  signals.append(("⚠️ Executable file extension", "bad"))
    if feats.get("trusted_domain"):  signals.append(("✅ Trusted domain", "good"))
    if feats.get("subdomain_count", 0) >= 3:
        signals.append((f"⚠️ Deep subdomains ({feats['subdomain_count']})", "bad"))
    if feats.get("hyphen_count", 0) >= 3:
        signals.append((f"⚠️ Many hyphens ({feats['hyphen_count']})", "bad"))
    if feats.get("url_length", 0) > 100:
        signals.append((f"⚠️ Long URL ({feats['url_length']} chars)", "bad"))
    if feats.get("spam_keyword_count", 0) >= 2:
        signals.append((f"⚠️ Spam keywords ({feats['spam_keyword_count']})", "bad"))
    if not signals:
        signals.append(("✅ No suspicious signals found", "good"))

    return {
        "url": raw, "verdict": verdict,
        "safe_pct": safe_pct, "mal_pct": mal_pct,
        "signals": signals, "features": feats,
    }


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔐 URL Detector")
    st.markdown("---")
    st.markdown("""
**Project files linked:**
- `src/feature_extraction.py`
- `src/train_model.py`
- `data/generate_dataset.py`
- `models/best_model.joblib`

---

**Verdict guide:**
""")
    st.success("🟢  SAFE  —  below 30%")
    st.warning("🟡  SUSPICIOUS  —  30–49%")
    st.error("🔴  MALICIOUS  —  50% +")
    st.markdown("---")
    st.caption("Python · scikit-learn · Streamlit")


# ─────────────────────────────────────────────────────────────
# TITLE BAR
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
  <span style="font-size:1.6rem">🔐</span>
  <div>
    <h1>Malicious URL Detector</h1>
    <span>Random Forest &nbsp;·&nbsp; 36 features &nbsp;·&nbsp; 98%+ accuracy</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍  Single URL", "📋  Batch Analysis", "📊  Model Performance"])


# ══════════════════════════════════════════
# TAB 1 — SINGLE URL
# ══════════════════════════════════════════
with tab1:
    col_input, col_ex = st.columns([3, 1])

    with col_input:
        url_input = st.text_input(
            "URL",
            placeholder="https://example.com or paste a suspicious link…",
            key="single_url",
            label_visibility="collapsed",
        )

    with col_ex:
        examples = {
            "— quick example —":                              "",
            "✅ google.com":                                  "https://www.google.com/search?q=test",
            "✅ github.com":                                  "https://github.com/python/cpython",
            "🔴 paypal phish":  "http://paypal.com.secure-login-verify.xyz/account/update",
            "🔴 IP login":      "http://192.168.0.1/admin/login.php",
            "🔴 prize scam":    "http://free-iphone-winner.top/claim?id=99812",
            "🔴 typosquat":     "http://g00gle-security-alert.com/verify?user=you@mail.com",
        }
        ex = st.selectbox("Example", list(examples.keys()), label_visibility="collapsed")
        if examples.get(ex):
            url_input = examples[ex]

    scan_btn = st.button("🔍  Analyse URL", key="scan")

    if scan_btn and url_input:
        with st.spinner("Scanning…"):
            time.sleep(0.25)
            result = predict(url_input)

        v    = result["verdict"]
        icon = "🟢" if v == "SAFE" else "🟡" if v == "SUSPICIOUS" else "🔴"
        cls  = f"verdict-{v.lower()}"
        short = result["url"][:90] + ("…" if len(result["url"]) > 90 else "")
        conf  = result["safe_pct"] if v == "SAFE" else result["mal_pct"]

        # ── Verdict banner ──
        st.markdown(f"""
        <div class="verdict {cls}">
          <div class="verdict-icon">{icon}</div>
          <div class="verdict-body">
            <div class="verdict-label">{v}</div>
            <div class="verdict-url">{short}</div>
          </div>
          <div class="verdict-pct">{conf}%</div>
        </div>""", unsafe_allow_html=True)

        # ── Probability bars ──
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="card">
              <div class="card-title">Safe probability</div>
              <div class="prob-label"><span>Safe</span><span>{result['safe_pct']}%</span></div>
              <div class="bar-track"><div class="bar-fill bar-safe" style="width:{result['safe_pct']}%"></div></div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="card">
              <div class="card-title">Malicious probability</div>
              <div class="prob-label"><span>Malicious</span><span>{result['mal_pct']}%</span></div>
              <div class="bar-track"><div class="bar-fill bar-malicious" style="width:{result['mal_pct']}%"></div></div>
            </div>""", unsafe_allow_html=True)

        # ── Signal chips ──
        chips = "".join(
            f'<span class="sig-chip sig-{"good" if k == "good" else "bad"}">{s}</span>'
            for s, k in result["signals"]
        )
        st.markdown(f"""
        <div class="card">
          <div class="card-title">Signal breakdown</div>
          <div class="signals-wrap">{chips}</div>
        </div>""", unsafe_allow_html=True)

        # ── Feature vector ──
        with st.expander("View full feature vector (36 features)"):
            st.dataframe(
                pd.DataFrame(result["features"].items(), columns=["Feature", "Value"]).set_index("Feature"),
                use_container_width=True, height=360,
            )

    elif scan_btn:
        st.warning("Please enter a URL first.")


# ══════════════════════════════════════════
# TAB 2 — BATCH
# ══════════════════════════════════════════
with tab2:
    st.markdown("##### Paste URLs below — one per line")
    batch_text = st.text_area(
        "batch",
        height=160,
        placeholder="https://google.com\nhttp://free-prize.xyz/claim\nhttps://github.com/user/repo",
        label_visibility="collapsed",
        key="batch_urls",
    )

    if st.button("🔍  Analyse All", key="batch_scan"):
        urls = [u.strip() for u in batch_text.splitlines() if u.strip()]
        if not urls:
            st.warning("Paste at least one URL.")
        else:
            with st.spinner(f"Analysing {len(urls)} URL{'s' if len(urls) > 1 else ''}…"):
                rows = []
                for u in urls:
                    r = predict(u)
                    rows.append({
                        "URL":         u[:80] + ("…" if len(u) > 80 else ""),
                        "Verdict":     r["verdict"],
                        "Safe %":      r["safe_pct"],
                        "Malicious %": r["mal_pct"],
                    })

            df = pd.DataFrame(rows)

            # Summary
            c1, c2, c3 = st.columns(3)
            for col, key, colour, label in [
                (c1, "SAFE",       "#27ae60", "Safe"),
                (c2, "SUSPICIOUS", "#f39c12", "Suspicious"),
                (c3, "MALICIOUS",  "#e74c3c", "Malicious"),
            ]:
                n = (df["Verdict"] == key).sum()
                col.markdown(f"""<div class="metric-box">
                  <div class="metric-val" style="color:{colour}">{n}</div>
                  <div class="metric-lbl">{label}</div></div>""", unsafe_allow_html=True)

            st.markdown("")

            def colour_row(row):
                colours = {"MALICIOUS": "#fff5f5", "SUSPICIOUS": "#fffbf0", "SAFE": "#f0faf4"}
                bg = colours.get(row["Verdict"], "#ffffff")
                return [f"background-color:{bg}"] * len(row)

            st.dataframe(
                df.style.apply(colour_row, axis=1),
                use_container_width=True, height=320,
            )

            st.download_button(
                "⬇️  Download CSV", df.to_csv(index=False),
                file_name="url_scan_results.csv", mime="text/csv",
            )


# ══════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════
with tab3:

    @st.cache_data(show_spinner="Computing metrics…")
    def get_metrics():
        from data.generate_dataset import BENIGN, MALICIOUS as MAL_URLS
        urls   = BENIGN + MAL_URLS
        labels = [0] * len(BENIGN) + [1] * len(MAL_URLS)
        feats  = [extract_features(u) for u in urls]                      # ← feature_extraction.py
        X = pd.DataFrame(feats)[feat_cols].fillna(0).values.astype(float)
        y = np.array(labels)
        _, X_t, _, y_t = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        return y_t, model.predict(X_t), model.predict_proba(X_t)[:, 1]

    y_test, y_pred, y_prob = get_metrics()
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    rep = classification_report(y_test, y_pred,
                                target_names=["Benign", "Malicious"], output_dict=True)

    # ── Metrics ──
    m1, m2, m3, m4 = st.columns(4)
    for col, val, lbl in [
        (m1, f"{acc:.3f}", "Accuracy"),
        (m2, f"{auc:.3f}", "ROC-AUC"),
        (m3, f"{rep['Malicious']['precision']:.3f}", "Precision"),
        (m4, f"{rep['Malicious']['recall']:.3f}",    "Recall"),
    ]:
        col.markdown(f"""<div class="metric-box">
          <div class="metric-val">{val}</div>
          <div class="metric-lbl">{lbl}</div></div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Confusion matrix + ROC ──
    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.markdown("##### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(4, 3.2))
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, y_pred),
            display_labels=["Benign", "Malicious"]
        ).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Confusion Matrix", fontsize=11)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col_roc:
        st.markdown("##### ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig2, ax2 = plt.subplots(figsize=(4, 3.2))
        ax2.plot(fpr, tpr, color="#1a2236", lw=2, label=f"AUC = {auc:.3f}")
        ax2.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        ax2.set_xlabel("False Positive Rate", fontsize=9)
        ax2.set_ylabel("True Positive Rate", fontsize=9)
        ax2.set_title("ROC Curve", fontsize=11)
        ax2.legend(fontsize=9)
        ax2.spines[["top", "right"]].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    # ── Feature importance ──
    st.markdown("##### Feature Importance (Top 15)")
    clf = model.named_steps.get("clf") if hasattr(model, "named_steps") else None
    if clf and hasattr(clf, "feature_importances_"):
        fi = sorted(zip(feat_cols, clf.feature_importances_), key=lambda x: -x[1])[:15]
        names_fi, vals_fi = zip(*fi)
        fig3, ax3 = plt.subplots(figsize=(8, 3.8))
        ax3.barh(list(names_fi)[::-1], list(vals_fi)[::-1], color="#1a2236", height=0.6)
        ax3.set_xlabel("Importance Score", fontsize=9)
        ax3.tick_params(labelsize=9)
        ax3.spines[["top", "right"]].set_visible(False)
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)

    # ── Full report ──
    with st.expander("Full Classification Report"):
        st.dataframe(pd.DataFrame(rep).T.round(4), use_container_width=True)
