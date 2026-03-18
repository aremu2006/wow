"""
app.py  —  Malicious URL Detector  (Streamlit UI)
==================================================
Run:  streamlit run app.py

Features:
 • Single URL analysis with detailed signal breakdown
 • Batch URL analysis (paste multiple URLs)
 • Model performance metrics display
 • Real-time probability gauge
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, accuracy_score
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from src.feature_extraction import extract_features, FEATURE_COLUMNS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Malicious URL Detector",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-header {
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(135deg, #1e3a5f, #e74c3c);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
  }
  .subtitle { color: #6c757d; font-size: 1rem; margin-bottom: 1.5rem; }
  .verdict-safe {
    background: #d4edda; border: 1px solid #28a745; border-radius: 10px;
    padding: 1.2rem 1.5rem; text-align: center;
  }
  .verdict-malicious {
    background: #f8d7da; border: 1px solid #dc3545; border-radius: 10px;
    padding: 1.2rem 1.5rem; text-align: center;
  }
  .verdict-suspicious {
    background: #fff3cd; border: 1px solid #ffc107; border-radius: 10px;
    padding: 1.2rem 1.5rem; text-align: center;
  }
  .signal-bad  { color: #dc3545; font-size: 0.9rem; }
  .signal-good { color: #28a745; font-size: 0.9rem; }
  .metric-card {
    background: #f8f9fa; border-radius: 8px; padding: 1rem;
    border: 1px solid #dee2e6; text-align: center;
  }
</style>
""", unsafe_allow_html=True)


# ── Load / train model ─────────────────────────────────────────────────────────
MODEL_PATH = "models/best_model.joblib"

@st.cache_resource(show_spinner="Training model …")
def load_model():
    if not os.path.exists(MODEL_PATH):
        _auto_train()
    payload = joblib.load(MODEL_PATH)
    return payload["model"], payload["feature_columns"]


def _auto_train():
    """Auto-generate dataset & train if model file missing."""
    if not os.path.exists("data/urls_dataset.csv"):
        exec(open("data/generate_dataset.py").read())
    # Run training
    os.system("python src/train_model.py")


# ── Prediction ────────────────────────────────────────────────────────────────

BRAND_KEYWORDS = [
    'paypal','google','apple','microsoft','amazon','facebook',
    'instagram','netflix','ebay','steam','whatsapp','youtube',
    'dropbox','icloud','twitter','chase','wellsfargo','citibank',
]
TRUSTED_DOMAINS_SET = {
    'google.com','youtube.com','facebook.com','microsoft.com','apple.com',
    'amazon.com','github.com','twitter.com','linkedin.com','wikipedia.org',
    'instagram.com','netflix.com','stackoverflow.com','reddit.com','paypal.com',
}
SUSPICIOUS_TLDS = {
    'xyz','top','click','tk','ml','ga','cf','gq','pw','cc','su','biz',
    'info','online','site','live','stream','download','loan','win','ru',
}

def predict(model, feat_cols, url: str) -> dict:
    import re
    from urllib.parse import urlparse

    raw = url.strip()
    feats = extract_features(raw)

    # Fill missing domain_age
    if feats.get('domain_age_days', -1) == -1:
        feats['domain_age_days'] = 365

    X = np.array([feats.get(c, 0) for c in feat_cols]).reshape(1, -1)
    prob   = model.predict_proba(X)[0]
    label  = model.predict(X)[0]

    safe_pct = round(prob[0] * 100, 1)
    mal_pct  = round(prob[1] * 100, 1)

    if label == 1:
        verdict = "MALICIOUS"
    elif mal_pct >= 30:
        verdict = "SUSPICIOUS"
    else:
        verdict = "SAFE"

    # Build signal list
    signals = []
    try:
        parsed   = urlparse(raw if '://' in raw else 'http://' + raw)
        hostname = (parsed.hostname or '').lower()
        path     = parsed.path or ''
        tld      = hostname.split('.')[-1] if '.' in hostname else ''
        parts    = hostname.replace('www.', '').split('.')
        base     = '.'.join(parts[-2:]) if len(parts) >= 2 else hostname
    except Exception:
        hostname = path = tld = base = ''

    if feats.get('is_https'): signals.append(("✅ Uses HTTPS", "good"))
    else:                     signals.append(("⚠️ No HTTPS (insecure)", "bad"))
    if feats.get('is_ip_host'): signals.append(("⚠️ IP address used as host", "bad"))
    if feats.get('suspicious_tld'): signals.append((f"⚠️ Suspicious TLD (.{tld})", "bad"))
    if feats.get('brand_in_domain'): signals.append(("⚠️ Brand impersonation detected", "bad"))
    if feats.get('digit_in_word'): signals.append(("⚠️ Typosquatting (digits in word)", "bad"))
    if feats.get('phish_path_kw'): signals.append(("⚠️ Phishing keywords in path", "bad"))
    if feats.get('is_shortener'): signals.append(("⚠️ URL shortener (hides destination)", "bad"))
    if feats.get('at_sign'): signals.append(("⚠️ '@' symbol in URL (misleading)", "bad"))
    if feats.get('has_punycode'): signals.append(("⚠️ Punycode/IDN homograph attack", "bad"))
    if feats.get('executable_ext'): signals.append(("⚠️ Executable file extension", "bad"))
    if feats.get('trusted_domain'): signals.append(("✅ Registered trusted domain", "good"))
    if feats.get('subdomain_count', 0) >= 3:
        signals.append((f"⚠️ Deep subdomain nesting ({feats['subdomain_count']} levels)", "bad"))
    if feats.get('hyphen_count', 0) >= 3:
        signals.append((f"⚠️ Many hyphens in domain ({feats['hyphen_count']})", "bad"))
    if feats.get('url_length', 0) > 100:
        signals.append((f"⚠️ Very long URL ({feats['url_length']} chars)", "bad"))
    if feats.get('spam_keyword_count', 0) >= 2:
        signals.append((f"⚠️ Spam keywords detected ({feats['spam_keyword_count']})", "bad"))
    if feats.get('percent_count', 0) > 5:
        signals.append((f"⚠️ Heavy URL encoding ({feats['percent_count']} chars)", "bad"))

    return {
        "url": raw, "verdict": verdict,
        "safe_pct": safe_pct, "mal_pct": mal_pct,
        "signals": signals, "features": feats,
    }


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/cyber-security.png", width=80)
    st.markdown("## 🔐 URL Detector")
    st.markdown("---")
    st.markdown("""
**How it works:**
1. Extract 35 URL features
2. Random Forest classifier
3. Trained on 200+ URLs
4. Returns threat probability

**Feature categories:**
- Protocol & structure
- Domain & TLD analysis
- Brand/typosquat detection
- Path & query patterns
- Entropy signals
    """)
    st.markdown("---")
    st.markdown("**Verdict thresholds:**")
    st.success("🟢 SAFE: < 30% malicious")
    st.warning("🟡 SUSPICIOUS: 30–49%")
    st.error("🔴 MALICIOUS: ≥ 50%")
    st.markdown("---")
    st.caption("Built with Python, scikit-learn & Streamlit")


# ── Main content ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🔐 Malicious URL Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered threat analysis using Machine Learning — paste any URL below</div>', unsafe_allow_html=True)

model, feat_cols = load_model()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Single URL Analysis", "📋 Batch Analysis", "📊 Model Performance"])


# ────────────────────────────────────────────
# TAB 1 — SINGLE URL
# ────────────────────────────────────────────
with tab1:
    st.markdown("### Analyse a URL")
    url_input = st.text_input(
        "Enter URL:",
        placeholder="https://example.com or paste a suspicious link ...",
        key="single_url"
    )

    col_btn, col_ex = st.columns([1, 3])
    with col_btn:
        analyse_clicked = st.button("🔍 Analyse", type="primary", use_container_width=True)
    with col_ex:
        examples = {
            "Google (safe)": "https://www.google.com/search?q=test",
            "Paypal phish": "http://paypal.com.secure-login-verify.xyz/account/update",
            "IP login": "http://192.168.0.1/admin/login.php",
            "Prize scam": "http://free-iphone-winner.top/claim?id=99812",
            "GitHub (safe)": "https://github.com/python/cpython",
        }
        ex_choice = st.selectbox("Or try an example:", ["— pick one —"] + list(examples.keys()))
        if ex_choice != "— pick one —":
            url_input = examples[ex_choice]

    if analyse_clicked or (ex_choice != "— pick one —"):
        if url_input:
            with st.spinner("Analysing URL …"):
                time.sleep(0.3)
                result = predict(model, feat_cols, url_input)

            # Verdict banner
            v = result['verdict']
            cls = "verdict-safe" if v == "SAFE" else \
                  "verdict-suspicious" if v == "SUSPICIOUS" else "verdict-malicious"
            icon = "🟢" if v == "SAFE" else "🟡" if v == "SUSPICIOUS" else "🔴"

            st.markdown(f"""
            <div class="{cls}">
              <h2 style="margin:0">{icon} {v}</h2>
              <p style="margin:0.3rem 0 0">{url_input[:90]}{'...' if len(url_input)>90 else ''}</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")

            # Probability bars
            col1, col2 = st.columns(2)
            with col1:
                st.metric("✅ Safe probability",   f"{result['safe_pct']}%")
                st.progress(int(result['safe_pct']))
            with col2:
                st.metric("🔴 Malicious probability", f"{result['mal_pct']}%")
                st.progress(int(result['mal_pct']))

            st.markdown("---")
            st.markdown("#### 🔎 Signal Breakdown")
            sig_cols = st.columns(2)
            for i, (sig, kind) in enumerate(result['signals']):
                with sig_cols[i % 2]:
                    if kind == "good":
                        st.markdown(f'<span class="signal-good">{sig}</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="signal-bad">{sig}</span>', unsafe_allow_html=True)

            with st.expander("📐 Full Feature Vector"):
                feat_df = pd.DataFrame(
                    result['features'].items(), columns=["Feature", "Value"]
                ).set_index("Feature")
                st.dataframe(feat_df, use_container_width=True)
        else:
            st.warning("Please enter a URL to analyse.")


# ────────────────────────────────────────────
# TAB 2 — BATCH ANALYSIS
# ────────────────────────────────────────────
with tab2:
    st.markdown("### Batch URL Analysis")
    st.markdown("Paste one URL per line:")

    batch_input = st.text_area(
        "URLs:",
        height=180,
        placeholder="https://google.com\nhttp://free-prize.xyz/claim\nhttps://github.com/...",
        key="batch_urls"
    )

    if st.button("🔍 Analyse All", type="primary"):
        urls = [u.strip() for u in batch_input.splitlines() if u.strip()]
        if not urls:
            st.warning("Please enter at least one URL.")
        else:
            with st.spinner(f"Analysing {len(urls)} URLs …"):
                rows = []
                for u in urls:
                    r = predict(model, feat_cols, u)
                    rows.append({
                        "URL": u[:70] + ("..." if len(u) > 70 else ""),
                        "Verdict": r['verdict'],
                        "Safe %": r['safe_pct'],
                        "Malicious %": r['mal_pct'],
                    })

            df_result = pd.DataFrame(rows)

            def colour_verdict(val):
                if val == "MALICIOUS":   return "background-color:#f8d7da; color:#721c24"
                if val == "SUSPICIOUS":  return "background-color:#fff3cd; color:#856404"
                return "background-color:#d4edda; color:#155724"

            st.dataframe(
                df_result.style.applymap(colour_verdict, subset=["Verdict"]),
                use_container_width=True, height=350
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("🟢 Safe",       (df_result['Verdict'] == "SAFE").sum())
            c2.metric("🟡 Suspicious", (df_result['Verdict'] == "SUSPICIOUS").sum())
            c3.metric("🔴 Malicious",  (df_result['Verdict'] == "MALICIOUS").sum())

            csv_out = df_result.to_csv(index=False)
            st.download_button("⬇️ Download CSV", csv_out,
                               file_name="url_analysis_results.csv", mime="text/csv")


# ────────────────────────────────────────────
# TAB 3 — MODEL PERFORMANCE
# ────────────────────────────────────────────
with tab3:
    st.markdown("### Model Performance Dashboard")

    @st.cache_data(show_spinner="Loading performance metrics …")
    def get_metrics():
        from data.generate_dataset import BENIGN, MALICIOUS
        urls  = BENIGN + MALICIOUS
        labels = [0] * len(BENIGN) + [1] * len(MALICIOUS)
        feats  = [extract_features(u) for u in urls]
        X = pd.DataFrame(feats)[feat_cols].fillna(0).values.astype(float)
        y = np.array(labels)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2,
                                                stratify=y, random_state=42)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        return y_test, y_pred, y_prob

    y_test, y_pred, y_prob = get_metrics()

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    rep  = classification_report(y_test, y_pred,
                                 target_names=['Benign', 'Malicious'],
                                 output_dict=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{acc:.3f}")
    c2.metric("ROC-AUC",   f"{auc:.3f}")
    c3.metric("Precision", f"{rep['Malicious']['precision']:.3f}")
    c4.metric("Recall",    f"{rep['Malicious']['recall']:.3f}")

    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.markdown("##### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(4, 3.5))
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=['Benign', 'Malicious'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title("Confusion Matrix")
        st.pyplot(fig, use_container_width=True)

    with col_roc:
        st.markdown("##### ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig2, ax2 = plt.subplots(figsize=(4, 3.5))
        ax2.plot(fpr, tpr, color='steelblue', label=f"AUC = {auc:.3f}")
        ax2.plot([0, 1], [0, 1], 'k--', lw=0.8)
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curve")
        ax2.legend()
        st.pyplot(fig2, use_container_width=True)

    st.markdown("##### Feature Importance (Top 15)")
    if hasattr(model, 'named_steps') and 'clf' in model.named_steps:
        clf = model.named_steps['clf']
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            fi = sorted(zip(feat_cols, importances), key=lambda x: -x[1])[:15]
            names_fi, vals_fi = zip(*fi)
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            ax3.barh(list(names_fi)[::-1], list(vals_fi)[::-1], color='steelblue')
            ax3.set_xlabel("Importance Score")
            ax3.set_title("Top 15 Feature Importances")
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)

    st.markdown("##### Full Classification Report")
    report_df = pd.DataFrame(rep).T.round(4)
    st.dataframe(report_df, use_container_width=True)
