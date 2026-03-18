"""
app.py  —  Malicious URL Detector
===================================
100% self-contained Streamlit app — no src/ imports, no path issues.
Works locally and on Streamlit Cloud unchanged.

Run:  streamlit run app.py
"""

import os, sys, re, math, time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from urllib.parse import urlparse, parse_qs
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import streamlit as st

# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION  (from src/feature_extraction.py)
# ─────────────────────────────────────────────────────────────

SUSPICIOUS_TLDS = {
    'xyz','top','click','tk','ml','ga','cf','gq','pw','cc',
    'su','biz','info','online','site','live','stream','download',
    'loan','review','country','kim','science','work','party','trade',
    'cricket','date','faith','racing','accountant','win','bid',
    'men','icu','monster','cyou','buzz','sbs','ru',
}
TRUSTED_DOMAINS = {
    'google.com','youtube.com','facebook.com','microsoft.com','apple.com',
    'amazon.com','github.com','twitter.com','linkedin.com','wikipedia.org',
    'instagram.com','netflix.com','stackoverflow.com','reddit.com','paypal.com',
    'bbc.com','nytimes.com','dropbox.com','mozilla.org','cloudflare.com',
    'medium.com','kaggle.com','huggingface.co','arxiv.org','nature.com',
    'zoom.us','slack.com','notion.so','figma.com','canva.com','stripe.com',
    'shopify.com','heroku.com','vercel.com','netlify.com',
}
BRAND_KEYWORDS = [
    'paypal','google','apple','microsoft','amazon','facebook',
    'instagram','netflix','ebay','steam','whatsapp','youtube',
    'dropbox','icloud','twitter','chase','wellsfargo','citibank',
    'bankofamerica','boa','dhl','fedex','usps','ups',
]
URL_SHORTENERS = {
    'bit.ly','tinyurl.com','t.co','goo.gl','ow.ly','is.gd',
    'buff.ly','rebrand.ly','short.io','tiny.cc','cutt.ly',
}
PHISH_RE  = re.compile(
    r'login|signin|verify|account|update|secure|confirm|'
    r'password|credential|alert|suspend|unlock|recover|'
    r'reset|billing|payment|invoice', re.I)
EXEC_RE   = re.compile(r'\.(exe|bat|cmd|msi|scr|vbs|jar|apk|dmg|sh|ps1|crx|xpi)$', re.I)
SPAM_WORDS = ['free','win','prize','claim','urgent','alert','suspended','verify',
              'confirm','limited','offer','bonus','gift','reward','lucky','congratulation']


def _entropy(s):
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


def _domain_parts(hostname):
    clean = re.sub(r'^www\.', '', hostname.lower())
    parts = clean.split('.')
    if len(parts) >= 3:
        return '.'.join(parts[:-2]), parts[-2], parts[-1]
    if len(parts) == 2:
        return '', parts[0], parts[1]
    return '', hostname, ''


def extract_features(url: str) -> dict:
    raw = str(url).strip()
    f   = {}
    try:
        p = urlparse(raw if '://' in raw else 'http://' + raw)
    except Exception:
        p = urlparse('http://invalid')

    hostname   = (p.hostname or '').lower()
    path       = p.path or ''
    query      = p.query or ''
    scheme     = p.scheme or ''
    full_lower = raw.lower()

    _, domain, tld = _domain_parts(hostname)
    base = f"{domain}.{tld}" if domain and tld else hostname
    sub, _, _ = _domain_parts(hostname)

    f['is_https']          = int(scheme == 'https')
    f['is_http']           = int(scheme == 'http')
    f['url_length']        = len(raw)
    f['hostname_length']   = len(hostname)
    f['path_length']       = len(path)
    f['query_length']      = len(query)
    f['dot_count']         = hostname.count('.')
    f['hyphen_count']      = hostname.count('-')
    f['underscore_count']  = raw.count('_')
    f['at_sign']           = int('@' in raw)
    f['double_slash']      = int('//' in path)
    f['question_mark']     = int('?' in raw)
    f['ampersand_count']   = query.count('&')
    f['equals_count']      = query.count('=')
    f['percent_count']     = len(re.findall(r'%[0-9a-fA-F]{2}', raw))
    f['hash_count']        = int('#' in raw)
    hl = max(len(hostname), 1)
    f['digit_ratio']       = round(sum(c.isdigit() for c in hostname) / hl, 4)
    f['alpha_ratio']       = round(sum(c.isalpha() for c in hostname) / hl, 4)
    f['subdomain_count']   = len(sub.split('.')) if sub else 0
    f['suspicious_tld']    = int(tld in SUSPICIOUS_TLDS)
    f['tld_length']        = len(tld)
    f['is_ip_host']        = int(bool(re.match(r'^\d{1,3}(\.\d{1,3}){3}$', hostname)))
    f['trusted_domain']    = int(base in TRUSTED_DOMAINS)
    brand_hit              = any(b in hostname for b in BRAND_KEYWORDS)
    f['brand_in_domain']   = int(brand_hit and base not in TRUSTED_DOMAINS)
    f['digit_in_word']     = int(bool(re.search(r'[a-z]\d[a-z]', hostname)))
    f['phish_path_kw']     = int(bool(PHISH_RE.search(path)))
    f['executable_ext']    = int(bool(EXEC_RE.search(path)))
    f['path_depth']        = path.count('/')
    f['path_has_ip']       = int(bool(re.search(r'/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', path)))
    try:
        f['param_count']   = len(parse_qs(query))
    except Exception:
        f['param_count']   = 0
    f['hostname_entropy']  = round(_entropy(hostname), 4)
    f['path_entropy']      = round(_entropy(path), 4)
    f['is_shortener']      = int(hostname in URL_SHORTENERS)
    f['spam_keyword_count']= sum(w in full_lower for w in SPAM_WORDS)
    f['has_punycode']      = int('xn--' in hostname)
    f['domain_age_days']   = 365
    return f


FEATURE_COLUMNS = list(extract_features("http://example.com").keys())


# ─────────────────────────────────────────────────────────────
# TRAINING DATA  (from data/generate_dataset.py)
# ─────────────────────────────────────────────────────────────

BENIGN_URLS = [
    "https://www.google.com/search?q=python+tutorial",
    "https://github.com/scikit-learn/scikit-learn",
    "https://stackoverflow.com/questions/tagged/python",
    "https://www.wikipedia.org/wiki/Machine_learning",
    "https://docs.python.org/3/library/re.html",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.amazon.com/dp/B08N5WRWNW",
    "https://www.linkedin.com/in/johndoe",
    "https://twitter.com/user/status/123456789",
    "https://www.reddit.com/r/learnpython/",
    "https://medium.com/@author/article-title-abc",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://www.bbc.com/news/technology",
    "https://www.nytimes.com/2024/technology/ai.html",
    "https://www.microsoft.com/en-us/microsoft-365",
    "https://www.apple.com/iphone/",
    "https://www.paypal.com/us/home",
    "https://www.netflix.com/browse",
    "https://www.instagram.com/p/ABC123/",
    "https://www.facebook.com/events/12345/",
    "https://accounts.google.com/o/oauth2/auth?client_id=x",
    "https://mail.google.com/mail/u/0/#inbox",
    "https://drive.google.com/file/d/1abc/view",
    "https://www.dropbox.com/s/abc123/file.pdf",
    "https://support.apple.com/en-us/HT201994",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
    "https://nodejs.org/en/docs/",
    "https://reactjs.org/docs/getting-started.html",
    "https://vuejs.org/guide/introduction.html",
    "https://www.coursera.org/learn/machine-learning",
    "https://www.udemy.com/course/python-bootcamp/",
    "https://arxiv.org/abs/2303.08774",
    "https://pypi.org/project/scikit-learn/",
    "https://hub.docker.com/_/python",
    "https://kubernetes.io/docs/concepts/overview/",
    "https://aws.amazon.com/ec2/",
    "https://cloud.google.com/compute/docs",
    "https://azure.microsoft.com/en-us/products/virtual-machines",
    "https://www.cloudflare.com/learning/ddos/",
    "https://letsencrypt.org/getting-started/",
    "https://www.w3schools.com/python/",
    "https://realpython.com/python-f-strings/",
    "https://www.geeksforgeeks.org/python-programming-language/",
    "https://towardsdatascience.com/",
    "https://www.kaggle.com/competitions",
    "https://huggingface.co/models",
    "https://streamlit.io/",
    "https://fastapi.tiangolo.com/",
    "https://flask.palletsprojects.com/",
    "https://www.djangoproject.com/",
]

MALICIOUS_URLS = [
    "http://paypal.com.secure-login-verify.xyz/account/update?token=abc",
    "http://192.168.1.105/admin/login.php?redirect=home",
    "http://g00gle-security-alert.com/verify?user=victim@gmail.com",
    "http://amazon-prize-winner-2024.top/claim?id=99812&ref=email",
    "http://login.microsoftonline.com.phish.tk/oauth2/token",
    "http://secure.paypal-account-verify.ml/login?next=/dashboard",
    "http://bit.ly/3xFreeGift-Claim-Now-2024",
    "http://free-iphone-15-winner.xyz/claim?tracking=FB_AD_001",
    "http://your-bank-secure.suspicious-domain.cc/verify-identity",
    "http://update-your-netflix-billing.live/payment?ref=email",
    "http://apple-id-locked-alert.top/unlock?case=12345",
    "http://win-cash-prize-2024.tk/register?promo=WIN500",
    "http://download-crack-software.ml/setup.exe?id=12345",
    "http://verify-your-facebook-account.xyz/login",
    "http://amazon.com.fake-verify.biz/signin?ref=phish",
    "http://secure-login.paypa1-support.com/help/account",
    "http://google.account-suspended-alert.online/fix",
    "http://dropbox.com.secure.upload-files.info/share",
    "http://www.malware-delivery.net/payload.exe?dl=1",
    "http://urgent-action-required.top/account?email=user@mail.com",
    "http://virus-scan-results.xyz/remove?threatid=9912",
    "http://10.0.0.1/cgi-bin/login.cgi",
    "http://172.16.254.1/setup/admin?pass=admin",
    "http://user@malicious-host.tk/",
    "http://login.ebay.com.cheap-deals-now.pw/signin",
    "http://secure.chase.bank.account-suspended.ml/login",
    "http://track-my-package.xyz/usps?track=1Z999AA0",
    "http://covid-relief-fund.tk/apply?ref=govt",
    "http://faceb00k-security.xyz/recover?id=12345",
    "http://your-crypto-wallet-alert.top/connect?wallet=MetaMask",
    "http://steam-free-gift-card.ml/redeem?code=FREE2024",
    "http://click-here-to-earn-500-usd.top/?aff=1234",
    "http://tinyurl.com/free-adult-content-2024",
    "http://drive.google.com.file-share.xyz/d/1abc/view",
    "http://apple.com.account-locked.online/appleid/unlock",
    "http://secure-login-verify.amazon-account.cc/signin",
    "http://urgent.dhl-delivery-problem.top/track?id=9988",
    "http://bank-notification-alert.xyz/verify?acct=123456",
    "http://microsoft-tech-support-alert.tk/call?code=ERR_VIRUS",
    "http://irs-tax-refund-ready.ml/claim?ssn=needed",
    "http://youtube.com.premium-free.biz/activate",
    "http://instagram-verify-now.xyz/confirm?user=victim",
    "http://netflix.com.billing-update.online/payment",
    "http://fake-antivirus-scan.cc/remove?threats=99",
    "http://your-account-hacked-alert.xyz/secure?id=abc",
    "http://win-free-ps5-console.top/register?promo=PS5FREE",
    "http://paypal.billing.update-required.xyz/confirm",
    "http://icloud.apple.id-verify.cc/unlock",
    "http://bank.account.suspended.suspicious.xyz/verify",
    "http://confirm-you-are-human.xyz/click",
]


# ─────────────────────────────────────────────────────────────
# TRAIN MODEL  (inline — no subprocess needed)
# ─────────────────────────────────────────────────────────────

def _train_and_save(model_path: str):
    urls   = BENIGN_URLS + MALICIOUS_URLS
    labels = [0] * len(BENIGN_URLS) + [1] * len(MALICIOUS_URLS)
    X = pd.DataFrame([extract_features(u) for u in urls])[FEATURE_COLUMNS].fillna(0).values.astype(float)
    y = np.array(labels)
    pipe = Pipeline([("clf", RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=2,
        class_weight='balanced', random_state=42, n_jobs=-1
    ))])
    pipe.fit(X, y)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"model": pipe, "feature_columns": FEATURE_COLUMNS}, model_path)
    return pipe


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
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }
  .stApp { background: #f5f6f8; }

  .top-bar {
    background: #1a2236; border-radius: 12px;
    padding: 1.2rem 1.8rem; margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 12px;
  }
  .top-bar h1 { color: #fff; font-size: 1.4rem; font-weight: 600; margin: 0; }
  .top-bar span { color: #64b5f6; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }

  .card {
    background: #fff; border: 1px solid #e2e6ea;
    border-radius: 10px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
  }
  .card-title {
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #7a869a; margin-bottom: 1rem;
    font-family: 'IBM Plex Mono', monospace;
  }

  .verdict {
    border-radius: 10px; padding: 1.1rem 1.4rem; margin: 1rem 0;
    border-left: 5px solid; display: flex; align-items: center; gap: 14px;
  }
  .verdict-safe       { background: #f0faf4; border-color: #27ae60; }
  .verdict-suspicious { background: #fffbf0; border-color: #f39c12; }
  .verdict-malicious  { background: #fff5f5; border-color: #e74c3c; }
  .verdict-icon  { font-size: 1.8rem; line-height: 1; }
  .verdict-body  { flex: 1; }
  .verdict-label { font-size: 1.2rem; font-weight: 600; }
  .verdict-safe       .verdict-label { color: #1e8449; }
  .verdict-suspicious .verdict-label { color: #b7770d; }
  .verdict-malicious  .verdict-label { color: #c0392b; }
  .verdict-url {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem; color: #7a869a; margin-top: 3px; word-break: break-all;
  }
  .verdict-pct { font-family:'IBM Plex Mono',monospace; font-size:1.5rem; font-weight:600; text-align:right; }
  .verdict-safe       .verdict-pct { color: #27ae60; }
  .verdict-suspicious .verdict-pct { color: #f39c12; }
  .verdict-malicious  .verdict-pct { color: #e74c3c; }

  .signals-wrap { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 0.6rem; }
  .sig-chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 12px; border-radius: 20px; font-size: 0.82rem;
    font-family: 'IBM Plex Mono', monospace;
  }
  .sig-bad  { background: #fdedec; color: #c0392b; border: 1px solid #f5b7b1; }
  .sig-good { background: #eafaf1; color: #1e8449; border: 1px solid #a9dfbf; }

  .prob-label {
    display: flex; justify-content: space-between;
    font-size: 0.82rem; margin-bottom: 4px; color: #4a5568;
    font-family: 'IBM Plex Mono', monospace;
  }
  .bar-track { background: #e9ecef; border-radius: 99px; height: 9px; overflow: hidden; }
  .bar-fill  { height: 100%; border-radius: 99px; }
  .bar-safe      { background: #27ae60; }
  .bar-malicious { background: #e74c3c; }

  .metric-box {
    background: #fff; border: 1px solid #e2e6ea;
    border-radius: 10px; padding: 1rem; text-align: center;
  }
  .metric-val { font-size:1.6rem; font-weight:600; color:#1a2236; font-family:'IBM Plex Mono',monospace; }
  .metric-lbl { font-size:0.72rem; color:#7a869a; text-transform:uppercase; letter-spacing:0.07em; margin-top:3px; }

  section[data-testid="stSidebar"] { background: #1a2236 !important; }
  section[data-testid="stSidebar"] * { color: #c9d6e8 !important; }
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 { color: #fff !important; }
  section[data-testid="stSidebar"] hr { border-color: #2e3f5c !important; }

  .stButton > button {
    background: #1a2236 !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; padding: 0.55rem 1.4rem !important;
  }
  .stButton > button:hover { background: #26344f !important; }

  .stTextInput > div > input, .stTextArea > div > textarea {
    border: 1px solid #d0d7e2 !important; border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.88rem !important; background: #fff !important;
  }
  .stTextInput > div > input:focus, .stTextArea > div > textarea:focus {
    border-color: #2196f3 !important;
    box-shadow: 0 0 0 3px rgba(33,150,243,0.12) !important;
  }

  .stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #fff; border-radius: 10px;
    padding: 4px; border: 1px solid #e2e6ea;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 7px !important; padding: 8px 20px !important;
    font-weight: 500 !important; font-size: 0.88rem !important; color: #7a869a !important;
  }
  .stTabs [aria-selected="true"] { background: #1a2236 !important; color: #fff !important; }
  .stTabs [data-baseweb="tab-panel"] { padding-top: 1.2rem !important; }

  .stDownloadButton > button {
    background: transparent !important; color: #1a2236 !important;
    border: 1px solid #d0d7e2 !important; font-weight: 500 !important;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best_model.joblib")

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("First run — training model (takes ~20 sec)…"):
            mdl = _train_and_save(MODEL_PATH)
            return mdl, FEATURE_COLUMNS
    payload = joblib.load(MODEL_PATH)
    return payload["model"], payload["feature_columns"]

model, feat_cols = load_model()


# ─────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────
def predict(url: str) -> dict:
    raw   = url.strip()
    feats = extract_features(raw)
    feats["domain_age_days"] = max(feats.get("domain_age_days", 365), 0)
    X     = np.array([feats.get(c, 0) for c in feat_cols]).reshape(1, -1)
    prob  = model.predict_proba(X)[0]
    label = int(model.predict(X)[0])
    safe_pct = round(prob[0] * 100, 1)
    mal_pct  = round(prob[1] * 100, 1)
    verdict  = "MALICIOUS" if label == 1 else "SUSPICIOUS" if mal_pct >= 30 else "SAFE"

    try:
        ph = urlparse(raw if "://" in raw else "http://" + raw)
        hostname = (ph.hostname or "").lower()
        tld = hostname.split(".")[-1] if "." in hostname else ""
    except Exception:
        hostname = tld = ""

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
    if not any(k == "bad" for _, k in signals):
        signals.append(("✅ No suspicious signals found", "good"))

    return {"url": raw, "verdict": verdict, "safe_pct": safe_pct,
            "mal_pct": mal_pct, "signals": signals, "features": feats}


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔐 URL Detector")
    st.markdown("---")
    st.markdown("""
**How it works:**
1. Extract 36 URL features
2. Random Forest classifier
3. Trained on 100 labelled URLs
4. Returns threat probability

**Verdict guide:**
""")
    st.success("🟢  SAFE  —  below 30%")
    st.warning("🟡  SUSPICIOUS  —  30–49%")
    st.error("🔴  MALICIOUS  —  50% +")
    st.markdown("---")
    st.caption("Python · scikit-learn · Streamlit")


# ─────────────────────────────────────────────────────────────
# TITLE
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
            "URL", placeholder="https://example.com or paste a suspicious link…",
            key="single_url", label_visibility="collapsed",
        )
    with col_ex:
        examples = {
            "— quick example —":                              "",
            "✅ google.com":       "https://www.google.com/search?q=test",
            "✅ github.com":       "https://github.com/python/cpython",
            "🔴 paypal phish":     "http://paypal.com.secure-login-verify.xyz/account/update",
            "🔴 IP login":         "http://192.168.0.1/admin/login.php",
            "🔴 prize scam":       "http://free-iphone-winner.top/claim?id=99812",
            "🔴 typosquat":        "http://g00gle-security-alert.com/verify?user=you@mail.com",
        }
        ex = st.selectbox("Example", list(examples.keys()), label_visibility="collapsed")
        if examples.get(ex):
            url_input = examples[ex]

    if st.button("🔍  Analyse URL", key="scan") and url_input:
        with st.spinner("Scanning…"):
            time.sleep(0.2)
            result = predict(url_input)

        v     = result["verdict"]
        icon  = "🟢" if v == "SAFE" else "🟡" if v == "SUSPICIOUS" else "🔴"
        cls   = f"verdict-{v.lower()}"
        short = result["url"][:90] + ("…" if len(result["url"]) > 90 else "")
        conf  = result["safe_pct"] if v == "SAFE" else result["mal_pct"]

        st.markdown(f"""
        <div class="verdict {cls}">
          <div class="verdict-icon">{icon}</div>
          <div class="verdict-body">
            <div class="verdict-label">{v}</div>
            <div class="verdict-url">{short}</div>
          </div>
          <div class="verdict-pct">{conf}%</div>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""<div class="card">
              <div class="card-title">Safe probability</div>
              <div class="prob-label"><span>Safe</span><span>{result['safe_pct']}%</span></div>
              <div class="bar-track"><div class="bar-fill bar-safe" style="width:{result['safe_pct']}%"></div></div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="card">
              <div class="card-title">Malicious probability</div>
              <div class="prob-label"><span>Malicious</span><span>{result['mal_pct']}%</span></div>
              <div class="bar-track"><div class="bar-fill bar-malicious" style="width:{result['mal_pct']}%"></div></div>
            </div>""", unsafe_allow_html=True)

        chips = "".join(
            f'<span class="sig-chip sig-{"good" if k=="good" else "bad"}">{s}</span>'
            for s, k in result["signals"]
        )
        st.markdown(f"""<div class="card">
          <div class="card-title">Signal breakdown</div>
          <div class="signals-wrap">{chips}</div>
        </div>""", unsafe_allow_html=True)

        with st.expander("View full feature vector (36 features)"):
            st.dataframe(
                pd.DataFrame(result["features"].items(), columns=["Feature","Value"]).set_index("Feature"),
                use_container_width=True, height=360,
            )


# ══════════════════════════════════════════
# TAB 2 — BATCH
# ══════════════════════════════════════════
with tab2:
    st.markdown("##### Paste URLs below — one per line")
    batch_text = st.text_area(
        "batch", height=160,
        placeholder="https://google.com\nhttp://free-prize.xyz/claim\nhttps://github.com/user/repo",
        label_visibility="collapsed", key="batch_urls",
    )

    if st.button("🔍  Analyse All", key="batch_scan"):
        urls = [u.strip() for u in batch_text.splitlines() if u.strip()]
        if not urls:
            st.warning("Paste at least one URL.")
        else:
            with st.spinner(f"Analysing {len(urls)} URL{'s' if len(urls)>1 else ''}…"):
                rows = []
                for u in urls:
                    r = predict(u)
                    rows.append({"URL": u[:80]+("…" if len(u)>80 else ""),
                                 "Verdict": r["verdict"],
                                 "Safe %": r["safe_pct"],
                                 "Malicious %": r["mal_pct"]})

            df = pd.DataFrame(rows)

            c1, c2, c3 = st.columns(3)
            for col, key, colour, label in [
                (c1, "SAFE",       "#27ae60", "Safe"),
                (c2, "SUSPICIOUS", "#f39c12", "Suspicious"),
                (c3, "MALICIOUS",  "#e74c3c", "Malicious"),
            ]:
                col.markdown(f"""<div class="metric-box">
                  <div class="metric-val" style="color:{colour}">{(df['Verdict']==key).sum()}</div>
                  <div class="metric-lbl">{label}</div></div>""", unsafe_allow_html=True)

            st.markdown("")

            def colour_row(row):
                bg = {"MALICIOUS":"#fff5f5","SUSPICIOUS":"#fffbf0","SAFE":"#f0faf4"}.get(row["Verdict"],"#fff")
                return [f"background-color:{bg}"] * len(row)

            st.dataframe(df.style.apply(colour_row, axis=1), use_container_width=True, height=320)
            st.download_button("⬇️  Download CSV", df.to_csv(index=False),
                               file_name="url_scan_results.csv", mime="text/csv")


# ══════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════
with tab3:

    @st.cache_data(show_spinner="Computing metrics…")
    def get_metrics():
        urls   = BENIGN_URLS + MALICIOUS_URLS
        labels = [0]*len(BENIGN_URLS) + [1]*len(MALICIOUS_URLS)
        X = pd.DataFrame([extract_features(u) for u in urls])[feat_cols].fillna(0).values.astype(float)
        y = np.array(labels)
        _, X_t, _, y_t = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        return y_t, model.predict(X_t), model.predict_proba(X_t)[:, 1]

    y_test, y_pred, y_prob = get_metrics()
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    rep = classification_report(y_test, y_pred,
                                target_names=["Benign","Malicious"], output_dict=True)

    m1, m2, m3, m4 = st.columns(4)
    for col, val, lbl in [
        (m1, f"{acc:.3f}", "Accuracy"),
        (m2, f"{auc:.3f}", "ROC-AUC"),
        (m3, f"{rep['Malicious']['precision']:.3f}", "Precision"),
        (m4, f"{rep['Malicious']['recall']:.3f}", "Recall"),
    ]:
        col.markdown(f"""<div class="metric-box">
          <div class="metric-val">{val}</div>
          <div class="metric-lbl">{lbl}</div></div>""", unsafe_allow_html=True)

    st.markdown("")
    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.markdown("##### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(4, 3.2))
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                               display_labels=["Benign","Malicious"]).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Confusion Matrix", fontsize=11)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col_roc:
        st.markdown("##### ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig2, ax2 = plt.subplots(figsize=(4, 3.2))
        ax2.plot(fpr, tpr, color="#1a2236", lw=2, label=f"AUC = {auc:.3f}")
        ax2.plot([0,1],[0,1],"k--",lw=0.8,alpha=0.4)
        ax2.set_xlabel("False Positive Rate", fontsize=9)
        ax2.set_ylabel("True Positive Rate", fontsize=9)
        ax2.set_title("ROC Curve", fontsize=11)
        ax2.legend(fontsize=9)
        ax2.spines[["top","right"]].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    st.markdown("##### Feature Importance (Top 15)")
    clf = getattr(model, "named_steps", {}).get("clf")
    if clf and hasattr(clf, "feature_importances_"):
        fi = sorted(zip(feat_cols, clf.feature_importances_), key=lambda x: -x[1])[:15]
        names_fi, vals_fi = zip(*fi)
        fig3, ax3 = plt.subplots(figsize=(8, 3.8))
        ax3.barh(list(names_fi)[::-1], list(vals_fi)[::-1], color="#1a2236", height=0.6)
        ax3.set_xlabel("Importance Score", fontsize=9)
        ax3.tick_params(labelsize=9)
        ax3.spines[["top","right"]].set_visible(False)
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)

    with st.expander("Full Classification Report"):
        st.dataframe(pd.DataFrame(rep).T.round(4), use_container_width=True)
