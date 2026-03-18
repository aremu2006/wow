"""
src/feature_extraction.py
==========================
Lexical + host-based URL feature extraction.
Works without tldextract or python-whois installed
(uses built-in re + urllib.parse as fallbacks).
"""

import re
import math
import socket
from urllib.parse import urlparse, parse_qs

# ── Optional imports (graceful fallback) ──────────────────────────────────────
try:
    import tldextract
    HAS_TLDEXTRACT = True
except ImportError:
    HAS_TLDEXTRACT = False

try:
    import whois
    HAS_WHOIS = True
except ImportError:
    HAS_WHOIS = False

# ── Static lookup tables ──────────────────────────────────────────────────────

SUSPICIOUS_TLDS = {
    'xyz', 'top', 'click', 'tk', 'ml', 'ga', 'cf', 'gq', 'pw', 'cc',
    'su', 'biz', 'info', 'online', 'site', 'live', 'stream', 'download',
    'loan', 'review', 'country', 'kim', 'science', 'work', 'party', 'trade',
    'cricket', 'date', 'faith', 'racing', 'accountant', 'win', 'bid',
    'men', 'icu', 'monster', 'cyou', 'buzz', 'sbs', 'ru',
}

TRUSTED_DOMAINS = {
    'google.com', 'youtube.com', 'facebook.com', 'microsoft.com', 'apple.com',
    'amazon.com', 'github.com', 'twitter.com', 'linkedin.com', 'wikipedia.org',
    'instagram.com', 'netflix.com', 'stackoverflow.com', 'reddit.com', 'paypal.com',
    'bbc.com', 'nytimes.com', 'dropbox.com', 'mozilla.org', 'cloudflare.com',
    'medium.com', 'kaggle.com', 'huggingface.co', 'arxiv.org', 'nature.com',
    'zoom.us', 'slack.com', 'notion.so', 'figma.com', 'canva.com', 'stripe.com',
    'shopify.com', 'heroku.com', 'vercel.com', 'netlify.com', 'firebase.google.com',
    'aws.amazon.com', 'cloud.google.com', 'azure.microsoft.com',
}

BRAND_KEYWORDS = [
    'paypal', 'google', 'apple', 'microsoft', 'amazon', 'facebook',
    'instagram', 'netflix', 'ebay', 'steam', 'whatsapp', 'youtube',
    'dropbox', 'icloud', 'twitter', 'chase', 'wellsfargo', 'citibank',
    'bankofamerica', 'boa', 'dhl', 'fedex', 'usps', 'ups',
]

URL_SHORTENERS = {
    'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly', 'is.gd',
    'buff.ly', 'rebrand.ly', 'short.io', 'tiny.cc', 'cutt.ly',
}

PHISH_WORDS = re.compile(
    r'login|signin|verify|account|update|secure|confirm|'
    r'password|credential|alert|suspend|unlock|recover|reset|'
    r'billing|payment|invoice', re.I
)

SPAM_WORDS = [
    'free', 'win', 'prize', 'claim', 'urgent', 'alert',
    'suspended', 'verify', 'confirm', 'limited', 'offer',
    'bonus', 'gift', 'reward', 'lucky', 'congratulation',
]

EXEC_EXT = re.compile(
    r'\.(exe|bat|cmd|msi|scr|vbs|jar|apk|dmg|sh|ps1|crx|xpi)$', re.I
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


def _parse_domain_parts(hostname: str):
    """Return (subdomain, domain, tld) using tldextract if available, else regex."""
    hostname = hostname.lower()
    if HAS_TLDEXTRACT:
        ext = tldextract.extract(hostname)
        return ext.subdomain, ext.domain, ext.suffix
    # Fallback: simple split
    clean = re.sub(r'^www\.', '', hostname)
    parts = clean.split('.')
    if len(parts) >= 3:
        return '.'.join(parts[:-2]), parts[-2], parts[-1]
    elif len(parts) == 2:
        return '', parts[0], parts[1]
    else:
        return '', hostname, ''


def _domain_age_days(hostname: str) -> int:
    """Return approximate domain age in days. Returns -1 if unavailable."""
    if not HAS_WHOIS:
        return -1
    try:
        import datetime
        w = whois.whois(hostname)
        created = w.creation_date
        if isinstance(created, list):
            created = created[0]
        if created is None:
            return -1
        delta = datetime.datetime.now() - created
        return delta.days
    except Exception:
        return -1


# ── Main feature extractor ────────────────────────────────────────────────────

def extract_features(url: str) -> dict:
    """
    Extract 35 numerical/binary features from a URL string.

    Returns
    -------
    dict : feature_name -> numeric value
    """
    raw = str(url).strip()
    features = {}

    # ── Parse ────────────────────────────────────────────────────────────────
    try:
        parsed = urlparse(raw if '://' in raw else 'http://' + raw)
    except Exception:
        parsed = urlparse('http://invalid')

    hostname  = (parsed.hostname or '').lower()
    path      = parsed.path or ''
    query     = parsed.query or ''
    fragment  = parsed.fragment or ''
    scheme    = parsed.scheme or ''
    full_lower = raw.lower()

    subdomain, domain, tld = _parse_domain_parts(hostname)
    base_domain = f"{domain}.{tld}" if domain and tld else hostname

    # ── 1. Protocol ──────────────────────────────────────────────────────────
    features['is_https']        = int(scheme == 'https')
    features['is_http']         = int(scheme == 'http')

    # ── 2. URL length features ───────────────────────────────────────────────
    features['url_length']       = len(raw)
    features['hostname_length']  = len(hostname)
    features['path_length']      = len(path)
    features['query_length']     = len(query)

    # ── 3. Special character counts ──────────────────────────────────────────
    features['dot_count']        = hostname.count('.')
    features['hyphen_count']     = hostname.count('-')
    features['underscore_count'] = raw.count('_')
    features['at_sign']          = int('@' in raw)
    features['double_slash']     = int('//' in path)
    features['question_mark']    = int('?' in raw)
    features['ampersand_count']  = query.count('&')
    features['equals_count']     = query.count('=')
    features['percent_count']    = len(re.findall(r'%[0-9a-fA-F]{2}', raw))
    features['hash_count']       = int('#' in raw)

    # ── 4. Digit/alpha ratios ────────────────────────────────────────────────
    digits = sum(c.isdigit() for c in hostname)
    alphas = sum(c.isalpha() for c in hostname)
    hl = max(len(hostname), 1)
    features['digit_ratio']      = round(digits / hl, 4)
    features['alpha_ratio']      = round(alphas / hl, 4)

    # ── 5. Subdomain depth ───────────────────────────────────────────────────
    features['subdomain_count']  = len(subdomain.split('.')) if subdomain else 0

    # ── 6. TLD flags ─────────────────────────────────────────────────────────
    features['suspicious_tld']   = int(tld in SUSPICIOUS_TLDS)
    features['tld_length']       = len(tld)

    # ── 7. IP address as host ────────────────────────────────────────────────
    features['is_ip_host']       = int(bool(re.match(
        r'^\d{1,3}(\.\d{1,3}){3}$', hostname)))

    # ── 8. Trusted / brand checks ────────────────────────────────────────────
    features['trusted_domain']   = int(base_domain in TRUSTED_DOMAINS)
    brand_hit = any(b in hostname for b in BRAND_KEYWORDS)
    features['brand_in_domain']  = int(brand_hit and base_domain not in TRUSTED_DOMAINS)

    # ── 9. Typosquatting ─────────────────────────────────────────────────────
    features['digit_in_word']    = int(bool(re.search(r'[a-z]\d[a-z]', hostname)))

    # ── 10. Path features ────────────────────────────────────────────────────
    features['phish_path_kw']    = int(bool(PHISH_WORDS.search(path)))
    features['executable_ext']   = int(bool(EXEC_EXT.search(path)))
    features['path_depth']       = path.count('/')
    features['path_has_ip']      = int(bool(re.search(
        r'/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', path)))

    # ── 11. Query params ─────────────────────────────────────────────────────
    try:
        features['param_count']  = len(parse_qs(query))
    except Exception:
        features['param_count']  = 0

    # ── 12. Entropy ──────────────────────────────────────────────────────────
    features['hostname_entropy'] = round(_shannon_entropy(hostname), 4)
    features['path_entropy']     = round(_shannon_entropy(path), 4)

    # ── 13. URL shortener ────────────────────────────────────────────────────
    features['is_shortener']     = int(hostname in URL_SHORTENERS)

    # ── 14. Spam keywords ────────────────────────────────────────────────────
    features['spam_keyword_count'] = sum(w in full_lower for w in SPAM_WORDS)

    # ── 15. Punycode (IDN homograph attack) ──────────────────────────────────
    features['has_punycode']     = int('xn--' in hostname)

    # ── 16. Domain age (days) — requires python-whois ────────────────────────
    features['domain_age_days']  = _domain_age_days(base_domain)

    return features


# Column order (used by train_model.py and app.py to ensure consistent input)
FEATURE_COLUMNS = list(extract_features("http://example.com").keys())


if __name__ == "__main__":
    test_urls = [
        "https://www.google.com/search?q=test",
        "http://paypal.com.secure-login-verify.xyz/account?token=abc",
        "http://192.168.1.1/admin/login.php",
    ]
    import pandas as pd
    rows = [extract_features(u) for u in test_urls]
    df = pd.DataFrame(rows)
    print(df[['url_length', 'is_https', 'suspicious_tld',
              'brand_in_domain', 'is_ip_host', 'hostname_entropy']].to_string())
