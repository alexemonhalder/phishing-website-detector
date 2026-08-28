"""
feature_extractor.py

Computes the 30 features used by the UCI "Phishing Websites" dataset
(Mohammad, McCluskey & Thabtah) directly from a live URL, so the trained
model can be used on real-world links instead of pre-extracted rows.

Feature order MUST match the column order the model was trained on:
having_IP_Address, URL_Length, Shortining_Service, having_At_Symbol,
double_slash_redirecting, Prefix_Suffix, having_Sub_Domain, SSLfinal_State,
Domain_registeration_length, Favicon, port, HTTPS_token, Request_URL,
URL_of_Anchor, Links_in_tags, SFH, Submitting_to_email, Abnormal_URL,
Redirect, on_mouseover, RightClick, popUpWidnow, Iframe, age_of_domain,
DNSRecord, web_traffic, Page_Rank, Google_Index, Links_pointing_to_page,
Statistical_report

Notes on accuracy:
- Several original-paper features relied on services that are dead or paywalled
  today (Alexa web_traffic rank, Google PageRank, Google's old site: index count,
  PhishTank/StopBadware "Statistical_report" blacklists). Those four are
  approximated or defaulted to a neutral value (0) and flagged in the response
  so you know the prediction leans more on the other 26 live-computed signals.
- This module makes live network calls (HTTP fetch, WHOIS, DNS) and can take
  a few seconds per request. Consider caching results by domain in production.
"""

import re
import socket
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
import tldextract
import dns.resolver
from bs4 import BeautifulSoup

try:
    import whois  # python-whois
except ImportError:
    whois = None

TIMEOUT = 3
HEADERS = {"User-Agent": "Mozilla/5.0"}

SHORTENING_SERVICES = re.compile(
    r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|"
    r"is\.gd|cli\.gs|yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|"
    r"su\.pr|twurl\.nl|snipurl\.com|short\.to|budurl\.com|ping\.fm|post\.ly|"
    r"just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|doiop\.com|short\.ie|"
    r"kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|lnkd\.in|db\.tt|"
    r"qr\.ae|adf\.ly|cur\.lv|v\.gd|tinyarrows|po\.st"
)


def _safe_get(url):
    try:
        return requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
    except requests.RequestException:
        return None


def _get_domain(url):
    ext = tldextract.extract(url)
    return ".".join(part for part in [ext.domain, ext.suffix] if part)


def _get_registered_domain_with_sub(url):
    ext = tldextract.extract(url)
    return ext


class PhishingFeatureExtractor:
    def __init__(self, url: str):
        if not re.match(r"^https?://", url, re.I):
            url = "http://" + url
        self.url = url
        self.parsed = urlparse(url)
        self.hostname = self.parsed.hostname or ""
        self.domain = _get_domain(url)
        self.response = _safe_get(url)
        self.soup = None
        self.final_url = url
        self.warnings = []

        if self.response is not None:
            self.final_url = self.response.url
            try:
                self.soup = BeautifulSoup(self.response.text, "html.parser")
            except Exception:
                self.soup = None
        else:
            self.warnings.append(
                "Could not fetch page content (site unreachable/blocking bots); "
                "HTML-based features default to the 'suspicious' value."
            )

        self._whois_data = self._get_whois()

    # ---------- helpers ----------

    def _get_whois(self):
        if whois is None:
            return None
        try:
            return whois.whois(self.domain)
        except Exception:
            return None

    @staticmethod
    def _first_date(value):
        if isinstance(value, list):
            value = value[0] if value else None
        return value

    # ---------- individual features (return -1, 0, or 1 per original schema) ----------

    def having_IP_Address(self):
        try:
            socket.inet_aton(self.hostname)
            return -1  # IP address used directly -> phishing signal
        except (OSError, TypeError):
            return 1

    def URL_Length(self):
        n = len(self.url)
        if n < 54:
            return 1
        elif n <= 75:
            return 0
        return -1

    def Shortining_Service(self):
        return -1 if SHORTENING_SERVICES.search(self.url) else 1

    def having_At_Symbol(self):
        return -1 if "@" in self.url else 1

    def double_slash_redirecting(self):
        # legit "//" only appears right after the protocol
        after_protocol = self.url.find("//", 8)
        return -1 if after_protocol > 7 else 1

    def Prefix_Suffix(self):
        return -1 if "-" in self.domain else 1

    def having_Sub_Domain(self):
        ext = _get_registered_domain_with_sub(self.url)
        sub = ext.subdomain
        dots = sub.count(".") + 1 if sub else 0
        if dots == 0:
            return 1
        elif dots == 1:
            return 0
        return -1

    def SSLfinal_State(self):
        if self.parsed.scheme == "https" and self.response is not None:
            return 1
        elif self.parsed.scheme == "https":
            return 0
        return -1

    def Domain_registeration_length(self):
        if not self._whois_data:
            return -1
        try:
            exp = self._first_date(self._whois_data.expiration_date)
            create = self._first_date(self._whois_data.creation_date)
            if not exp or not create:
                return -1
            months = (exp - create).days / 30
            return 1 if months >= 12 else -1
        except Exception:
            return -1

    def Favicon(self):
        if not self.soup:
            return -1
        icon = self.soup.find("link", rel=re.compile("icon", re.I))
        if not icon or not icon.get("href"):
            return 1  # no favicon tag isn't itself suspicious; default neutral-legit
        href = icon["href"]
        return 1 if self.domain in href or href.startswith("/") else -1

    def port(self):
        return 1 if not self.parsed.port or self.parsed.port in (80, 443) else -1

    def HTTPS_token(self):
        return -1 if "https" in (self.parsed.hostname or "").replace("https://", "") and self.parsed.scheme != "https" else 1

    def Request_URL(self):
        if not self.soup:
            return -1
        tags = self.soup.find_all(["img", "audio", "embed", "iframe", "script"])
        total, external = 0, 0
        for t in tags:
            src = t.get("src")
            if not src:
                continue
            total += 1
            if self.domain not in src and src.startswith(("http", "//")):
                external += 1
        if total == 0:
            return 1
        pct_external = external / total * 100
        if pct_external < 22:
            return 1
        elif pct_external < 61:
            return 0
        return -1

    def URL_of_Anchor(self):
        if not self.soup:
            return -1
        anchors = self.soup.find_all("a", href=True)
        if not anchors:
            return 1
        suspicious = 0
        for a in anchors:
            href = a["href"].strip().lower()
            if href in ("#", "") or href.startswith(("javascript:void", "#content")):
                suspicious += 1
            elif href.startswith("http") and self.domain not in href:
                suspicious += 1
        pct = suspicious / len(anchors) * 100
        if pct < 31:
            return 1
        elif pct < 67:
            return 0
        return -1

    def Links_in_tags(self):
        if not self.soup:
            return -1
        tags = self.soup.find_all(["meta", "script", "link"])
        total, external = 0, 0
        for t in tags:
            src = t.get("src") or t.get("href")
            if not src:
                continue
            total += 1
            if self.domain not in src and src.startswith(("http", "//")):
                external += 1
        if total == 0:
            return 1
        pct = external / total * 100
        if pct < 17:
            return 1
        elif pct < 81:
            return 0
        return -1

    def SFH(self):
        if not self.soup:
            return -1
        form = self.soup.find("form")
        if not form:
            return 1
        action = (form.get("action") or "").strip().lower()
        if action in ("", "about:blank"):
            return -1
        if action.startswith("http") and self.domain not in action:
            return 0
        return 1

    def Submitting_to_email(self):
        if not self.soup:
            return 1
        html = str(self.soup)
        return -1 if ("mailto:" in html or "mail(" in html) else 1

    def Abnormal_URL(self):
        if not self._whois_data:
            return -1
        try:
            registrant = str(self._whois_data.get("domain_name", "")).lower()
            return 1 if self.domain.split(".")[0] in registrant else -1
        except Exception:
            return -1

    def Redirect(self):
        if self.response is None:
            return 0
        return 1 if len(self.response.history) >= 4 else 0

    def on_mouseover(self):
        if not self.soup:
            return 1
        html = str(self.soup)
        return -1 if "onmouseover" in html.lower() and "window.status" in html.lower() else 1

    def RightClick(self):
        if not self.soup:
            return 1
        html = str(self.soup).lower()
        return -1 if "event.button==2" in html or "contextmenu" in html else 1

    def popUpWidnow(self):
        if not self.soup:
            return 1
        html = str(self.soup).lower()
        return -1 if "window.open" in html and ("text" in html or "input" in html) else 1

    def Iframe(self):
        if not self.soup:
            return 1
        return -1 if self.soup.find("iframe") else 1

    def age_of_domain(self):
        if not self._whois_data:
            return -1
        try:
            create = self._first_date(self._whois_data.creation_date)
            if not create:
                return -1
            months = (datetime.now(timezone.utc).replace(tzinfo=None) - create).days / 30
            return 1 if months >= 6 else -1
        except Exception:
            return -1

    def DNSRecord(self):
        try:
            dns.resolver.resolve(self.domain, "A")
            return 1
        except Exception:
            return -1

    # ---- features without a reliable free live source: default to neutral (0) ----

    def web_traffic(self):
        self.warnings.append("web_traffic: Alexa rank API is discontinued; defaulted to neutral (0).")
        return 0

    def Page_Rank(self):
        self.warnings.append("Page_Rank: Google PageRank API is discontinued; defaulted to neutral (0).")
        return 0

    def Google_Index(self):
        self.warnings.append("Google_Index: not checked live (would require a paid search API); defaulted to indexed (1).")
        return 1

    def Links_pointing_to_page(self):
        self.warnings.append("Links_pointing_to_page: no free backlink API used; defaulted to neutral (0).")
        return 0

    def Statistical_report(self):
        self.warnings.append("Statistical_report: PhishTank/StopBadware blacklist not queried live; defaulted to clean (-1 = not reported).")
        return -1

    # ---------- public API ----------

    FEATURE_ORDER = [
        "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
        "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State",
        "Domain_registeration_length", "Favicon", "port", "HTTPS_token", "Request_URL",
        "URL_of_Anchor", "Links_in_tags", "SFH", "Submitting_to_email", "Abnormal_URL",
        "Redirect", "on_mouseover", "RightClick", "popUpWidnow", "Iframe", "age_of_domain",
        "DNSRecord", "web_traffic", "Page_Rank", "Google_Index", "Links_pointing_to_page",
        "Statistical_report",
    ]

    def extract(self):
        values = []
        for name in self.FEATURE_ORDER:
            method = getattr(self, name)
            try:
                values.append(method())
            except Exception as e:
                self.warnings.append(f"{name}: extraction failed ({e}); defaulted to -1.")
                values.append(-1)
        return dict(zip(self.FEATURE_ORDER, values)), self.warnings
