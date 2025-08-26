import os, json, math
from datetime import datetime, timezone
import requests, pandas as pd, numpy as np, streamlit as st

# ──────────────────────────────────────────────────────────────────────────
# App config
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Macro + Volatility Trading Platform (Consolidated)", layout="wide")
st.title("🏦 Macro + Volatility Trading Platform — Consolidated")

DATA_DIR = "data"
THEMES_CSV = f"{DATA_DIR}/themes.csv"
SCORECARD_CSV = f"{DATA_DIR}/trade_scorecard.csv"
PORTFOLIO_CSV = f"{DATA_DIR}/portfolio_nav.csv"
os.makedirs(DATA_DIR, exist_ok=True)

# Assets covered in Vol Monitor (extendable)
ASSETS = ["BTC", "ETH"]
SYMBOLS = {"BTC": ["BTC-USD", "BTCUSD", "BTC"], "ETH": ["ETH-USD", "ETHUSD", "ETH"]}
TENOR_SHORT_TRIES = ["1w", "7d", "P7D"]
TENOR_LONG_TRIES  = ["30d", "1m", "30D", "P30D"]
DELTA_TRIES       = [49, 50, 0.49, 0.50]  # try ATM-ish variants

# InvestDEFY endpoint (correct)
BASE_URL = "https://api.investdefy.com/v1/data/volatility-surface"

# Fallback realized vol for display (you can wire live RV later)
RV_DEFAULT = 0.32

# 🔑 Embedded API key (you can leave as-is or replace with st.secrets)
# You can also paste a key in the Settings tab; session uses: pasted → env/secrets → embedded (fallback)
EMBEDDED_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJjYThiY2NjMy00NTdhLTRjMjktYjYyYy02NDMyYWQ3OGRjZTUiLCJleHAiOjE3NTU4NjQ4NDAsImlhdCI6MTc1MzI3Mjg0MCwiaXNzIjoiaW52ZXN0ZGVmeS5jb20ifQ.8jyYpidD6kdTvWR8fcPh4boBjFQPo3wh4UVoKMkQ0w4"

# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
def get_api_key():
    k = st.session_state.get("INVESTDEFY_API_KEY")
    if k: return k.strip()
    try:
        k = st.secrets.get("INVESTDEFY_API_KEY", "") or os.environ.get("INVESTDEFY_API_KEY", "")
    except Exception:
        k = os.environ.get("INVESTDEFY_API_KEY", "")
    return (k or EMBEDDED_KEY).strip()

def _get(url, headers):
    try:
        r = requests.get(url, headers=headers, timeout=20)
        return r.status_code, r.text, r.headers
    except Exception as e:
        return -1, str(e), {}

def parse_iv(payload_text, tenor_hint=None, delta_hint=None):
    """Parse common shapes from the volatility-surface API."""
    try:
        d = json.loads(payload_text)
    except Exception:
        return None
    if isinstance(d, (int, float)):
        return float(d)
    if isinstance(d, dict):
        if "iv" in d:
            try: return float(d["iv"])
            except: return None
        # dime-shaped array
        if "dime" in d and isinstance(d["dime"], list):
            best_iv, best_diff = None, 1e9
            for row in d["dime"]:
                try:
                    t = str(row.get("tenor", "")).lower()
                    dh = float(row.get("delta", 0))
                    if tenor_hint is not None and str(tenor_hint).lower() == t:
                        diff = abs(float(delta_hint) - dh) if delta_hint is not None else 0
                        if diff < best_diff:
                            best_diff = diff; best_iv = float(row["iv"])
                except:
                    pass
            return best_iv
        # wrapped shape
        if "data" in d and isinstance(d["data"], dict) and "iv" in d["data"]:
            try: return float(d["data"]["iv"])
            except: pass
    return None

def build_urls(symbol, tenor, delta):
    """Try multiple parameter names (symbol/underlying, tenor/maturity/expiry, delta/targetDelta, asDecimal)."""
    ten = str(tenor); d = str(delta)
    combos = [
        [("symbol", symbol),     ("tenor", ten),    ("delta", d)],
        [("underlying", symbol), ("tenor", ten),    ("delta", d)],
        [("symbol", symbol),     ("maturity", ten), ("delta", d)],
        [("symbol", symbol),     ("expiry", ten),   ("delta", d)],
        [("symbol", symbol),     ("tenor", ten),    ("targetDelta", d)],
        [("underlying", symbol), ("maturity", ten), ("targetDelta", d)],
        [("underlying", symbol), ("expiry", ten),   ("targetDelta", d)],
        [("symbol", symbol),     ("tenor", ten),    ("delta", d),        ("asDecimal", "true")],
        [("symbol", symbol),     ("maturity", ten), ("delta", d),        ("asDecimal", "true")],
        [("symbol", symbol),     ("tenor", ten),    ("targetDelta", d),  ("asDecimal", "true")],
    ]
    urls = []
    for kvs in combos:
        qs = "&".join([f"{k}={v}" for k, v in kvs])
        urls.append(f"{BASE_URL}?{qs}")
    return urls

@st.cache_data(ttl=60, show_spinner=False)
def fetch_iv(asset, tenor_kind):
    """Return (iv, successful_url, diagnostics_list)"""
    assert tenor_kind in ("short", "long")
    headers = {"Authorization": f"Bearer {get_api_key()}", "Accept": "application/json"}
    tenors = TENOR_SHORT_TRIES if tenor_kind == "short" else TENOR_LONG_TRIES
    tried = []
    for sym in SYMBOLS[asset]:
        for ten in tenors:
            for d in DELTA_TRIES:
                for url in build_urls(sym, ten, d):
                    code, body, _ = _get(url, headers)
                    tried.append({"status": code, "url": url, "preview": (body[:400] if isinstance(body, str) else str(body))})
                    if code == 200:
                        iv = parse_iv(body, tenor_hint=ten, delta_hint=d)
                        if isinstance(iv, (int, float)) and np.isfinite(iv):
                            return iv, url, tried
    return None, None, tried

def regime_from_vol(iv_s, iv_l):
    """Simple macro regime detector from term structure."""
    if iv_s is None or iv_l is None: return "Neutral"
    skew = iv_l - iv_s
    if skew > 0.03:  return "Calm/Carry (Contango)"
    if skew < -0.03: return "Stress/Backwardation"
    return "Neutral"

def load_themes():
    cols = ["Theme", "Category", "Score", "Notes"]
    if os.path.exists(THEMES_CSV):
        try:
            df = pd.read_csv(THEMES_CSV)
            for c in cols:
                if c not in df.columns: df[c] = np.nan
            return df[cols]
        except: pass
    # Defaults
    return pd.DataFrame([
        {"Theme":"Reflation","Category":"Macro","Score":4.2,"Notes":"Growth up, inflation up"},
        {"Theme":"Disinflation","Category":"Macro","Score":3.4,"Notes":"Inflation cooling"},
        {"Theme":"Risk-Off","Category":"Macro","Score":4.0,"Notes":"Wider risk aversion"},
        {"Theme":"Crypto Adoption","Category":"Crypto","Score":3.8,"Notes":"Flows supportive"},
        {"Theme":"Rates Cut Cycle","Category":"Rates","Score":3.5,"Notes":"Central bank easing"},
    ])

def save_themes(df): df.to_csv(THEMES_CSV, index=False)

def load_scorecard():
    cols = ["Timestamp","Theme","Asset","Idea","Score","Weight(%)","Signal","Hedge","Exp","Status"]
    if os.path.exists(SCORECARD_CSV):
        try:
            d = pd.read_csv(SCORECARD_CSV)
            for c in cols:
                if c not in d.columns: d[c] = np.nan
            return d[cols]
        except: pass
    return pd.DataFrame(columns=cols)

def save_scorecard(df): df.to_csv(SCORECARD_CSV, index=False)

def load_portfolio():
    cols = ["Timestamp","Asset","NAV"]
    if os.path.exists(PORTFOLIO_CSV):
        try:
            d = pd.read_csv(PORTFOLIO_CSV)
            for c in cols:
                if c not in d.columns: d[c] = np.nan
            return d[cols]
        except: pass
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return pd.DataFrame([{"Timestamp": now, "Asset": a, "NAV": 100.0} for a in ASSETS])

def save_portfolio(df): df.to_csv(PORTFOLIO_CSV, index=False)

def post_slack(webhook, text):
    if not webhook: return
    try: requests.post(webhook, json={"text": text}, timeout=8)
    except Exception: pass

# ──────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    st.caption("InvestDEFY IV → Themes → Signals → Playbook → Trades → Portfolio")
    try:
        from streamlit_autorefresh import st_autorefresh
        st.checkbox("Auto-refresh", value=False, key="auto")
        st.number_input("Interval (seconds)", 15, 600, 60, 15, key="ival")
        if st.session_state.get("auto"):
            st_autorefresh(interval=int(st.session_state.get("ival", 60)) * 1000, key="autokey")
    except Exception:
        pass
    st.divider()
    api_override = st.text_input("InvestDEFY API Key (optional override)", value=st.session_state.get("INVESTDEFY_API_KEY",""), type="password")
    if st.button("Use this API key"):
        if api_override.strip():
            st.session_state["INVESTDEFY_API_KEY"] = api_override.strip()
            st.success("Saved to session. Reload Vol_Monitor.")
        else:
            st.warning("Paste a non-empty key.")
    slack_hook = st.text_input("Slack Webhook (optional alerts)", value=st.session_state.get("SLACK_WEBHOOK",""))
    if st.button("Save Slack Hook"):
        st.session_state["SLACK_WEBHOOK"] = slack_hook.strip()
        st.success("Slack webhook saved for this session.")

# ──────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Dashboard",
    "Vol_Monitor",
    "Theme_Scores",
    "Signal_Matrix",
    "Playbook",
    "Trade_Scorecard",
    "Portfolio",
    "Backtest",
    "Diagnostics",
])
tab_dash, tab_vol, tab_theme, tab_signal, tab_play, tab_score, tab_port, tab_bt, tab_diag = tabs

# ──────────────────────────────────────────────────────────────────────────
# Vol_Monitor
# ──────────────────────────────────────────────────────────────────────────
with tab_vol:
    st.subheader("Vol Monitor — ATM-ish (49–50Δ), Short vs Long")
    rows=[]; calls=[]; diag={}
    for a in ASSETS:
        iv_s, u1, t1 = fetch_iv(a, "short")
        iv_l, u2, t2 = fetch_iv(a, "long")
        diag[a] = (t1 + t2)[:14]
        rv7 = RV_DEFAULT
        skew = (iv_l - iv_s) if (isinstance(iv_l,(int,float)) and isinstance(iv_s,(int,float))) else None
        rows.append({"Asset":a,"IV_short":iv_s,"IV_long":iv_l,"RV_7d":rv7,"Term_Skew":skew,"Regime": regime_from_vol(iv_s, iv_l)})
        calls.append({"asset":a,"short_url":u1,"long_url":u2})
    vol_df = pd.DataFrame(rows)
    st.dataframe(vol_df, use_container_width=True)
    st.markdown("**Last successful URLs**")
    st.code(json.dumps(calls, indent=2))
    st.session_state["__diag__"] = diag
    st.session_state["__vol__"] = vol_df

# ──────────────────────────────────────────────────────────────────────────
# Theme_Scores (editable)
# ──────────────────────────────────────────────────────────────────────────
with tab_theme:
    st.subheader("Theme Scores (editable; scores out of 5)")
    themes = load_themes()
    edited = st.data_editor(themes, use_container_width=True, num_rows="dynamic")
    if st.button("Save Themes"):
        save_themes(edited)
        st.success("Theme scores saved.")
    safe = edited.copy()
    safe["Score"] = pd.to_numeric(safe["Score"], errors="coerce").fillna(0)
    total = safe["Score"].sum() or 1.0
    safe["Weight"] = safe["Score"] / total
    st.markdown("**Weights from Scores**")
    st.dataframe(safe[["Theme","Category","Score","Weight","Notes"]], use_container_width=True)
    st.session_state["__themes__"] = safe

# ──────────────────────────────────────────────────────────────────────────
# Playbook (macro regime → trade ideas)
# ──────────────────────────────────────────────────────────────────────────
with tab_play:
    st.subheader("Playbook — Macro regime → Trade ideas & hedges")
    play = pd.DataFrame({
        "Macro_Regime":["Calm/Carry (Contango)","Neutral","Stress/Backwardation"],
        "Equities":["Long QQQ / Carry","Barbell (Quality + Value)","Defensive sectors (XLU), reduce beta"],
        "FX":["Short JPY, Long carry","Neutral USD","Long USD, Long CHF"],
        "Fixed_Income":["Short duration","Neutral duration","Long duration (TLT)"],
        "Crypto":["Long calendar (sell 1w, buy 1m)","Delta-neutral carry","Long gamma / short calendar"],
        "Hedges":["VIX puts (income)","Modest tail hedges","Gold, VIX calls, TLT calls"]
    })
    st.dataframe(play, use_container_width=True)
    st.session_state["__playbook__"] = play

# ──────────────────────────────────────────────────────────────────────────
# Signal_Matrix (themes + vol regime → signals)
# ──────────────────────────────────────────────────────────────────────────
with tab_signal:
    st.subheader("Signal Matrix — from Themes & Vol")
    vol_df = st.session_state.get("__vol__", pd.DataFrame())
    th = st.session_state.get("__themes__", load_themes())
    regime = "Neutral"
    if not vol_df.empty and (vol_df["Asset"]=="BTC").any():
        row = vol_df[vol_df["Asset"]=="BTC"].iloc[0]
        regime = regime_from_vol(row["IV_short"], row["IV_long"])
    st.metric("Detected Macro Regime (from term skew)", regime)

    th_sorted = th.sort_values("Score", ascending=False) if not th.empty else th
    top3 = th_sorted.head(3)["Theme"].tolist() if not th.empty else []
    matrix_rows=[]
    for a in ASSETS:
        if regime == "Calm/Carry (Contango)":
            sig = "Long Calendar (sell short tenor, buy long tenor)"
        elif regime == "Stress/Backwardation":
            sig = "Long Gamma (buy short tenor) / Short Calendar"
        else:
            sig = "Neutral / Carry-light"
        score = float(th_sorted["Score"].head(5).mean()) if not th_sorted.empty else 0.0
        matrix_rows.append({"Asset":a,"Signal":sig,"Theme_Boost":round(score,2),"Top_Themes":", ".join(top3)})
    sig_df = pd.DataFrame(matrix_rows)
    st.dataframe(sig_df, use_container_width=True)
    st.session_state["__signals__"] = sig_df

# ──────────────────────────────────────────────────────────────────────────
# Trade_Scorecard (auto-pick ideas where theme score > 3)
# ──────────────────────────────────────────────────────────────────────────
with tab_score:
    st.subheader("Trade Scorecard (auto-add ideas where Theme Score > 3)")
    sig = st.session_state.get("__signals__", pd.DataFrame())
    th = st.session_state.get("__themes__", load_themes())
    sc = load_scorecard()
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    to_add = []
    for _, thr in th.iterrows():
        try:
            if float(thr["Score"]) > 3.0:
                for a in ASSETS:
                    # map theme to current signal
                    sig_map = st.session_state.get("__signals__", pd.DataFrame()).set_index("Asset")
                    idea_sig = sig_map.loc[a]["Signal"] if (not sig_map.empty and a in sig_map.index) else "Neutral"
                    to_add.append({
                        "Timestamp": now,
                        "Theme": thr["Theme"],
                        "Asset": a,
                        "Idea": f"{a} — {idea_sig}",
                        "Score": float(thr["Score"]),
                        "Weight(%)": round(100*float(thr["Score"])/(th["Score"].sum() or 1.0), 2),
                        "Signal": idea_sig,
                        "Hedge":"See Playbook",
                        "Exp":"1w/1m",
                        "Status":"Open"
                    })
        except: pass
    if to_add:
        sc = pd.concat([sc, pd.DataFrame(to_add)], ignore_index=True)
        save_scorecard(sc)
    st.dataframe(sc.tail(50), use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────
# Portfolio (allocations linked to theme weights; regime-aware NAV)
# ──────────────────────────────────────────────────────────────────────────
with tab_port:
    st.subheader("Portfolio — Auto allocations from Theme Weights & Regime")
    th = st.session_state.get("__themes__", load_themes()).copy()
    if th.empty:
        st.info("Please add some themes on Theme_Scores tab.")
    else:
        th["Weight"] = pd.to_numeric(th["Score"], errors="coerce").fillna(0)
        th["Weight"] = th["Weight"] / (th["Weight"].sum() or 1.0)
        st.dataframe(th[["Theme","Category","Score","Weight"]], use_container_width=True)

    port = load_portfolio()
    vol_df = st.session_state.get("__vol__", pd.DataFrame())
    regime = "Neutral"
    if not vol_df.empty and (vol_df["Asset"]=="BTC").any():
        row = vol_df[vol_df["Asset"]=="BTC"].iloc[0]
        regime = regime_from_vol(row["IV_short"], row["IV_long"])
    st.metric("Current Regime", regime)

    # simple NAV update model: random noise + regime skew tilt + theme weight carry
    if st.button("Update NAV (toy)"):
        base = port.groupby("Asset").tail(1).set_index("Asset")["NAV"].to_dict()
        skew_boost = 0.002 if regime=="Calm/Carry (Contango)" else (-0.002 if regime=="Stress/Backwardation" else 0.0005)
        wsum = float(th["Weight"].sum()) if not th.empty else 0.0
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        new_rows=[]
        for a in ASSETS:
            nav = base.get(a, 100.0)
            shock = np.random.normal(0, 0.002)
            nav = nav * (1 + shock + skew_boost + 0.001*wsum)
            new_rows.append({"Timestamp":now,"Asset":a,"NAV":round(nav,3)})
        port = pd.concat([port, pd.DataFrame(new_rows)], ignore_index=True)
        save_portfolio(port)
        if st.session_state.get("SLACK_WEBHOOK"):
            post_slack(st.session_state["SLACK_WEBHOOK"], f"Regime: {regime} | NAV updated.")
        st.success("NAV updated.")
    if not port.empty:
        st.line_chart(port.pivot_table(index="Timestamp", columns="Asset", values="NAV"))

# ──────────────────────────────────────────────────────────────────────────
# Backtest (toy illustrative)
# ──────────────────────────────────────────────────────────────────────────
with tab_bt:
    st.subheader("Backtest (toy illustration)")
    np.random.seed(2)
    dates = pd.date_range(end=pd.Timestamp.utcnow(), periods=120)
    rows=[]
    for d in dates:
        for a in ASSETS:
            skew = np.random.normal(0.02, 0.015)
            pnl  = np.random.normal(0, 0.003) + (0.002 if skew>0 else -0.001)
            rows.append({"Date": d, "Asset": a, "Skew": skew, "PnL": pnl})
    bt = pd.DataFrame(rows)
    bt["NAV"] = 100 + bt.groupby("Asset")["PnL"].cumsum()*100
    st.line_chart(bt.pivot_table(index="Date", columns="Asset", values="NAV"))

# ──────────────────────────────────────────────────────────────────────────
# Dashboard (summary KPIs)
# ──────────────────────────────────────────────────────────────────────────
with tab_dash:
    st.subheader("Dashboard — Summary")
    vol_df = st.session_state.get("__vol__", pd.DataFrame())
    th = st.session_state.get("__themes__", load_themes())
    sc = load_scorecard()
    cols = st.columns(3)
    with cols[0]:
        if not vol_df.empty and (vol_df["Asset"]=="BTC").any():
            row = vol_df[vol_df["Asset"]=="BTC"].iloc[0]
            st.metric("BTC 1w IV", f"{row['IV_short'] if pd.notnull(row['IV_short']) else '—'}")
            st.metric("BTC 1m IV", f"{row['IV_long'] if pd.notnull(row['IV_long']) else '—'}")
            st.metric("Term Skew", f"{row['Term_Skew'] if pd.notnull(row['Term_Skew']) else '—'}")
        else:
            st.info("Open Vol_Monitor to populate BTC IVs.")
    with cols[1]:
        if not th.empty:
            top = th.sort_values("Score", ascending=False).head(5)[["Theme","Score"]]
            st.write("Top Themes")
            st.dataframe(top, use_container_width=True)
        else:
            st.info("Add themes in Theme_Scores.")
    with cols[2]:
        st.write("Trade Scorecard (latest)")
        st.dataframe(sc.tail(10), use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────────────
with tab_diag:
    st.subheader("Diagnostics")
    diag = st.session_state.get("__diag__", {})
    if not diag:
        st.info("Open Vol_Monitor first to populate diagnostics.")
    else:
        for a, logs in diag.items():
            st.markdown(f"### {a}")
            for i, entry in enumerate(logs):
                st.text(f"[{i}] {entry['status']} — {entry['url']}")
                st.code(entry['preview'])
