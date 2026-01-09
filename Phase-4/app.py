# Phase-4/app.py
from __future__ import annotations

import json
import os
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

from loaders.bundle_loader import load_latest_bundle, BundleNotFoundError

# Loading env
load_dotenv()

# ----------------------------
# OpenAI client
# ----------------------------
def make_question_context(bundle: dict, question: str) -> str:
    forecasts = bundle.get("forecasts", {})
    insights = bundle.get("insights", {})
    recs = bundle.get("recommendations", [])

    q = question.lower()

    ctx = {
        "generated_at_utc": bundle.get("generated_at_utc"),
    }

    if "price" in q or "spike" in q:
        ctx["price"] = {
            "price_1h": forecasts.get("price_1h"),
            "price_insights": insights.get("price"),
        }

    if "demand" in q:
        ctx["demand"] = {
            "demand_24h_first5": forecasts.get("demand_24h", [])[:5],
            "demand_insights": insights.get("demand"),
        }

    ctx["recommendations"] = recs[:2]

    return json.dumps(ctx, indent=2)


def ask_engy(question: str, bundle: dict) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "ERROR: OPENAI_API_KEY is not set in your environment."

    client = OpenAI(api_key=api_key)

    context = make_question_context(bundle, question)

    resp = client.responses.create(
        model="gpt-4.1-mini",  # faster than gpt-5-mini
        input=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": f"PHASE-3 BUNDLE JSON (ONLY SOURCE):\n{context}"},
            {"role": "user", "content": f"Question: {question}"},
        ],
    )
    return resp.output_text

# ----------------------------
# LLM prompt rules (STRICT)
# ----------------------------
def build_system_prompt() -> str:
    return (
        "You are Engy, a Phase-4 Decision Copilot for Energy Intelligence.\n"
        "SOURCE RULES (STRICT):\n"
        "- Use ONLY the provided Phase-3 bundle JSON. No outside knowledge. No browsing.\n"
        "- If info is missing, output: Not available in this Phase-4 build.\n"
        "- Do NOT create new forecasts.\n\n"
        "OUTPUT RULES (STRICT):\n"
        "Return ONLY valid JSON (no markdown, no extra text) with this exact schema:\n"
        "{\n"
        '  "answer": string,                 \n'
        '  "actions": [string],              \n'
        '  "references": [                   \n'
        "     {\"field\": string, \"value\": string}\n"
        "  ]\n"
        "}\n\n"
        "STYLE:\n"
        "- answer: plain English, 1–3 short sentences, no bundle field names.\n"
        "- actions: MAX 3 items, concrete steps the user can do.\n"
        "- references: MAX 4 items. Use bundle paths like forecasts.price_1h[0].forecast_ts.\n"
        "- Put raw timestamps and numeric values ONLY in references, not in answer.\n"
        "- If user asks 'why' or 'evidence', keep answer brief and rely on references.\n"
    )

def render_engy_response(raw_text: str) -> str:
    """
    Converts Engy's JSON output into a clean user-facing response:
    - Plain answer
    - Actions
    - References (at the end)
    """
    try:
        data = json.loads(raw_text)
    except Exception:
        # fallback: if model didn't follow JSON contract
        return raw_text

    answer = (data.get("answer") or "").strip()
    actions = data.get("actions") or []
    refs = data.get("references") or []

    out_lines = []
    if answer:
        out_lines.append(answer)

    if actions:
        out_lines.append("\n**What to do**")
        for a in actions[:5]:
            out_lines.append(f"- {str(a).strip()}")

    if refs:
        out_lines.append("\n**References (Phase-3 bundle)**")
        for r in refs[:6]:
            field = str(r.get("field", "")).strip()
            value = str(r.get("value", "")).strip()
            if field and value:
                out_lines.append(f"- `{field}` = {value}")
            elif field:
                out_lines.append(f"- `{field}`")

    return "\n".join(out_lines).strip()


def make_bundle_context(bundle: dict) -> str:
    safe = {
        "generated_at_utc": bundle.get("generated_at_utc"),
        "as_of": bundle.get("as_of"),
        "limitations": bundle.get("limitations"),
        "thresholds": bundle.get("thresholds"),
        "insights": bundle.get("insights"),
        "recommendations": bundle.get("recommendations"),
        "forecasts": {
            "price_1h": (bundle.get("forecasts") or {}).get("price_1h"),
            "demand_24h_first5": ((bundle.get("forecasts") or {}).get("demand_24h", [])[:5]),
            "demand_24h_len": len(((bundle.get("forecasts") or {}).get("demand_24h", []) or [])),
        },
    }
    return json.dumps(safe, indent=2)

# ----------------------------
# Bundle parsing helpers
# ----------------------------
def to_display(v: Any) -> str:
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v)
    return str(v)


def demand_24h_to_df(demand_obj) -> pd.DataFrame:
    if demand_obj is None:
        return pd.DataFrame()

    if isinstance(demand_obj, list) and demand_obj and isinstance(demand_obj[0], dict):
        df = pd.DataFrame(demand_obj)
    elif isinstance(demand_obj, dict) and "points" in demand_obj:
        df = pd.DataFrame(demand_obj["points"])
    elif isinstance(demand_obj, dict):
        df = pd.DataFrame(demand_obj)
    else:
        return pd.DataFrame()

    if df.empty:
        return df

    if "forecast_ts" in df.columns and "yhat_demand_mw" in df.columns:
        out = df[["forecast_ts", "yhat_demand_mw"]].copy()
        out = out.rename(columns={"forecast_ts": "ts_utc", "yhat_demand_mw": "demand"})
        out["ts_utc"] = pd.to_datetime(out["ts_utc"], errors="coerce", utc=True)
        out["demand"] = pd.to_numeric(out["demand"], errors="coerce")
        out = out.dropna(subset=["ts_utc", "demand"]).sort_values("ts_utc")
        return out

    return df


def price_1h_to_value(price_obj):
    if price_obj is None:
        return None

    if isinstance(price_obj, list) and price_obj:
        row = price_obj[0]
        if isinstance(row, dict) and "yhat_price_per_mwh" in row:
            try:
                return float(row["yhat_price_per_mwh"]) / 1000.0
            except Exception:
                return None

    if isinstance(price_obj, dict) and "yhat_price_per_mwh" in price_obj:
        try:
            return float(price_obj["yhat_price_per_mwh"]) / 1000.0
        except Exception:
            return None

    if isinstance(price_obj, (int, float)):
        return float(price_obj)

    return None


def get_as_of(bundle: Dict[str, Any]) -> Dict[str, Any]:
    ao = bundle.get("as_of", {})
    return ao if isinstance(ao, dict) else {}


def pick_recommendation(bundle: dict) -> str:
    recs = bundle.get("recommendations", [])
    if not isinstance(recs, list) or not recs:
        return "Not available in this Phase-4 build."

    for r in recs:
        if isinstance(r, dict):
            txt = r.get("message") or r.get("text") or r.get("recommendation")
            if txt and "price" in str(txt).lower():
                return str(txt)

    r0 = recs[0]
    if isinstance(r0, dict):
        return str(r0.get("message") or r0.get("text") or r0.get("recommendation") or r0)
    return str(r0)


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Energy Intelligence — Phase 4", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
:root{
  --bg: #0B0F17;
  --card: rgba(255,255,255,0.05);
  --card2: rgba(255,255,255,0.035);
  --stroke: rgba(255,255,255,0.10);
  --muted2: rgba(255,255,255,0.55);
  --shadow: 0 18px 40px rgba(0,0,0,.45);
  --radius: 20px;
}

html, body, [data-testid="stAppViewContainer"]{
  background: radial-gradient(1200px 900px at 20% 0%, rgba(124,77,255,0.10), transparent 60%),
              radial-gradient(900px 700px at 90% 10%, rgba(0,255,170,0.07), transparent 55%),
              var(--bg) !important;
}

.block-container{
  padding-top: 3.75rem !important;
  padding-bottom: 2.5rem !important;
  max-width: 1400px;
}

/* ---------------------------------------
   IMPORTANT: Streamlit 1.52.2 “Card” style
   Use bordered containers instead of HTML wrappers.
---------------------------------------- */
div[data-testid="stVerticalBlockBorderWrapper"]{
  background: linear-gradient(180deg, var(--card), var(--card2)) !important;
  border: 1px solid var(--stroke) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow) !important;
}

/* Add padding inside bordered containers */
div[data-testid="stVerticalBlockBorderWrapper"] > div{
  padding: 18px 18px 14px 18px !important;
  border-radius: var(--radius) !important;
}

/* Title bar */
.titlebar{
  margin-top: 10px;
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap: 14px;
  padding: 18px 18px;
  border-radius: var(--radius);
  border: 1px solid var(--stroke);
  background: rgba(255,255,255,0.04);
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
  margin-bottom: 16px;
}

.brand{ display:flex; align-items:center; gap: 12px; }
.brand-icon{
  width: 42px; height: 42px;
  border-radius: 14px;
  display:flex; align-items:center; justify-content:center;
  background: rgba(255,165,0,0.16);
  border: 1px solid rgba(255,165,0,0.30);
}
.brand-title{ font-size: 32px; font-weight: 900; line-height: 1.1; margin: 0; }
.brand-sub{ color: var(--muted2); font-size: 13px; margin-top: 2px; }

.badge{
  display:inline-flex;
  align-items:center;
  gap:.4rem;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid var(--stroke);
  background: rgba(255,255,255,0.06);
  color: rgba(255,255,255,0.78);
  font-size: 12px;
}
.pill-row{ display:flex; gap: 10px; flex-wrap: wrap; margin-top: 6px; }

.small-muted { color: var(--muted2); font-size: 13px; }
.kpi { font-size: 54px; font-weight: 850; line-height: 1.0; margin: 0.35rem 0 0.10rem 0; }
.divider{ height: 1px; background: rgba(255,255,255,0.10); margin: 14px 0; }

[data-testid="stPlotlyChart"] > div{ border-radius: 16px; overflow: hidden; }

section[data-testid="stSidebar"]{
  background: rgba(255,255,255,0.02) !important;
  border-right: 1px solid rgba(255,255,255,0.06);
}

/* Engy header */
.engy-head{ display:flex; align-items:center; justify-content:center; gap: 14px; margin: 6px auto 6px auto; }
.engy-logo{
  width: 54px; height: 54px; border-radius: 18px;
  display:flex; align-items:center; justify-content:center;
  font-size: 24px;
  background: rgba(124,77,255,0.18);
  border: 1px solid rgba(124,77,255,0.35);
}
.engy-title{
  font-size: 22px;
  font-weight: 900;
  letter-spacing: -0.02em;
  color: rgba(255,255,255,0.95);
}
.center-banner{
  max-width: 920px;
  margin: 12px auto 10px auto;
  text-align: center;
  font-weight: 650;
  color: rgba(255,255,255,0.90);
}
.section-divider{
  width: 420px;
  height: 1px;
  margin: 22px auto 26px auto;
  background: linear-gradient(90deg, rgba(255,255,255,0.0), rgba(255,255,255,0.30), rgba(255,255,255,0.0));
}

/* Center chat input */
div[data-testid="stChatInput"]{
  max-width: 920px;
  margin: 0 auto 22px auto;
}
div[data-testid="stChatInput"] textarea{
  font-size: 16px !important;
  line-height: 1.35 !important;
  padding: 14px 14px !important;
  border-radius: 16px !important;
}
div[data-testid="stChatInput"] > div{ border-radius: 18px !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Sidebar: bundle selection 
# ----------------------------
outputs_dir = "../Phase-3/outputs"
pattern = "phase3_bundle_*.json"
show_debug = False

try:
    bundle, bundle_path = load_latest_bundle(outputs_dir=outputs_dir, pattern=pattern)
except BundleNotFoundError as e:
    st.error(str(e))
    st.stop()

st.sidebar.caption("Loaded")
st.sidebar.code(bundle_path)

# ----------------------------
# Title bar
# ----------------------------
st.markdown(
    """
<div class="titlebar">
  <div class="brand">
    <div class="brand-icon">⚡</div>
    <div>
      <div class="brand-title">Energy Intelligence — Decision Dashboard</div>
      <div class="brand-sub">Grounded in Phase-3 bundle • Snapshot mode • No live data</div>
    </div>
  </div>
  <div class="pill-row">
    <span class="badge">Bundle: latest</span>
    <span class="badge">Phase-4 UI</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Extract forecasts
# ----------------------------
forecasts = bundle.get("forecasts", {}) if isinstance(bundle.get("forecasts", {}), dict) else {}
demand_obj = forecasts.get("demand_24h")
price_obj = forecasts.get("price_1h")

demand_df = demand_24h_to_df(demand_obj)
price_value = price_1h_to_value(price_obj)

peak_demand = float(demand_df["demand"].max()) if ("demand" in demand_df.columns and not demand_df.empty) else None
avg_demand = float(demand_df["demand"].mean()) if ("demand" in demand_df.columns and not demand_df.empty) else None

as_of = get_as_of(bundle)
price_asof = to_display(as_of.get("price_forecast_ts_utc", "Not available"))

# ----------------------------
# Engy header + banner
# ----------------------------
st.markdown(
    """
<div class="engy-head">
  <div class="engy-logo">🤖</div>
  <div class="engy-title">Engy — AI Decision Copilot</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="center-banner">
  Ask me anything about this bundle. I will only use Phase-3 fields and will cite them.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ----------------------------
# Chat
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages[-10:]:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if st.session_state.messages:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

user_msg = st.chat_input("Ask Engy Anything!")
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            raw = ask_engy(user_msg, bundle)
            answer = render_engy_response(raw)
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ----------------------------
# Main row: Price card (left) + Demand card (right)
# ----------------------------
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    with st.container(border=True):
        st.markdown("#### 💲 Next Hour Energy Pricing")

        if price_value is None:
            st.markdown("<div class='kpi'>—</div>", unsafe_allow_html=True)
            st.caption("Not available in this Phase-4 build.")
        else:
            st.markdown(
                f"<div class='kpi'>${price_value:,.3f} <span class='small-muted'>/kWh</span></div>",
                unsafe_allow_html=True,
            )
            st.caption("Converted from $/MWh ÷ 1000 for display.")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        st.markdown("<div class='small-muted'>AS-OF (UTC)</div>", unsafe_allow_html=True)
        st.write(price_asof)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        st.markdown("**💡 Recommendation**")
        st.write(pick_recommendation(bundle))

with col_right:
    with st.container(border=True):
        st.markdown("#### 📉 24-Hour Demand Forecast")

        k1, k2 = st.columns(2)
        k1.metric("Peak Demand", f"{peak_demand:,.1f} MW" if peak_demand is not None else "Not available")
        k2.metric("Average Demand", f"{avg_demand:,.1f} MW" if avg_demand is not None else "Not available")

        if demand_df.empty or "ts_utc" not in demand_df.columns or "demand" not in demand_df.columns:
            st.warning("24-hour demand forecast not available (or schema not recognized) in this bundle.")
        else:
            fig = px.line(demand_df, x="ts_utc", y="demand")
            fig.update_traces(
                hovertemplate="Time (UTC): %{x}<br>Demand: %{y:,.0f} MW<extra></extra>",
                line=dict(width=3),
            )
            fig.update_layout(
                template="plotly_dark",
                height=430,
                margin=dict(l=8, r=8, t=10, b=10),
                xaxis_title=None,
                yaxis_title="Demand (MW)",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        insights = bundle.get("insights", {}) if isinstance(bundle.get("insights", {}), dict) else {}
        peak_windows = insights.get("peak_demand_windows") or insights.get("peak_windows")

        if peak_windows:
            st.info(f"**Peak Alert (bundle):** {to_display(peak_windows)}")
        else:
            st.caption("Peak alert: Not available in bundle.")

# ----------------------------
# Bottom strip
# ----------------------------
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
with st.container(border=True):
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("Current Load", "Not available")
    b2.metric("Grid Capacity", "Not available")
    b3.metric("Renewable Mix", "Not available")
    b4.metric("CO₂ Intensity", "Not available")
    b5.metric("Last Updated", to_display(bundle.get("generated_at_utc", "Not available")))

# ----------------------------
# Optional context/debug panels
# ----------------------------
with st.expander("Context (timestamps, limitations, snapshot mode)", expanded=False):
    as_of_obj = get_as_of(bundle)

    c1, c2, c3 = st.columns(3)
    c1.metric("Demand features max (UTC)", to_display(as_of_obj.get("demand_feature_ts_max_utc", "Not available")))
    c2.metric("Price features max (UTC)", to_display(as_of_obj.get("price_feature_ts_max_utc", "Not available")))
    c3.metric("Price forecast ts (UTC)", to_display(as_of_obj.get("price_forecast_ts_utc", "Not available")))

    window = as_of_obj.get("demand_forecast_window_utc")
    if isinstance(window, (list, tuple)) and len(window) == 2:
        st.write(f"**Demand forecast window (UTC):** {window[0]} → {window[1]}")
    else:
        st.write("**Demand forecast window (UTC):** Not available")

    st.info(
        "Frozen snapshot mode: Phase-3 outputs are generated from a frozen dataset snapshot for reproducibility. "
        "Phase-4 displays only what exists in the bundle and does not generate new forecasts."
    )

    limitations = bundle.get("limitations")
    if limitations is not None:
        st.markdown("**Limitations (from bundle)**")
        st.write(limitations)

if show_debug:
    with st.expander("DEBUG: forecast payload shapes (temporary)", expanded=False):
        st.write("demand_24h type:", type(demand_obj))
        st.write("price_1h type:", type(price_obj))

        if isinstance(demand_obj, list) and demand_obj:
            st.write("demand_24h[0] keys:", list(demand_obj[0].keys()))
            st.write("demand_24h[0] sample:", demand_obj[0])
        elif isinstance(demand_obj, dict):
            st.write("demand_24h dict keys:", list(demand_obj.keys()))

        if isinstance(price_obj, list) and price_obj and isinstance(price_obj[0], dict):
            st.write("price_1h[0] keys:", list(price_obj[0].keys()))
            st.write("price_1h[0] sample:", price_obj[0])
            st.write("Raw price ($/MWh):", price_obj[0].get("yhat_price_per_mwh"))
            st.write("Forecast ts:", price_obj[0].get("forecast_ts"))
        else:
            st.write("price_1h sample:", price_obj)
