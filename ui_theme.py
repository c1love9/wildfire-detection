"""Presentation styling shared by the image and live-monitoring workspaces."""
from html import escape
import streamlit as st


def apply_theme():
    st.markdown('''<style>
    :root {--fire-accent:#ff784f; --fire-cyan:#62d6da;}
    [data-testid="stAppViewContainer"] {background:radial-gradient(ellipse at 80% 0%,#132438 0%,#0b1220 58%);}
    [data-testid="stHeader"] {background:rgba(11,18,32,.9);}
    .block-container {max-width:1560px;padding:1.6rem 2.2rem 3rem;}
    [data-testid="stSidebar"] {background:#0d1728;border-right:1px solid #243249;}
    h1,h2,h3,p,label {font-family:"Microsoft YaHei","Segoe UI",sans-serif;}
    h3 {font-size:1.13rem!important;letter-spacing:.02em;}
    [data-testid="stCaptionContainer"] {color:#9aacc1;line-height:1.7;}
    [data-testid="stMetric"] {padding:14px 16px;background:#121f32;border:1px solid #26354c;border-radius:12px;}
    [data-testid="stMetricLabel"] {color:#a8b8cb;font-size:.85rem;}
    [data-testid="stMetricValue"] {font-family:"Segoe UI",sans-serif;font-size:1.65rem;color:#eef5ff;}
    [data-testid="stVerticalBlockBorderWrapper"] {border-radius:16px!important;}
    [data-testid="stExpander"] {border-color:#29384f;border-radius:11px;background:rgba(18,31,50,.45);}
    [data-baseweb="select"]>div {background:#162338;border-color:#2a3b53;border-radius:9px;}
    .stButton>button {border-radius:9px;min-height:2.65rem;font-weight:600;}
    button[kind="primary"] {background:linear-gradient(110deg,#c84e2e,#b73b2b);border-color:#ef6b45;color:white;}
    [data-testid="stTabs"] [data-baseweb="tab-list"] {gap:1.2rem;border-bottom:1px solid #27364b;}
    [data-testid="stTabs"] [aria-selected="true"] {color:#ff9976;}
    [data-testid="stImage"] img {border-radius:11px;}
    [data-testid="stElementContainer"]:has([data-testid="stImage"]),
    [data-testid="stFullScreenFrame"]:has([data-testid="stImage"]),
    [data-testid="stFullScreenFrame"]:has([data-testid="stImage"])>div,
    [data-testid="stImage"], [data-testid="stImageContainer"], [data-testid="stImageCaption"]
      {width:100%!important;max-width:100%!important;min-width:0!important;}
    [data-testid="stImage"] img {width:100%!important;max-width:100%!important;height:auto;}
    .fire-hero {display:flex;justify-content:space-between;gap:24px;align-items:center;padding:25px 30px;margin-bottom:22px;
      border:1px solid #2c3e55;border-radius:18px;background:linear-gradient(115deg,#192a40,#122034 65%,#1c2c38);}
    .fire-kicker {font-size:11px;letter-spacing:.22em;color:#ff9f7c;font-weight:700;margin-bottom:10px;}
    .fire-hero h1 {font-size:clamp(25px,2.5vw,35px);font-weight:700;letter-spacing:.035em;margin:0 0 10px;padding:0;color:#f5f8fc;}
    .fire-hero p {margin:0;max-width:790px;color:#adbed1;font-size:14px;line-height:1.8;}
    .fire-mark {width:65px;height:65px;flex-shrink:0;border-radius:18px;display:flex;align-items:center;justify-content:center;
      font-size:30px;background:#ff784f17;border:1px solid #ff784f40;}
    .fire-tags {display:flex;gap:10px;flex-wrap:wrap;margin-top:15px;}
    .fire-tags span {border:1px solid #34475e;background:#172b40;border-radius:30px;padding:4px 11px;color:#c7d9e9;font-size:11px;}
    .fire-state {display:flex;justify-content:space-between;align-items:center;padding:12px 16px;margin:4px 0 14px;border-radius:10px;
      border:1px solid #34445a;background:#142338;font-size:13px;color:#a9bdce;gap:12px;}
    .fire-state strong {color:#76dcca;white-space:nowrap;}
    .fire-state strong:before {content:"";display:inline-block;width:7px;height:7px;border-radius:50%;background:currentColor;margin-right:8px;}
    .fire-state.busy strong {color:#ffb88a;}
    @media(max-width:850px) {.block-container{padding:1rem;} .fire-hero{padding:20px;} .fire-mark{display:none;}}
    </style>''', unsafe_allow_html=True)


def page_header(title, description, section='LIVE MONITORING', tags=()):
    labels = ''.join(f'<span>{escape(label)}</span>' for label in tags)
    st.markdown(f'''<div class="fire-hero"><div><div class="fire-kicker">FIREWATCH / {escape(section)}</div>
      <h1>{escape(title)}</h1><p>{escape(description)}</p><div class="fire-tags">{labels}</div></div>
      <div class="fire-mark">◈</div></div>''', unsafe_allow_html=True)


def state_banner(label, detail, busy=False):
    st.markdown(f'<div class="fire-state{" busy" if busy else ""}"><strong>{escape(label)}</strong>'
                f'<span>{escape(detail)}</span></div>', unsafe_allow_html=True)
