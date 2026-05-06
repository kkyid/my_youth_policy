"""Main entry point - Home page with RAG search interface."""
from __future__ import annotations

import os
import json
import re
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# Page Config
st.set_page_config(
    page_title="청년 주택 정책 검색",
    layout="wide"
)

# Light imports
from core.ui import inject_ui

# Heavy imports
from core import vector_db, rag_engine, retrievers as retr, prompts as prompt_store

load_dotenv()

# ── LangSmith traceable (없으면 no-op) ───────────────────────────
try:
    from langsmith import traceable as _ls_traceable
except ImportError:
    try:
        from langsmith.run_helpers import traceable as _ls_traceable
    except ImportError:
        def _ls_traceable(**kw):  # type: ignore
            def _d(fn): return fn
            return _d

# UI Injection (this brings in the global 80% width constraint for the header)
inject_ui()


# ── DB 캐시 워밍업 (앱 최초 구동 시 1회만 실행) ──────────────────────
def _warmup_db_cache() -> None:
    """vectorstore + BM25 인덱스를 앱 시작 시점에 미리 빌드."""
    for col in (vector_db.HOUSING_COLLECTION, vector_db.FINANCE_COLLECTION):
        vector_db.get_cached_vectorstore(col)
        retr.get_cached_bm25_retriever(col)

if not st.session_state.get("_db_cache_ready"):
    with st.spinner("🔄 정책 DB를 불러오는 중입니다... (최초 1회)"):
        _warmup_db_cache()
    st.session_state["_db_cache_ready"] = True


def _policy_url(item: dict) -> str:
    """정책 URL 반환. 없으면 카테고리별 기본 URL."""
    url = (item.get("url") or "").strip()
    if url:
        return url
    if "금융" in (item.get("category") or ""):
        return "https://nhuf.molit.go.kr/"
    return "https://housing.seoul.go.kr/"


def _card_html(item: dict, i: int, disqualified: bool = False) -> str:
    """TOP3 카드 HTML 생성."""
    rank_class  = ["top1", "top2", "top3"][i] if i < 3 else "top3"
    rank_label  = f"TOP{i+1}"
    is_finance  = "금융" in (item.get("category") or "")
    cat_class   = "finance" if is_finance else "housing"
    cat_label   = item.get("category") or ("금융" if is_finance else "주택")
    summary     = item.get("summary") or ""
    reason      = item.get("reason")  or ""
    name        = item.get("policy_name") or "-"
    url         = _policy_url(item)
    link_color  = "#1D4ED8" if is_finance else "#6D28D9"
    disq_reason = item.get("disq_reason", "")

    disq_overlay = ""
    card_style   = ""
    if disqualified:
        disq_overlay = '''
<div style="position:absolute;inset:0;background:rgba(239,68,68,0.08);border-radius:18px;
            display:flex;align-items:center;justify-content:center;
            z-index:10;pointer-events:none;">
  <div style="font-size:7rem;line-height:1;color:#EF4444;font-weight:900;opacity:0.7;">✕</div>
</div>'''
        card_style = "opacity:0.55;filter:grayscale(40%);"

    return f'''<div class="policy-card {cat_class}" style="position:relative;{card_style}">
{disq_overlay}
<div class="rank-badge {rank_class}">{rank_label}</div>
<div style="margin-top:0.6rem;">
  <span class="cat-badge {cat_class}">{cat_label}</span>
</div>
<div class="policy-card-title">{name}</div>
<div class="card-summary">
  <span class="card-section-label">📌 핵심 내용</span>{summary}
</div>
<div class="card-reason">
  <span class="card-section-label">💡 선정 이유</span>{reason}
</div>
<div style="margin-top:auto; padding-top:10px; border-top:1px solid #F1F5F9; text-align:right;">
  <a href="{url}" target="_blank" style="color:{link_color}; text-decoration:none; font-weight:700; font-size:0.85rem;">바로가기 →</a>
</div>
</div>'''


def _detail_card_html(detail: dict) -> str:
    """정책 상세 정보 카드 HTML."""
    is_finance   = "금융" in (detail.get("category") or "")
    border_color = "#3B82F6" if is_finance else "#8B5CF6"
    hdr_color    = "#1D4ED8" if is_finance else "#6D28D9"
    name         = detail.get("policy_name") or "-"
    overview     = detail.get("overview") or ""
    eligibility  = detail.get("eligibility") or []
    benefits     = detail.get("benefits") or []
    how_to       = detail.get("how_to_apply") or ""
    deadline     = detail.get("deadline") or "별도 공고"
    docs         = detail.get("required_docs") or []
    caution      = detail.get("caution") or ""
    url          = (detail.get("url") or "").strip() or (
        "https://nhuf.molit.go.kr/" if is_finance else "https://housing.seoul.go.kr/"
    )

    def _ul(items):
        if not items:
            return "<span style='color:#94A3B8'>정보 없음</span>"
        return (
            "<ul style='margin:0;padding-left:1.2rem;color:#475569;font-size:1.0rem;line-height:1.8'>"
            + "".join(f"<li>{it}</li>" for it in items)
            + "</ul>"
        )

    caution_block = (
        "<div style='background:rgba(248,113,113,0.07);border-left:3px solid #f87171;"
        "border-radius:8px;padding:0.7rem 1rem;'>"
        "<div style='font-size:0.85rem;font-weight:700;color:#DC2626;margin-bottom:0.2rem;'>"
        "⚠️ 주의사항</div>"
        f"<div style='font-size:0.98rem;color:#475569;line-height:1.7'>{caution}</div>"
        "</div>"
    ) if caution else ""

    parts = [
        f"<div style='background:white;border:1px solid #E2E8F0;border-top:3px solid {border_color};",
        "border-radius:18px;padding:1.6rem 1.4rem;box-shadow:0 4px 18px rgba(0,0,0,0.06);",
        "display:flex;flex-direction:column;gap:1rem;'>",
        f"<div style='font-size:1.15rem;font-weight:700;color:#1E293B;border-bottom:1px solid #F1F5F9;",
        f"padding-bottom:0.7rem;'>{name}</div>",
        "<div style='background:#F8FAFC;border-left:3px solid #A78BFA;border-radius:8px;padding:0.8rem 1rem;'>",
        "<div style='font-size:0.85rem;font-weight:700;color:#7C3AED;margin-bottom:0.3rem;'>📌 정책 개요</div>",
        f"<div style='font-size:1.0rem;color:#334155;line-height:1.7'>{overview}</div></div>",
        "<div><div style='font-size:0.9rem;font-weight:700;color:#059669;margin-bottom:0.4rem;'>✅ 자격요건</div>",
        "<div style='background:rgba(52,211,153,0.07);border-left:3px solid #34d399;border-radius:8px;padding:0.7rem 1rem;'>",
        f"{_ul(eligibility)}</div></div>",
        "<div><div style='font-size:0.9rem;font-weight:700;color:#2563EB;margin-bottom:0.4rem;'>🎁 지원 혜택</div>",
        "<div style='background:rgba(59,130,246,0.07);border-left:3px solid #60a5fa;border-radius:8px;padding:0.7rem 1rem;'>",
        f"{_ul(benefits)}</div></div>",
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;'>",
        "<div style='background:#F8FAFC;border-radius:10px;padding:0.8rem 1rem;'>",
        "<div style='font-size:0.85rem;font-weight:700;color:#64748B;margin-bottom:0.3rem;'>📅 신청 기간</div>",
        f"<div style='font-size:0.98rem;color:#334155'>{deadline}</div></div>",
        "<div style='background:#F8FAFC;border-radius:10px;padding:0.8rem 1rem;'>",
        "<div style='font-size:0.85rem;font-weight:700;color:#64748B;margin-bottom:0.3rem;'>📎 필요 서류</div>",
        f"{_ul(docs)}</div></div>",
        "<div><div style='font-size:0.9rem;font-weight:700;color:#475569;margin-bottom:0.4rem;'>🗺️ 신청 방법</div>",
        f"<div style='font-size:1.0rem;color:#334155;line-height:1.7'>{how_to}</div></div>",
        caution_block,
        f"<div style='margin-top:auto;padding-top:0.8rem;border-top:1px solid #F1F5F9;text-align:right;'>",
        f"<a href='{url}' target='_blank' style='color:{hdr_color};text-decoration:none;font-weight:700;font-size:0.95rem;'>바로가기 →</a>",
        "</div></div>",
    ]
    return "".join(parts)


# --- Custom CSS for Home Page Layout (Overriding Global UI) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&family=Playfair+Display:wght@700&display=swap');

    /* core/ui.py에서 전역으로 적용한 본문 50% 너비 제한을 무효화하고 컨테이너(80%)를 꽉 채우도록 허용 */
    [data-testid="stVerticalBlock"] > div:nth-child(n+3) {
        max-width: 100% !important;
        width: 100% !important;
    }
    
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.5rem; font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 0.5rem 0; letter-spacing: -0.5px;
    }
    .sub-title {
        text-align: center; color: #64748B;
        font-size: 1.1rem; margin-bottom: 2rem; font-weight: 400;
    }

    /* 채팅 입력창을 화면의 50%로 중앙 정렬 (컨테이너 80% 내부에서 62.5% = 전체의 50%) */
    .stChatInputContainer {
        max-width: 62.5% !important;
        margin: 0 auto !important;
    }

    /* Policy Card (Light Theme adapted from provided code) */
    .policy-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 1.7rem 1.3rem 1.4rem;
        margin-bottom: 1rem;
        position: relative;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
        height: 100%;
        display: flex;
        flex-direction: column;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .policy-card {
        min-height: 360px;
    }
    .policy-card.finance { border-top: 3px solid #3B82F6; }
    .policy-card.housing { border-top: 3px solid #8B5CF6; }
    .policy-card.finance:hover {
        border-color: #3B82F6;
        box-shadow: 0 8px 24px rgba(59,130,246,0.15);
        transform: translateY(-2px);
    }
    .policy-card.housing:hover {
        border-color: #8B5CF6;
        box-shadow: 0 8px 24px rgba(139,92,246,0.15);
        transform: translateY(-2px);
    }
    .cat-badge {
        font-size: 0.75rem; font-weight: 700;
        padding: 0.18rem 0.7rem; border-radius: 20px;
        display: inline-block; margin-bottom: 0.45rem; margin-top: 0.2rem;
        letter-spacing: 0.3px;
    }
    .cat-badge.finance { background: #DBEAFE; color: #1D4ED8; }
    .cat-badge.housing { background: #EDE9FE; color: #6D28D9; }
    .rank-badge.top1 { background: linear-gradient(135deg, #fbbf24, #f59e0b); box-shadow: 0 4px 12px rgba(251,191,36,0.35); }
    .rank-badge.top2 { background: linear-gradient(135deg, #94a3b8, #64748b); box-shadow: 0 4px 12px rgba(148,163,184,0.3); }
    .rank-badge.top3 { background: linear-gradient(135deg, #cd7c4a, #a0522d); box-shadow: 0 4px 12px rgba(205,124,74,0.3); }
    .card-summary {
        background: #F8FAFC; border-left: 3px solid #A78BFA;
        border-radius: 8px; padding: 0.65rem 0.85rem;
        font-size: 0.9rem; color: #334155; line-height: 1.65; margin-bottom: 0.6rem;
    }
    .card-pros {
        background: rgba(52,211,153,0.07); border-left: 3px solid #34d399;
        border-radius: 8px; padding: 0.65rem 0.85rem;
        font-size: 0.88rem; color: #334155; line-height: 1.65; margin-bottom: 0.5rem;
    }
    .card-cons {
        background: rgba(248,113,113,0.06); border-left: 3px solid #f87171;
        border-radius: 8px; padding: 0.65rem 0.85rem;
        font-size: 0.88rem; color: #334155; line-height: 1.65;
    }
    .card-section-label {
        font-size: 0.76rem; font-weight: 700; display: block; margin-bottom: 0.28rem;
    }
    .card-pros .card-section-label { color: #059669; }
    .card-cons .card-section-label { color: #DC2626; }
    .card-summary .card-section-label { color: #7C3AED; }
    .card-reason {
        background: rgba(251,191,36,0.07); border-left: 3px solid #fbbf24;
        border-radius: 8px; padding: 0.65rem 0.85rem;
        font-size: 0.88rem; color: #334155; line-height: 1.65;
    }
    .card-reason .card-section-label { color: #B45309; }
    .rank-badge {
        position: absolute; top: -14px; left: 18px;
        color: white; font-size: 0.85rem; font-weight: 700;
        padding: 0.4rem 1rem; border-radius: 20px;
    }

    .policy-card-title {
        font-size: 1.15rem; font-weight: 700; color: #1E293B;
        margin: 0.7rem 0 1rem; padding-bottom: 0.7rem;
        border-bottom: 1px solid #F1F5F9;
        min-height: 3rem;
    }
    .policy-section { margin-bottom: 0.9rem; font-size: 0.95rem; line-height: 1.6; }
    .policy-section-label {
        color: #64748B; font-size: 0.85rem; font-weight: 600;
        margin-bottom: 0.35rem; display: block;
    }
    .policy-section-value { color: #334155; font-weight: 500; }
    /* ── Chat Area Container ── */
    .chat-container {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 22px;
        padding: 1rem;
        box-shadow: 0 6px 24px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }

    /* ── 보고서 ──────────────────────────────────── */
    .report-hero {
        background: linear-gradient(135deg, rgba(167,139,250,0.15), rgba(96,165,250,0.1));
        border: 1px solid rgba(167,139,250,0.25);
        border-radius: 22px; padding: 2.2rem; margin: 1rem 0 1.5rem;
        text-align: center; position: relative; overflow: hidden;
        box-shadow: 0 6px 24px rgba(0,0,0,0.05);
    }
    .report-hero::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 5px;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399, #fbbf24);
    }
    .report-hero-title {
        font-family: 'Playfair Display', serif; font-size: 1.9rem; font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;
    }
    .report-hero-summary { color: #475569; font-size: 1.08rem; line-height: 1.8; margin-top: 0.8rem; }
    .report-section-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 18px; padding: 1.7rem; margin-bottom: 1.3rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
    }
    .report-section-title {
        font-size: 1.25rem; font-weight: 700; color: #1E293B;
        margin-bottom: 1.1rem; display: flex; align-items: center; gap: 0.6rem;
    }
    .pros-box {
        background: rgba(52,211,153,0.08); border-left: 4px solid #34d399;
        border-radius: 12px; padding: 1.1rem 1.3rem;
    }
    .cons-box {
        background: rgba(248,113,113,0.08); border-left: 4px solid #f87171;
        border-radius: 12px; padding: 1.1rem 1.3rem; margin-top: 0.8rem;
    }
    .pros-title { color: #059669; font-weight: 700; font-size: 1rem; margin-bottom: 0.6rem; }
    .cons-title { color: #DC2626; font-weight: 700; font-size: 1rem; margin-bottom: 0.6rem; }
    .pros-box ul, .cons-box ul { margin: 0; padding-left: 1.3rem; color: #475569; font-size: 0.95rem; line-height: 1.8; }
    
    .report-table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; font-size: 0.98rem; }
    .report-table th {
        background: #F8FAFC; color: #475569;
        padding: 0.85rem 1rem; text-align: left;
        border-bottom: 2px solid #E2E8F0; font-weight: 700; font-size: 0.95rem;
    }
    .report-table td {
        padding: 0.85rem 1rem; color: #334155;
        border-bottom: 1px solid #F1F5F9; vertical-align: top; line-height: 1.7;
    }
    .report-table tr:last-child td { border-bottom: none; }
    
    .strategy-box {
        background: rgba(96,165,250,0.08); border-left: 4px solid #60a5fa;
        border-radius: 12px; padding: 1.2rem 1.4rem; color: #475569;
        font-size: 1rem; line-height: 1.8; margin-top: 0.5rem;
    }
    .recommend-box {
        background: linear-gradient(135deg, rgba(251,191,36,0.1), rgba(167,139,250,0.08));
        border: 1px solid rgba(251,191,36,0.3); border-radius: 16px;
        padding: 1.6rem; color: #1E293B; font-size: 1.05rem; line-height: 1.8; margin-top: 0.5rem;
    }
    .warning-box {
        background: rgba(248,113,113,0.08); border-left: 4px solid #f87171;
        border-radius: 12px; padding: 1.1rem 1.4rem; color: #475569;
        font-size: 1rem; line-height: 1.8; margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ── 채팅 히스토리 초기화 ──────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── 카카오톡 스타일 채팅 렌더링 (IFrame 방식) ───────────────
def render_chat_history():
    if not st.session_state.messages:
        return

    msgs_html = ""
    for msg in st.session_state.messages:
        role = msg["role"]
        text = msg["content"]
        if role == "user":
            msgs_html += f"""
            <div class="msg-row me">
              <div class="avatar me">U</div>
              <div class="bubble-col">
                <div class="bubble me">{text}</div>
              </div>
            </div>"""
        else:
            text_html = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            msgs_html += f"""
            <div class="msg-row bot">
              <div class="avatar bot">A</div>
              <div class="bubble-col">
                <div class="sender-name">친구</div>
                <div class="bubble bot">{text_html}</div>
              </div>
            </div>"""

    html = f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');
      * {{ box-sizing: border-box; margin: 0; padding: 0; }}
      body {{
        font-family: 'Noto Sans KR', sans-serif;
        background: transparent;
        padding: 10px;
        overflow-y: auto;
        overflow-x: hidden;
      }}
      body::-webkit-scrollbar {{ width: 6px; }}
      body::-webkit-scrollbar-thumb {{ background: #CBD5E1; border-radius: 3px; }}
      
      /* 채팅을 감싸는 흰색 컨테이너 스타일 */
      .chat-wrapper {{
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 22px;
        padding: 1.5rem 1rem;
        box-shadow: 0 6px 24px rgba(0,0,0,0.05);
        margin: 5px;
      }}
      
      .msg-row {{ display: flex; margin-bottom: 12px; align-items: flex-end; gap: 8px; width: 100%; }}
      .msg-row.me  {{ flex-direction: row-reverse; }}
      .msg-row.bot {{ flex-direction: row; }}
      
      .avatar {{
        width: 36px; height: 36px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem; font-weight: 700; flex-shrink: 0;
      }}
      .avatar.me  {{ background: linear-gradient(135deg,#a78bfa,#60a5fa); color: white; }}
      .avatar.bot {{ background: #F1F5F9; border: 1px solid #E2E8F0; color: #64748B; }}
      
      .bubble-col {{ display: flex; flex-direction: column; max-width: 68%; }}
      .msg-row.me  .bubble-col {{ align-items: flex-end; }}
      .msg-row.bot .bubble-col {{ align-items: flex-start; }}
      
      .sender-name {{ font-size: 0.7rem; color: #64748B; margin-bottom: 0.25rem; font-weight: 600; margin-left: 4px; }}
      
      .bubble {{
        padding: 0.75rem 1rem; border-radius: 18px;
        font-size: 0.87rem; line-height: 1.6; word-break: break-word;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
      }}
      .bubble.me {{
        background: linear-gradient(135deg,#a78bfa,#60a5fa);
        color: white; font-weight: 500;
        border-bottom-right-radius: 4px;
      }}
      .bubble.bot {{
        background: white;
        border: 1px solid #E2E8F0;
        color: #1E293B;
        border-bottom-left-radius: 4px;
      }}
    </style>
    <div class="chat-wrapper">
      <div id="chat">{msgs_html}</div>
    </div>
    <script>
      window.onload = function() {{
        window.scrollTo(0, document.body.scrollHeight);
      }};
    </script>
    """
    n = len(st.session_state.messages)
    # 컨테이너 패딩 등을 고려하여 높이 약간 조정 (채팅창 높이와 간격 조절)
    components.html(html, height=min(120 + n * 85, 520), scrolling=True)


# ── JSON 파싱 및 보고서 렌더링 헬퍼 ────────────────────────────────────────
def extract_json(text):
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        return match.group(1)
    start = text.find('{')
    end   = text.rfind('}')
    if start != -1 and end != -1:
        return text[start:end+1]
    return text

def add_linebreaks_after_period(text):
    if not isinstance(text, str):
        return text
    # 숫자로 시작하는 리스트 항목(예: "1. ", "2. ")을 보호하기 위해,
    # 마침표 바로 앞이 숫자가 아닌 경우에만 마침표 뒤 공백을 줄바꿈(<br>)으로 치환합니다.
    return re.sub(r'(?<!\d)\.\s+', '.<br>', text)

def render_list_html(items):
    if not items:
        return ""
    if isinstance(items, str):
        items = [items]
    return "<ul>" + "".join([f"<li>{add_linebreaks_after_period(item)}</li>" for item in items]) + "</ul>"

def render_report(data):
    summary   = add_linebreaks_after_period(data.get("summary", ""))
    hero_html = f"""
    <div class="report-hero">
        <div class="report-hero-title">📊 맞춤형 정책 종합 보고서</div>
        <div style="color:#475569;font-size:0.95rem;">AI가 분석한 당신을 위한 최적의 정책 가이드</div>
        <div class="report-hero-summary">{summary}</div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)

    policy_analysis = data.get("policy_analysis", [])
    if policy_analysis:
        # ── 📋 정책별 분석 요약 테이블 ─────────────────────
        rows = "".join([
            f'<tr>'
            f'<td><strong style="color:#1E293B">{p.get("title","-")}</strong></td>'
            f'<td style="color:#{"059669" if p.get("type")=="금융" else "2563EB"}">{p.get("type","-")}</td>'
            f'<td>{add_linebreaks_after_period(p.get("core","-"))}</td>'
            f'</tr>'
            for p in policy_analysis
        ])
        st.markdown(f"""
        <div class="report-section-card">
            <div class="report-section-title">📋 정책별 분석 요약</div>
            <table class="report-table">
                <thead><tr><th style="width:28%">정책명</th><th style="width:12%">유형</th><th>핵심 내용</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # ── 장단점 비교: 단일 HTML 그리드 (equal-height 보장) ──────
        cards_html = ""
        for p in policy_analysis:
            pros_html = render_list_html(p.get("pros", []))
            cons_html = render_list_html(p.get("cons", []))
            is_fin    = p.get("type") == "금융"
            hdr_color = "#1D4ED8" if is_fin else "#6D28D9"
            bg_color  = "#DBEAFE" if is_fin else "#EDE9FE"
            cards_html += f"""
            <div style="background:white; border:1px solid #E2E8F0; border-radius:16px;
                        padding:1.4rem; box-shadow:0 4px 14px rgba(0,0,0,0.05);
                        display:flex; flex-direction:column; gap:0.8rem;">
              <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.2rem;">
                <span style="background:{bg_color}; color:{hdr_color}; font-size:0.75rem;
                             font-weight:700; padding:0.15rem 0.6rem; border-radius:20px; white-space:nowrap; flex-shrink:0;">
                  {p.get("type", "정책")}
                </span>
                <span style="font-size:1rem; font-weight:700; color:#1E293B;">
                  {p.get("title", "-")}
                </span>
              </div>
              <div style="background:rgba(52,211,153,0.07); border-left:3px solid #34d399;
                          border-radius:8px; padding:0.8rem 1rem;">
                <div style="color:#059669; font-size:0.8rem; font-weight:700; margin-bottom:0.4rem;">✅ 장점</div>
                {pros_html}
              </div>
              <div style="background:rgba(248,113,113,0.06); border-left:3px solid #f87171;
                          border-radius:8px; padding:0.8rem 1rem; margin-top:auto;">
                <div style="color:#DC2626; font-size:0.8rem; font-weight:700; margin-bottom:0.4rem;">⚠️ 단점 · 유의사항</div>
                {cons_html}
              </div>
            </div>"""
        n_cols = len(policy_analysis)
        html_content = f"""
        <div class="report-section-card">
          <div class="report-section-title">📊 장단점 비교</div>
          <div style="display:grid; grid-template-columns:repeat({n_cols},1fr); gap:1rem; align-items:stretch;">
            {cards_html}
          </div>
        </div>
        """
        st.markdown(html_content.replace('\n', ''), unsafe_allow_html=True)

    # ── 정책 조합 전략 ─────────────────────────────────────
    combination = add_linebreaks_after_period(data.get("combination", ""))
    if combination:
        st.markdown(f"""
        <div class="report-section-card">
            <div class="report-section-title">🔗 정책 조합 전략</div>
            <div class="strategy-box">{combination}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── 주의사항 및 리스크 ─────────────────────────────────
    risks = data.get("risks", "")
    if risks:
        risks_html = render_list_html(risks) if isinstance(risks, list) else add_linebreaks_after_period(risks)
        st.markdown(f"""
        <div class="report-section-card">
            <div class="report-section-title">⚠️ 주의사항 및 리스크</div>
            <div class="warning-box">{risks_html}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── 종합 추천 및 행동 계획 ─────────────────────────────
    recommendation = add_linebreaks_after_period(data.get("recommendation", ""))
    if recommendation:
        st.markdown(f"""
        <div class="report-section-card">
            <div class="report-section-title">🎯 종합 추천 및 행동 계획</div>
            <div class="recommend-box">{recommendation}</div>
        </div>
        """, unsafe_allow_html=True)


# ── 맨 위 스크롤 앵커 ────────────────────────────────────────
# ── 메인 화면 ─────────────────────────────────────────────
st.markdown('<div class="sub-title">나에게 맞는 정책을 찾고 종합 보고서를 받아보세요</div>', unsafe_allow_html=True)

# 1. Chat Area (화면의 50% 너비로 고정)
col_l, col_chat, col_r = st.columns([1.5, 5, 1.5])
with col_chat:
    chat_slot = st.empty()

with chat_slot:
    render_chat_history()

# 2. Chat Input (하단 고정, CSS로 62.5% 처리됨 = 화면의 50%)
prompt = st.chat_input("질문을 입력해주세요.")

if prompt:
    if st.session_state.get("ask_mode"):
        # 추가 정보 응답 → 원래 질문에 합쳐서 파이프라인 진행
        base_q = st.session_state.pop("base_question", "")
        combined = base_q + "\n추가 정보: " + prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.ask_mode = False
        st.session_state.pending_prompt = combined
    elif "last_res" in st.session_state:
        # ── Top3+보고서 출력 이후 follow-up 처리 ────────────────
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.followup_prompt = prompt
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.pending_prompt = prompt
    st.rerun()

# 2.5-A Follow-up 처리 (Top3 이후 후속 대화)
if "followup_prompt" in st.session_state:
    followup = st.session_state.pop("followup_prompt")
    res       = st.session_state.get("last_res", {})
    top3      = res.get("top3", [])
    contexts  = res.get("contexts_text", [])
    orig_q    = res.get("original_question") or res.get("housing_query") or ""

    # 원본 질문 + 후속 메시지를 합쳐서 LLM에 전달 (맥락 유지)
    full_context = f"[원래 질문] {orig_q}\n[추가 메시지] {followup}"

    with st.spinner("메시지를 분석하는 중..."):
        intent_result = rag_engine.detect_followup_intent(full_context, top3)

    intent   = intent_result.get("intent", "other")
    policies = intent_result.get("policies", [])

    if intent == "qualify":
        with st.spinner("자격요건을 확인하는 중..."):
            qual_list = rag_engine.check_qualification(full_context, top3, contexts)
        disq = [q["policy_name"] for q in qual_list if q.get("status") == "disqualified"]
        # 판단 이유 top3에 주입
        reason_map = {q["policy_name"]: q.get("reason", "") for q in qual_list}
        for p in top3:
            p["disq_reason"] = reason_map.get(p.get("policy_name", ""), "")
        st.session_state.disqualified_names = disq
        st.session_state.last_res["top3"] = top3
        reply = (
            f"확인했습니다! "
            + (f"**{', '.join(disq)}** 정책은 제공해 주신 조건으로는 해당되지 않을 수 있어요." if disq
               else "제공해 주신 조건으로는 Top3 정책 모두 해당 가능성이 있습니다.")
            + "\n\n자세히 알고 싶은 정책이 있으시면 아래의 상세보기를 눌러주세요!"
        )
        st.session_state.messages.append({"role": "assistant", "content": reply})

    elif intent == "detail" and policies:
        # 캐시 확인 후 없으면 on-demand fetch
        cache = st.session_state.get("policy_details_cache", {})
        missing = [p for p in policies if p not in cache]
        if missing:
            with st.spinner(f"정책 상세 정보를 불러오는 중... ({len(missing)}개)"):
                for name in missing:
                    cache[name] = rag_engine.get_policy_detail(orig_q, name, contexts)
        st.session_state.policy_details_cache = cache
        st.session_state.detail_mode = True
        st.session_state.detail_policies = policies
        reply = f"**{', '.join(policies)}** 정책의 상세 정보를 불러왔어요."
        st.session_state.messages.append({"role": "assistant", "content": reply})

    else:
        reply = "죄송해요, 잘 이해하지 못했어요. 자격요건 정보를 추가로 알려주시거나, 자세히 알고 싶은 정책명을 말씀해주세요."
        st.session_state.messages.append({"role": "assistant", "content": reply})

    st.rerun()

# 2.5 Pipeline Execution with UX (순차적 렌더링 및 로딩 표시)
if "pending_prompt" in st.session_state:
    import time
    import concurrent.futures as _cf

    current_prompt = st.session_state.pending_prompt
    cfg          = retr.load_retriever_config()
    prompts_dict = prompt_store.load_prompts()

    _t_start = time.perf_counter()

    # ── LangSmith: 이번 검색 전체를 루트 span으로 묶기 ───────────
    try:
        from langsmith.run_helpers import get_current_run_tree as _grt
        import langsmith as _ls
        _ls_run_ctx = _ls.trace(
            name="home_search",
            run_type="chain",
            inputs={"question": current_prompt},
            metadata={
                "retriever_alias": cfg.get("alias", "unknown"),
                "retriever_units": " + ".join(
                    u.get("type", "?") for u in cfg.get("units", [])
                    if u.get("active") and u.get("type") not in (None, "", "미설정")
                ),
                "reranker": "ON" if cfg.get("reranker", {}).get("enabled") else "OFF",
                "llm_model": cfg.get("llm_model", "gpt-4o-mini"),
            },
        )
        _ls_run_ctx.__enter__()
    except Exception:
        _ls_run_ctx = None

    # ── Ask+Decompose & 투기적 검색 병렬 실행 ────────────────────
    with st.spinner("질문을 분석하는 중..."):
        def _do_ask():
            return rag_engine.ask_and_decompose(
                current_prompt,
                temperature=float(prompts_dict.get("ask_temp", 0.0)),
            )

        def _do_speculative_retrieve():
            return rag_engine.retrieve_candidates(
                current_prompt, current_prompt, cfg
            )

        with _cf.ThreadPoolExecutor(max_workers=2) as _ex:
            _ask_fut      = _ex.submit(_do_ask)
            _retrieve_fut = _ex.submit(_do_speculative_retrieve)
            ask_result    = _ask_fut.result()
            _spec_result  = _retrieve_fut.result()

    _t_ask = time.perf_counter()
    print(f"[시간] 질문분석 + 투기적검색 병렬: {_t_ask - _t_start:.2f}초")

    if ask_result["status"] == "ASK":
        follow_up = ask_result.get("question") or "조금 더 자세히 알려주실 수 있나요?"
        st.session_state.messages.append({"role": "assistant", "content": follow_up})
        st.session_state.ask_mode      = True
        st.session_state.base_question = current_prompt
        del st.session_state.pending_prompt
        st.rerun()

    # ── READY: 투기적 검색 결과 재사용 ───────────────────────────
    predecomposed = {
        "housing_query": ask_result.get("housing_query", ""),
        "finance_query": ask_result.get("finance_query", ""),
    }

    st.session_state.messages.append(
        {"role": "assistant", "content": "분석을 시작합니다. 결과를 아래에서 확인해 보세요."}
    )
    with chat_slot:
        render_chat_history()

    col_l_res, col_res, col_r_res = st.columns([1, 6, 1])

    with col_res:
        top3_placeholder   = st.empty()
        report_placeholder = st.empty()

        try:
            with top3_placeholder:
                with st.spinner("**💡 최적의 Top 3 정책을 선정하는 중...**"):
                    _t_top3_start = time.perf_counter()
                    stage1 = rag_engine.run_pipeline_top3(
                        current_prompt, cfg, prompts_dict,
                        predecomposed=predecomposed,
                        pretrieved=_spec_result,
                    )
                    top3         = stage1["top3"]
                    housing_docs = stage1["housing_docs"]
                    finance_docs = stage1["finance_docs"]
                    housing_q    = stage1["housing_query"]
                    finance_q    = stage1["finance_query"]
                    _t_top3_end  = time.perf_counter()
                    print(f"[시간] Top3 선정: {_t_top3_end - _t_top3_start:.2f}초")

            # ① Top3 관련 문서만 필터링해서 보고서 컨텍스트로 사용
            _top3_names = {p.get("policy_name", "").strip() for p in top3 if p.get("policy_name")}
            _filtered_contexts = [
                d.page_content for d in (housing_docs + finance_docs)
                if any(
                    name in (d.metadata.get("policy_name") or d.metadata.get("title") or d.page_content[:80])
                    for name in _top3_names
                )
            ] or stage1["contexts_text"]  # 매칭 없으면 전체 fallback
            print(f"[컨텍스트] 전체 {len(stage1['contexts_text'])}개 → Top3 필터링 후 {len(_filtered_contexts)}개")

            # ② Top3 확정 즉시: 보고서 스트리밍 + 상세정보 프리패치 동시 시작
            import queue as _queue
            import threading as _threading

            _report_queue   = _queue.Queue()
            _details_result = {}
            _t_report_start = time.perf_counter()

            def _stream_to_queue():
                try:
                    for _chunk in rag_engine.stream_report(
                        current_prompt, top3, _filtered_contexts, prompts_dict
                    ):
                        _report_queue.put(_chunk)
                except Exception as _e:
                    _report_queue.put(_e)
                finally:
                    _report_queue.put(None)

            def _prefetch_details():
                result = rag_engine.prefetch_policy_details(
                    current_prompt, top3, _filtered_contexts
                )
                _details_result.update(result)
                print(f"[프리패치] 상세정보 {len(result)}개 완료")

            _report_thread  = _threading.Thread(target=_stream_to_queue,    daemon=True)
            _prefetch_thread = _threading.Thread(target=_prefetch_details,  daemon=True)
            _report_thread.start()
            _prefetch_thread.start()

            # Top3 카드 렌더링 (보고서 스트리밍과 동시 진행)
            with top3_placeholder.container():
                st.markdown(
                    "<div style='font-size:1.6rem; font-weight:700;"
                    " background:linear-gradient(90deg,#FB7185,#FDBA74);"
                    " -webkit-background-clip:text; -webkit-text-fill-color:transparent;"
                    " margin-top:1rem; margin-bottom:2rem; text-align:center;'>"
                    "🏆 맞춤 추천 정책 TOP3</div>",
                    unsafe_allow_html=True,
                )
                grid_html = '<div style="display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; align-items:stretch;">'
                for i, item in enumerate(top3[:3]):
                    grid_html += _card_html(item, i)
                grid_html += '</div>'
                st.markdown(grid_html.replace('\n', ''), unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

            # 버퍼링된 보고서 청크 소비
            with report_placeholder:
                _stream_status = st.empty()
                _full_report   = ""
                while True:
                    _chunk = _report_queue.get()
                    if _chunk is None:
                        break
                    if isinstance(_chunk, Exception):
                        raise _chunk
                    _full_report += _chunk
                    _stream_status.markdown(
                        f"**📝 보고서 작성 중...** `{len(_full_report)}자`",
                        unsafe_allow_html=True,
                    )
                report = _full_report
                _t_report_end = time.perf_counter()
                print(f"[시간] 종합보고서 스트리밍: {_t_report_end - _t_report_start:.2f}초")
                print(f"[시간] 전체 소요시간 (질문→보고서 완료): {_t_report_end - _t_start:.2f}초")

            with report_placeholder.container():
                st.markdown("<br>", unsafe_allow_html=True)
                try:
                    json_str    = extract_json(report)
                    report_data = json.loads(json_str, strict=False)
                    render_report(report_data)
                except Exception:
                    st.warning("보고서 형식 변환 실패, 텍스트로 표시합니다.")
                    st.markdown(
                        '<div class="report-hero-title">✨ 맞춤 정책 종합 보고서</div>',
                        unsafe_allow_html=True,
                    )
                    with st.container(border=True):
                        st.markdown(report, unsafe_allow_html=True)

            res = {
                "original_question": current_prompt,
                "housing_query":  housing_q,
                "finance_query":  finance_q,
                "housing_docs":   [{"content": d.page_content, "metadata": d.metadata} for d in housing_docs],
                "finance_docs":   [{"content": d.page_content, "metadata": d.metadata} for d in finance_docs],
                "top3":           top3,
                "report":         report,
                "contexts_text":  stage1["contexts_text"],
            }

            st.session_state.last_res = res
            st.session_state.disqualified_names = []
            st.session_state.detail_mode = False
            # 프리패치가 이미 완료됐으면 바로 사용, 아직 진행 중이면 빈 dict (클릭 시 on-demand)
            st.session_state.policy_details_cache = dict(_details_result)
            print(f"[프리패치] 저장 시점 캐시: {list(_details_result.keys())}")

            # ── follow-up 안내 멘트 ───────────────────────────────────
            follow_up_msg = (
                "자격요건을 추가로 알려주시면 해당 안 되는 정책을 표시해드릴게요!\n\n"
                "아래 항목이 있으면 입력해주세요:\n"
                "• 무주택 여부 및 기간 (예: 무주택 3년)\n"
                "• 정확한 연소득 (예: 연봉 3,500만원)\n"
                "• 혼인신고 여부/기간 (예: 결혼 2년차)\n"
                "• 현재 보유 자산 (예: 자산 8천만원)"
            )
            st.session_state.messages.append({"role": "assistant", "content": follow_up_msg})

            del st.session_state.pending_prompt
            # LangSmith 루트 span 닫기
            try:
                if _ls_run_ctx:
                    _ls_run_ctx.__exit__(None, None, None)
            except Exception:
                pass
            st.rerun()

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
            # LangSmith 루트 span 닫기 (에러 시)
            try:
                if _ls_run_ctx:
                    _ls_run_ctx.__exit__(type(e), e, None)
            except Exception:
                pass
            if "pending_prompt" in st.session_state:
                del st.session_state.pending_prompt

# 3. Results Section
elif "last_res" in st.session_state:
    res              = st.session_state.last_res
    disqualified_set = set(st.session_state.get("disqualified_names", []))
    detail_mode      = st.session_state.get("detail_mode", False)
    detail_policies  = st.session_state.get("detail_policies", [])
    details_cache    = st.session_state.get("policy_details_cache", {})

    col_l_res, col_res, col_r_res = st.columns([1, 6, 1])

    with col_res:
        # ── TOP3 카드 (항상 표시, 자격 안 되면 X 오버레이) ──────────
        st.markdown(
            "<div style='font-size:1.6rem; font-weight:700;"
            " background:linear-gradient(90deg,#FB7185,#FDBA74);"
            " -webkit-background-clip:text; -webkit-text-fill-color:transparent;"
            " margin-top:1rem; margin-bottom:2rem; text-align:center;'>"
            "🏆 맞춤 추천 정책 TOP3</div>",
            unsafe_allow_html=True,
        )

        # ── 상세보기 버튼 (Top3 카드 위) ──────────────────────────────
        _btn_cols = st.columns(3)
        for _bi, _item in enumerate(res["top3"][:3]):
            _pname = _item.get("policy_name", f"정책 {_bi+1}")
            with _btn_cols[_bi]:
                if st.button(f"📋 상세보기", key=f"detail_btn_{_bi}", use_container_width=True):
                    cache = st.session_state.get("policy_details_cache", {})
                    if _pname not in cache:
                        with st.spinner(f"'{_pname}' 상세 정보 불러오는 중..."):
                            orig_q = res.get("original_question") or res.get("housing_query") or ""
                            cache[_pname] = rag_engine.get_policy_detail(orig_q, _pname, res.get("contexts_text", []))
                        st.session_state.policy_details_cache = cache
                    st.session_state.detail_mode     = True
                    st.session_state.detail_policies = [_pname]
                    st.rerun()

        grid_html = '<div style="display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; align-items:stretch;">'
        for i, item in enumerate(res["top3"][:3]):
            is_disq = item.get("policy_name", "") in disqualified_set
            grid_html += _card_html(item, i, disqualified=is_disq)
        grid_html += '</div>'
        st.markdown(grid_html.replace('\n', ''), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 상세 모드: 보고서 대신 상세 카드 가로 배치 ─────────────
        if detail_mode and detail_policies:
            n_cols = len(detail_policies)
            cols_html = f'<div style="display:grid;grid-template-columns:repeat({n_cols},1fr);gap:1rem;align-items:stretch;">'
            for name in detail_policies:
                detail = details_cache.get(name, {"policy_name": name, "overview": "정보를 불러오는 중입니다."})
                cols_html += _detail_card_html(detail)
            cols_html += '</div>'
            st.markdown(
                "<div style='font-size:1.3rem;font-weight:700;color:#1E293B;"
                "margin-bottom:1.2rem;'>📋 정책 상세 정보</div>",
                unsafe_allow_html=True,
            )
            st.markdown(cols_html.replace('\n', ''), unsafe_allow_html=True)

        else:
            # ── 종합 보고서 ────────────────────────────────────────
            try:
                json_str    = extract_json(res["report"])
                report_data = json.loads(json_str)
                render_report(report_data)
            except Exception:
                st.warning("보고서 형식 변환 실패, 텍스트로 표시합니다.")
                st.markdown(
                    '<div class="report-hero-title">✨ 맞춤 정책 종합 보고서</div>',
                    unsafe_allow_html=True,
                )
                with st.container(border=True):
                    st.markdown(res["report"], unsafe_allow_html=True)




