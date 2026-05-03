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

# UI Injection (this brings in the global 80% width constraint for the header)
inject_ui()


def _policy_url(item: dict) -> str:
    """정책 URL 반환. 없으면 카테고리별 기본 URL."""
    url = (item.get("url") or "").strip()
    if url:
        return url
    if "금융" in (item.get("category") or ""):
        return "https://nhuf.molit.go.kr/"
    return "https://housing.seoul.go.kr/"


def _card_html(item: dict, i: int) -> str:
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
    return f'''<div class="policy-card {cat_class}">
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
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.pending_prompt = prompt
    st.rerun()

# 2.5 Pipeline Execution with UX (순차적 렌더링 및 로딩 표시)
if "pending_prompt" in st.session_state:
    current_prompt = st.session_state.pending_prompt

    cfg          = retr.load_retriever_config()
    prompts_dict = prompt_store.load_prompts()

    # ── Ask 단계: 정보 부족 시 되묻기 ────────────────────────────
    with st.spinner("질문을 분석하는 중..."):
        ask_result = rag_engine.ask_or_ready(
            current_prompt,
            ask_prompt=prompts_dict.get("ask"),
            temperature=float(prompts_dict.get("ask_temp", 0.0)),
        )

    if ask_result["status"] == "ASK":
        follow_up = ask_result.get("question") or "조금 더 자세히 알려주실 수 있나요?"
        st.session_state.messages.append({"role": "assistant", "content": follow_up})
        st.session_state.ask_mode      = True
        st.session_state.base_question = current_prompt
        del st.session_state.pending_prompt
        st.rerun()

    # ── READY: 파이프라인 진행 ────────────────────────────────────
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
                    stage1       = rag_engine.run_pipeline_top3(current_prompt, cfg, prompts_dict)
                    top3         = stage1["top3"]
                    housing_docs = stage1["housing_docs"]
                    finance_docs = stage1["finance_docs"]
                    housing_q    = stage1["housing_query"]
                    finance_q    = stage1["finance_query"]

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

            # ── 보고서 스트리밍 ──────────────────────────────────────
            with report_placeholder:
                _stream_status = st.empty()
                _full_report   = ""
                for _chunk in rag_engine.stream_report(
                    current_prompt, top3, stage1["contexts_text"], prompts_dict
                ):
                    _full_report += _chunk
                    _stream_status.markdown(
                        f"**📝 보고서 작성 중...** `{len(_full_report)}자`",
                        unsafe_allow_html=True,
                    )
                report = _full_report

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
                "housing_query":  housing_q,
                "finance_query":  finance_q,
                "housing_docs":   [{"content": d.page_content, "metadata": d.metadata} for d in housing_docs],
                "finance_docs":   [{"content": d.page_content, "metadata": d.metadata} for d in finance_docs],
                "top3":           top3,
                "report":         report,
                "contexts_text":  stage1["contexts_text"],
            }

            st.session_state.last_res = res
            del st.session_state.pending_prompt
            st.rerun()

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
            if "pending_prompt" in st.session_state:
                del st.session_state.pending_prompt

# 3. Results Section
elif "last_res" in st.session_state:
    res = st.session_state.last_res

    col_l_res, col_res, col_r_res = st.columns([1, 6, 1])

    with col_res:
        st.markdown(
            "<div style='font-size:1.6rem; font-weight:700;"
            " background:linear-gradient(90deg,#FB7185,#FDBA74);"
            " -webkit-background-clip:text; -webkit-text-fill-color:transparent;"
            " margin-top:1rem; margin-bottom:2rem; text-align:center;'>"
            "🏆 맞춤 추천 정책 TOP3</div>",
            unsafe_allow_html=True,
        )

        grid_html = '<div style="display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; align-items:stretch;">'
        for i, item in enumerate(res["top3"][:3]):
            grid_html += _card_html(item, i)
        grid_html += '</div>'
        st.markdown(grid_html.replace('\n', ''), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

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
