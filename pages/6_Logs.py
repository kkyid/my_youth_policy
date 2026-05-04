import streamlit as st
st.set_page_config(page_title="Logs", layout="wide", initial_sidebar_state="collapsed")

import json
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ui import inject_ui, render_page_title
from core import logger as exp_logger

inject_ui()
render_page_title("Logs")

st.markdown("""
<style>
.log-table { width:100%; border-collapse:collapse; font-size:0.95rem; }
.log-table th {
    background:#F1F5F9; color:#475569; font-weight:700;
    padding:10px 14px; text-align:left; border-bottom:2px solid #CBD5E1;
    white-space:nowrap;
}
.log-table td {
    padding:12px 14px; border-bottom:1px solid #E2E8F0;
    vertical-align:top; color:#1E293B; line-height:1.6;
}
.log-table tr:hover td { background:#F8FAFC; }
.log-table .col-q  { width:35%; min-width:200px; overflow-wrap:break-word; }
.log-table .col-gt { width:35%; min-width:200px; overflow-wrap:break-word; color:#475569; font-size:0.85rem; }
.log-table .col-tag { width:100px; min-width:100px; color:#64748B; font-size:0.8rem; }
.log-table .col-sc { width:62px; max-width:62px; text-align:center; font-size:0.935rem; font-weight:600; font-variant-numeric:tabular-nums; }
.log-table th.col-sc { text-align:center; }
.log-table .col-rt { width:140px; min-width:140px; }
.log-table .col-dt { width:100px; min-width:100px; color:#94A3B8; font-size:0.8rem; white-space:nowrap; }
.log-table .retr-badge {
    display:inline-block; background:#F1F5F9; border:1px solid #CBD5E1;
    border-radius:5px; padding:2px 8px; margin-bottom:3px;
    font-size:0.78rem; color:#475569; white-space:nowrap;
}
.log-table .rerank-on {
    display:inline-block; background:#EEF2FF; border:1px solid #C7D2FE;
    border-radius:5px; padding:2px 8px; margin-top:4px;
    font-size:0.78rem; color:#4F46E5; font-weight:600; white-space:nowrap;
}
.log-table .rerank-off {
    display:inline-block; background:#F8FAFC; border:1px solid #E2E8F0;
    border-radius:5px; padding:2px 8px; margin-top:4px;
    font-size:0.78rem; color:#94A3B8; white-space:nowrap;
}
.log-table .extract-on {
    display:inline-block; background:#FFF7ED; border:1px solid #FED7AA;
    border-radius:5px; padding:2px 8px; margin-top:4px;
    font-size:0.78rem; color:#C2410C; font-weight:600; white-space:nowrap;
}
.log-table .extract-off {
    display:inline-block; background:#F8FAFC; border:1px solid #E2E8F0;
    border-radius:5px; padding:2px 8px; margin-top:4px;
    font-size:0.78rem; color:#94A3B8; white-space:nowrap;
}
.log-table .preproc-badge {
    display:inline-block; background:#ECFDF5; border:1px solid #A7F3D0;
    border-radius:5px; padding:2px 8px; margin-bottom:3px;
    font-size:0.78rem; color:#065F46; white-space:nowrap;
}
/* 1. Logs 페이지 전체 컨테이너 너비를 80%로 조정 (ui.py 기본값과 일치시켜 헤더 정렬 유지) */
div.stMainBlockContainer {
    max-width: 70% !important;
    width: 70% !important;
}

/* 2. 본문 영역을 화면 전체의 70%가 되도록 조정 (컨테이너 80% * 본문 87.5% = 70%) */
[data-testid="stVerticalBlock"] > div:nth-child(n+3) {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 auto !important; /* 중앙 정렬 유지 */
}

/* 3. 테이블 레이아웃 최적화: 가로로 충분히 늘어나도록 설정 */
.log-table {
    width: 100% !important;
    table-layout: auto !important; /* auto로 변경하여 긴 내용이 있는 컬럼이 자연스럽게 넓어지도록 함 */
    border-collapse: collapse;
}

/* 4. 멀티셀렉트 위젯 자체의 너비 강제 조정 */
div[data-testid="stMultiSelect"] {
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)


@st.dialog("📊 RAGAS 지표 개선 방법", width="large")
def show_improvement_guide():
    st.markdown("""
### 1. Faithfulness (충실도)가 낮다면?
> **"모델이 문서에 없는 말을 지어냄 (환각 현상)"**

**해결 방안:**
- **System Prompt 수정** — "반드시 주어진 문서 안에서만 답변해라"라고 강력하게 제약.
- **Temperature 낮추기** — LLM의 창의성을 줄여서 사실 위주로만 답변하도록 설정 (예: 0.0 ~ 0.2).

---

### 2. Answer Relevance (답변 관련성)가 낮다면?
> **"대답은 하는데 질문의 의도에서 벗어남"**

**해결 방안:**
- **Few-shot Prompting** — 원하는 답변의 예시(질문-답변 세트)를 프롬프트에 넣어줌.
- **출력 형식 고정** — JSON이나 마크다운 등 특정 형식을 강제하여 명확성을 높임.

---

### 3. Context Precision (맥락 정밀도)이 낮다면?
> **"검색 결과에 쓰레기 정보가 너무 많음"**

**해결 방안:**
- **Reranker 도입** — 임베딩 검색 결과를 다시 한번 순위를 매겨서 정밀도를 높임.
- **쿼리 변환** — 질문을 검색에 더 적합한 형태로 LLM을 통해 재작성.

---

### 4. Context Recall (맥락 재현율)이 낮다면?
> **"답을 내기 위해 꼭 필요한 정보가 검색에서 빠짐"**

**해결 방안:**
- **Chunk Size 조정** — 문서를 너무 작게 잘랐다면 문맥이 끊길 수 있으니 크기를 키움.
- **Hybrid Search** — 벡터 검색(Semantic)뿐만 아니라 키워드 검색(BM25)을 섞어서 사용.
""")


# ── 공통 헬퍼 ─────────────────────────────────────────────────────
def fmt_retrievers_html(cfg) -> str:
    if not cfg:
        return "<span style='color:#94A3B8'>—</span>"
    html = ""
    pre = cfg.get("preprocessing", {})
    if pre:
        sq = pre.get("self_query", {})
        qt = pre.get("query_transform", {})
        sq_label = "Self Query●ON" if sq.get("enabled") else "Self Query●OFF"
        qt_method = qt.get("method", "없음")
        html += f'<div><span class="preproc-badge">{sq_label} · {qt_method}</span></div>'
    units = cfg.get("units", [])
    active = [u for u in units if u.get("active") and u.get("type") not in (None, "", "미설정")]
    if active:
        raw_weights = [float(u.get("weight", 1.0)) for u in active]
        total_w = sum(raw_weights) or 1.0
        norm_weights = [w / total_w for w in raw_weights]
        for u, nw in zip(active, norm_weights):
            html += (f'<div><span class="retr-badge">'
                     f'{u.get("type","?")} k={u.get("k","?")} {u.get("search_type","")} w={nw:.0%}'
                     f'</span></div>')
    elif not html:
        t = cfg.get("type", "")
        k = cfg.get("k", "")
        html += f'<span class="retr-badge">{t} k={k}</span>' if t else "<span style='color:#94A3B8'>—</span>"
    rerank = cfg.get("reranker", {})
    if rerank.get("enabled"):
        html += f'<div><span class="rerank-on">Reranker●ON top{rerank.get("final_k","?")}</span></div>'
    else:
        html += '<div><span class="rerank-off">Reranker○OFF</span></div>'
    extractor = cfg.get("extractor", {})
    if extractor.get("enabled"):
        ext_model = extractor.get("model", "gpt-4o-mini")
        html += f'<div><span class="extract-on">LLM-Extract●ON ({ext_model})</span></div>'
    else:
        html += '<div><span class="extract-off">LLM-Extract○OFF</span></div>'
    return html


def _score(val) -> str:
    try:
        return f"{float(val):.3f}"
    except Exception:
        return "—"


def _retriever_label(cfg) -> str:
    """리트리버 설정 → 짧은 텍스트 레이블"""
    if not cfg:
        return "—"
    units = cfg.get("units", [])
    active = [u for u in units if u.get("active") and u.get("type") not in (None, "", "미설정")]
    parts = [u.get("type", "?") for u in active]
    label = "+".join(parts) if parts else "—"
    pre = cfg.get("preprocessing", {})
    qt_method = pre.get("query_transform", {}).get("method", "없음")
    sq_on = pre.get("self_query", {}).get("enabled", False)
    if qt_method != "없음":
        label += f" / {qt_method}"
    if sq_on:
        label += " / SQ"
    if cfg.get("reranker", {}).get("enabled"):
        label += " / Rerank"
    if cfg.get("extractor", {}).get("enabled"):
        label += " / Extract"
    return label


# ── 로그 로드 ─────────────────────────────────────────────────────
records = exp_logger.load_logs()
records.sort(key=lambda r: r.get("ts", ""), reverse=True)

# ── 상단 액션 바 (개선방법:왼쪽, 건수:중앙, 삭제:오른쪽)
@st.dialog("⚠️ 로그 전체 삭제 확인")
def confirm_clear_logs_dialog():
    st.warning("정말로 모든 실험 로그를 삭제하시겠습니까?\n이 작업은 되돌릴 수 없으며, 모든 과거 기록이 영구 삭제됩니다.")
    c1, c2 = st.columns(2)
    if c1.button("🔥 네, 전체 삭제", type="primary", use_container_width=True):
        exp_logger.clear_logs()
        st.success("모든 로그가 삭제되었습니다.")
        st.rerun()
    if c2.button("취소", use_container_width=True):
        st.rerun()

c1, c2, c3 = st.columns([1.6, 5, 1.6])
with c1:
    if st.button("📊 개선방법", use_container_width=True):
        show_improvement_guide()

with c2:
    st.markdown(
        f'<p style="margin-top:10px;color:#64748B;font-size:0.85rem;text-align:center;">'
        f'총 <b>{len(records):,}</b>건의 실험 기록</p>',
        unsafe_allow_html=True,
    )

with c3:
    if st.button("🗑️ 전체 삭제", use_container_width=True):
        confirm_clear_logs_dialog()

if not records:
    st.info("아직 실험 로그가 없습니다. Home 또는 Evaluation 페이지에서 RAG 를 실행해 보세요.")
    st.stop()


# ════════════════════════════════════════════════════════════════
# 탭
# ════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["📋 개별 실험", "📊 태그별 비교"])


# ════════════════════════════════════════════════════════════════
# 탭 1 — 개별 실험 (기존)
# ════════════════════════════════════════════════════════════════
with tab1:
    PAGE_SIZE = 3
    if "logs_page" not in st.session_state:
        st.session_state.logs_page = 1

    total_pages = max(1, (len(records) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(1, min(st.session_state.logs_page, total_pages))
    start = (page - 1) * PAGE_SIZE
    slice_records = records[start:start + PAGE_SIZE]

    header = (
        "<thead><tr>"
        "<th class='col-tag'>태그</th>"
        "<th class='col-q'>질문</th>"
        "<th class='col-gt'>테스트셋 답변</th>"
        "<th class='col-sc'>Faithful</th>"
        "<th class='col-sc'>Ans Rel</th>"
        "<th class='col-sc'>Ctx Prec</th>"
        "<th class='col-sc'>Ctx Rec</th>"
        "<th class='col-rt'>리트리버</th>"
        "<th class='col-dt'>날짜</th>"
        "</tr></thead>"
    )
    rows_html = ""
    for r in slice_records:
        m   = r.get("metrics") or {}
        q   = (r.get("question") or "").replace("<", "&lt;")
        gt  = (r.get("ground_truth") or "—").replace("<", "&lt;")
        dt  = r.get("ts", "")[5:16].replace("T", " ") # 월-일 시:분 만 표시하여 압축
        tag = (r.get("tag") or "—").replace("<", "&lt;")
        rows_html += (
            "<tr>"
            f"<td class='col-tag'>{tag}</td>"
            f"<td class='col-q'>{q}</td>"
            f"<td class='col-gt'>{gt}</td>"
            f"<td class='col-sc'>{_score(m.get('faithfulness'))}</td>"
            f"<td class='col-sc'>{_score(m.get('answer_relevance'))}</td>"
            f"<td class='col-sc'>{_score(m.get('context_precision'))}</td>"
            f"<td class='col-sc'>{_score(m.get('context_recall'))}</td>"
            f"<td class='col-rt'>{fmt_retrievers_html(r.get('retriever'))}</td>"
            f"<td class='col-dt'>{dt}</td>"
            "</tr>"
        )
    st.markdown(
        f"<table class='log-table'>{header}<tbody>{rows_html}</tbody></table>",
        unsafe_allow_html=True,
    )

    nav_l, nav_m, nav_r = st.columns([1, 2, 1])
    with nav_l:
        if st.button("◀ 이전", disabled=(page <= 1), use_container_width=True):
            st.session_state.logs_page = page - 1
            st.rerun()
    with nav_m:
        st.markdown(
            f"<div style='text-align:center;padding-top:8px;color:#64748B;font-size:0.85rem;'>"
            f"{page} / {total_pages} 페이지 &nbsp;·&nbsp; 3건씩</div>",
            unsafe_allow_html=True,
        )
    with nav_r:
        if st.button("다음 ▶", disabled=(page >= total_pages), use_container_width=True):
            st.session_state.logs_page = page + 1
            st.rerun()

    st.markdown("---")
    st.markdown("#### 질문 별 LLM이 가져온 Top3")

    def _label(rid):
        r = next((x for x in slice_records if x.get("id") == rid), None)
        return (r.get("question") or "")[:80] if r else rid[:8]

    selected_id = st.selectbox(
        "질문 선택",
        options=[r.get("id") for r in slice_records],
        format_func=_label,
        index=None,
        placeholder="질문을 선택하면 Top 3 문서가 표시됩니다",
    )
    if selected_id:
        selected = next((r for r in slice_records if r.get("id") == selected_id), None)
        if selected:
            st.markdown("**Top 3 문서**")
            st.json(selected.get("top3", []))


# ════════════════════════════════════════════════════════════════
# 탭 2 — 태그별 비교
# ════════════════════════════════════════════════════════════════
with tab2:
    METRIC_COLS = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]
    METRIC_LABELS = {
        "faithfulness":      "Faithfulness",
        "answer_relevance":  "Ans Relevance",
        "context_precision": "Ctx Precision",
        "context_recall":    "Ctx Recall",
    }

    eval_records = [
        r for r in records
        if r.get("metrics") and any(
            r["metrics"].get(k) is not None for k in METRIC_COLS
        )
    ]

    if not eval_records:
        st.info("평가 점수가 기록된 실험이 없습니다. Evaluation 페이지에서 평가를 실행하세요.")
    else:
        all_tags = sorted(set(r.get("tag") or "태그없음" for r in eval_records), reverse=True)

        st.caption("Evaluation 페이지에서 실험 태그를 붙여 실행한 배치들을 여기서 비교합니다.")

        selected_tags = st.multiselect(
            "비교할 태그 선택",
            options=all_tags,
            default=[all_tags[0]] if all_tags else [],
            key="cmp_tags",
        )

        if not selected_tags:
            st.warning("비교할 태그를 1개 이상 선택하세요.")
        else:
            from collections import defaultdict
            tag_records_map = defaultdict(list)
            for r in eval_records:
                tag = r.get("tag") or "태그없음"
                if tag in selected_tags:
                    tag_records_map[tag].append(r)

            # 태그별 집계
            tag_records_map = defaultdict(list)
            for r in eval_records:
                tag = r.get("tag") or "태그없음"
                if tag in selected_tags:
                    tag_records_map[tag].append(r)

            cmp_header = (
                "<thead><tr>"
                "<th class='col-tag'>태그</th>"
                "<th class='col-dt'>실험 수</th>"
                "<th class='col-sc'>Faithful</th>"
                "<th class='col-sc'>Ans Rel</th>"
                "<th class='col-sc'>Ctx Prec</th>"
                "<th class='col-sc'>Ctx Rec</th>"
                "<th class='col-id'>종합</th>"
                "<th class='col-rt'>리트리버</th>"
                
                "</tr></thead>"
            )
            
            cmp_rows_html = ""
            for tag in selected_tags:
                grp = tag_records_map[tag]
                if not grp:
                    continue
                
                # 평균 메트릭 계산
                metric_avgs = {}
                for k in METRIC_COLS:
                    vals = [r["metrics"][k] for r in grp if r.get("metrics") and r["metrics"].get(k) is not None]
                    metric_avgs[k] = sum(vals) / len(vals) if vals else 0.0
                    
                m_vals = list(metric_avgs.values())
                total_score = sum(m_vals) / len(m_vals) if m_vals else 0.0
                
                cmp_rows_html += (
                    "<tr>"
                    f"<td class='col-tag'><b>{tag}</b></td>"
                    f"<td class='col-dt'>{len(grp)}건</td>"
                    f"<td class='col-sc'>{_score(metric_avgs['faithfulness'])}</td>"
                    f"<td class='col-sc'>{_score(metric_avgs['answer_relevance'])}</td>"
                    f"<td class='col-sc'>{_score(metric_avgs['context_precision'])}</td>"
                    f"<td class='col-sc'>{_score(metric_avgs['context_recall'])}</td>"
                    f"<td class='col-id' style='color:#4F46E5; font-weight:bold;'>{_score(total_score)}</td>"
                    f"<td class='col-rt'>{fmt_retrievers_html(grp[0].get('retriever'))}</td>"
                    "</tr>"
                )

            if not cmp_rows_html:
                st.warning("선택한 태그에 해당하는 데이터가 없습니다.")
            else:
                st.markdown("#### 설정별 평균 메트릭")
                st.markdown(
                    f"<table class='log-table'>{cmp_header}<tbody>{cmp_rows_html}</tbody></table>",
                    unsafe_allow_html=True,
                )

