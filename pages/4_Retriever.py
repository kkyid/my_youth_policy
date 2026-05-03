import streamlit as st
import sys
from pathlib import Path

st.set_page_config(page_title="Retriever", layout="wide", initial_sidebar_state="collapsed")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ui import inject_ui, render_page_title
from core import retrievers as retr
from core.rag_engine import (
    DEFAULT_SELF_QUERY_PROMPT,
    DEFAULT_HYDE_PROMPT,
    DEFAULT_MULTI_QUERY_PROMPT,
    DEFAULT_DECOMPOSITION_PROMPT,
)

inject_ui()
render_page_title("Retriever")

# ── 설정 로드 ─────────────────────────────────────────────────────
cfg        = retr.load_retriever_config()
units_cfg  = cfg.get("units",         retr.DEFAULT_CONFIG["units"])
rerank_cfg = cfg.get("reranker",      retr.DEFAULT_CONFIG["reranker"])
preproc_cfg = cfg.get("preprocessing", retr.DEFAULT_CONFIG["preprocessing"])

# ── 선택지 ────────────────────────────────────────────────────────
UNIT_TYPES = [
    "미설정",
    "VectorStore",
    "BM25",
    "Multi-Query Retriever",
    "Parent Document Retriever",
    "Self-Querying Retriever",
]
SEARCH_TYPES = ["similarity", "mmr", "similarity_score_threshold"]

RERANKER_MODELS = {
    "한국어 — bongsoo/kpf-cross-encoder-v1": "bongsoo/kpf-cross-encoder-v1",
    "영어   — ms-marco-MiniLM-L-6-v2":    "cross-encoder/ms-marco-MiniLM-L-6-v2",
}

QT_METHODS = ["없음", "HyDE", "Multi-Query", "Decomposition"]

QT_DESCRIPTIONS = {
    "없음":         "원본 질문 그대로 사용합니다.",
    "HyDE":         "LLM이 가상 정책 문서를 생성하고, 그 텍스트로 벡터 검색합니다. 질문이 추상적일 때 유용합니다.",
    "Multi-Query":  "LLM이 질문을 3가지 다른 표현으로 변환해 각각 검색 후 병합합니다. 재현율(Recall)을 높입니다.",
    "Decomposition":"복합 질문을 2~4개 하위 질문으로 분해해 각각 검색 후 병합합니다. 복잡한 조건 질문에 유용합니다.",
}

QT_DEFAULT_PROMPTS = {
    "HyDE":         DEFAULT_HYDE_PROMPT,
    "Multi-Query":  DEFAULT_MULTI_QUERY_PROMPT,
    "Decomposition": DEFAULT_DECOMPOSITION_PROMPT,
}

# ════════════════════════════════════════════════════════════════
# 섹션 0: 전처리 (Pre-processing)
# ════════════════════════════════════════════════════════════════
st.markdown("### 전처리 (Pre-processing)")
st.caption("리트리버에 넘기기 전, 질문을 변환하거나 메타데이터 필터를 추출합니다.")

sq_cfg  = preproc_cfg.get("self_query",     retr.DEFAULT_CONFIG["preprocessing"]["self_query"])
qt_cfg  = preproc_cfg.get("query_transform", retr.DEFAULT_CONFIG["preprocessing"]["query_transform"])

with st.container(border=True):
    pc1, pc2 = st.columns([1, 2])

    # ── Self-Query 토글 ──────────────────────────────────────────
    with pc1:
        st.markdown("**Self-Query 필터**")
        sq_enabled = st.toggle(
            "사용",
            value=bool(sq_cfg.get("enabled", False)),
            key="sq_enabled",
            help="사용자 질문에서 연령/소득/가구형태 등 메타데이터 조건을 추출해 Chroma 필터로 적용합니다.",
        )
        if sq_enabled:
            with st.expander("Self-Query 프롬프트 편집", expanded=False):
                sq_prompt = st.text_area(
                    "프롬프트",
                    value=sq_cfg.get("prompt") or DEFAULT_SELF_QUERY_PROMPT,
                    height=300,
                    key="sq_prompt",
                    label_visibility="collapsed",
                )
        else:
            sq_prompt = sq_cfg.get("prompt") or ""

    # ── Query Transform ──────────────────────────────────────────
    with pc2:
        st.markdown("**Query Transform**")
        saved_method = qt_cfg.get("method", "없음")
        if saved_method not in QT_METHODS:
            saved_method = "없음"

        qt_method = st.radio(
            "변환 방식",
            options=QT_METHODS,
            index=QT_METHODS.index(saved_method),
            key="qt_method",
            horizontal=True,
            label_visibility="collapsed",
        )

        st.caption(QT_DESCRIPTIONS[qt_method])

        saved_prompts = qt_cfg.get("prompts", {})

        # 현재 선택된 방식의 프롬프트 편집
        if qt_method != "없음":
            with st.expander(f"{qt_method} 프롬프트 편집", expanded=False):
                qt_prompt_val = st.text_area(
                    "프롬프트",
                    value=saved_prompts.get(qt_method) or QT_DEFAULT_PROMPTS[qt_method],
                    height=250,
                    key=f"qt_prompt_{qt_method}",
                    label_visibility="collapsed",
                )
            # 다른 방식들 프롬프트는 저장된 값 유지
            qt_prompts = dict(saved_prompts)
            qt_prompts[qt_method] = qt_prompt_val
        else:
            qt_prompts = dict(saved_prompts)

new_preproc = {
    "self_query": {
        "enabled": sq_enabled,
        "prompt": sq_prompt if sq_enabled else "",
    },
    "query_transform": {
        "method": qt_method,
        "prompts": qt_prompts,
    },
}

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# 섹션 1: 리트리버 조합
# ════════════════════════════════════════════════════════════════

def _cur_type(i: int) -> str:
    key = f"u_type_{i}"
    if key in st.session_state:
        return st.session_state[key]
    return units_cfg[i].get("type", "미설정") if i < len(units_cfg) else "미설정"

n_active = sum(1 for i in range(3) if _cur_type(i) != "미설정")

st.markdown("### 리트리버 조합")
if n_active > 1:
    st.caption("활성화된 리트리버가 여러 개면 EnsembleRetriever로 자동 합산됩니다. 가중치 합은 자동 정규화돼요.")
else:
    st.caption("리트리버를 최대 3개까지 조합할 수 있어요.")

cols     = st.columns(3)
new_units = []

for i in range(3):
    u_cfg = (
        units_cfg[i] if i < len(units_cfg)
        else {"type": "미설정", "k": 5, "search_type": "similarity",
              "active": False, "weight": 1.0, "lambda_mult": 0.5, "score_threshold": 0.5}
    )

    with cols[i]:
        st.markdown(f"**리트리버 {i + 1}**")
        with st.container(border=True):

            u_type = st.selectbox(
                "종류",
                options=UNIT_TYPES,
                index=UNIT_TYPES.index(u_cfg.get("type", "미설정"))
                      if u_cfg.get("type", "미설정") in UNIT_TYPES else 0,
                key=f"u_type_{i}",
            )
            is_active = u_type != "미설정"
            is_bm25   = u_type == "BM25"

            u_k = st.slider(
                "검색 후보 수 (k)",
                1, 20,
                int(u_cfg.get("k", 5)) if is_active else 5,
                disabled=not is_active,
                key=f"u_k_{i}",
                help="리랭커 사용 시 이 수만큼 후보를 먼저 뽑고, 최종 Top-K로 압축해요.",
            )

            if is_bm25:
                u_search = "similarity"
                st.selectbox(
                    "검색 방식", options=["해당 없음 (BM25)"],
                    disabled=True, key=f"u_search_{i}",
                    help="BM25는 키워드 기반이라 검색 방식 설정이 필요 없어요.",
                )
            else:
                saved_stype = u_cfg.get("search_type", "similarity")
                u_search = st.selectbox(
                    "검색 방식",
                    options=SEARCH_TYPES if is_active else ["미설정"],
                    index=SEARCH_TYPES.index(saved_stype)
                          if is_active and saved_stype in SEARCH_TYPES else 0,
                    disabled=not is_active,
                    key=f"u_search_{i}",
                )

            u_lambda = float(u_cfg.get("lambda_mult", 0.5))
            if is_active and not is_bm25 and u_search == "mmr":
                u_lambda = st.slider(
                    "다양성 (lambda_mult)",
                    0.0, 1.0, u_lambda, 0.05,
                    key=f"u_lambda_{i}",
                    help="0 → 결과 다양성 최대  /  1 → 유사도 우선",
                )

            u_thresh = float(u_cfg.get("score_threshold", 0.5))
            if is_active and not is_bm25 and u_search == "similarity_score_threshold":
                u_thresh = st.slider(
                    "최소 유사도 (score_threshold)",
                    0.0, 1.0, u_thresh, 0.05,
                    key=f"u_thresh_{i}",
                    help="이 점수 미만의 문서는 결과에서 제외돼요.",
                )

            u_weight = float(u_cfg.get("weight", 1.0))
            if is_active and n_active > 1:
                u_weight = st.slider(
                    "가중치 (weight)",
                    0.1, 1.0, u_weight, 0.1,
                    key=f"u_weight_{i}",
                    help="다른 리트리버 대비 이 리트리버의 결과 비중이에요. 합계는 자동 정규화돼요.",
                )

            new_units.append({
                "type":            u_type,
                "k":               u_k,
                "search_type":     u_search,
                "active":          is_active,
                "weight":          u_weight if is_active else 1.0,
                "lambda_mult":     u_lambda,
                "score_threshold": u_thresh,
            })

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# 섹션 2: 리랭커
# ════════════════════════════════════════════════════════════════
st.markdown("### 리랭커")

r_enabled   = rerank_cfg.get("enabled", True)
r_final_k   = int(rerank_cfg.get("final_k", 3))
r_model_val = rerank_cfg.get("model", "bongsoo/kpf-cross-encoder-v1")

model_labels = list(RERANKER_MODELS.keys())
model_values = list(RERANKER_MODELS.values())
model_idx    = model_values.index(r_model_val) if r_model_val in model_values else 0

with st.container(border=True):
    rc1, rc2 = st.columns([1, 3])
    with rc1:
        r_enabled = st.toggle("사용", value=r_enabled)
    with rc2:
        r_model_label = st.selectbox(
            "모델",
            options=model_labels,
            index=model_idx,
            disabled=not r_enabled,
            help="한국어 콘텐츠에는 한국어 모델을 권장해요. 첫 실행 시 모델이 자동 다운로드돼요.",
        )

    r_final_k = st.slider(
        "최종 결과 수 (Final Top-K)",
        1, 10, r_final_k,
        disabled=not r_enabled,
        help="리랭킹 후 LLM에 전달할 최종 문서 수예요. 유닛의 검색 후보 수보다 작아야 해요.",
    )

new_rerank = {
    "enabled": r_enabled,
    "type":    "Cross-Encoder",
    "model":   RERANKER_MODELS.get(r_model_label, r_model_val) if r_enabled else r_model_val,
    "final_k": r_final_k,
}

# ── 저장 ─────────────────────────────────────────────────────────
new_cfg = {
    "alias":         "combined_search",
    "preprocessing": new_preproc,
    "units":         new_units,
    "reranker":      new_rerank,
}

if st.button("저장", type="primary", use_container_width=True):
    retr.save_retriever_config(new_cfg)
    st.success("저장됐어요.")
