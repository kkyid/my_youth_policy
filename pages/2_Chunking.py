"""Chunking — 5단계 파이프라인."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Chunking", layout="wide", initial_sidebar_state="collapsed")

from core.ui import inject_ui, render_page_title  # noqa: E402
from core import chunker, vector_db               # noqa: E402

inject_ui()
render_page_title("Chunking", "각 단계 결과는 미리보기 버튼으로 확인할 수 있습니다.")


# =============================================================================
# 선택지 상수
# =============================================================================
LLM_MODELS = {
    "gpt-4o-mini  (빠름 · 저렴)": "gpt-4o-mini",
    "gpt-4o       (정확 · 고비용)": "gpt-4o",
}
EMBEDDING_MODELS = {
    "text-embedding-3-small  (빠름 · 저렴)": "text-embedding-3-small",
    "text-embedding-3-large  (정확 · 고비용)": "text-embedding-3-large",
}
# SemanticChunker threshold 타입별 슬라이더 범위
THRESHOLD_CFG = {
    "percentile":         dict(min=50.0,  max=99.0, default=95.0, step=0.5,
                               help="상위 N% 이상의 유사도 변화점을 경계로 사용 (높을수록 적은 청크)"),
    "standard_deviation": dict(min=0.5,   max=3.0,  default=1.5,  step=0.1,
                               help="평균에서 N 표준편차 이상 벗어난 지점을 경계로 사용"),
    "interquartile":      dict(min=0.5,   max=3.0,  default=1.5,  step=0.1,
                               help="IQR(사분위 범위)의 N배 이상인 지점을 경계로 사용"),
    "gradient":           dict(min=50.0,  max=99.0, default=95.0, step=0.5,
                               help="유사도 기울기 변화의 상위 N%를 경계로 사용"),
}


# =============================================================================
# Session state 초기화
# =============================================================================
DEFAULTS: dict = {
    "ck_filename":          None,
    "ck_raw_text":          None,
    "ck_full_summary":      None,
    "ck_target_collection": None,
    "ck_target_label":      None,
    "ck_stage1_docs":       None,
    "ck_stage1_method":     None,
    "ck_stage1_stats":      None,
    "ck_stage2_docs":       None,
    "ck_stage2_method":     None,
    "ck_stage2_stats":      None,
    "ck_final_docs":        None,
    "ck_cfg_seeded":        False,   # 설정 최초 로드 여부
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =============================================================================
# 설정 파일에서 위젯 초기값 주입 (최초 1회)
# =============================================================================
if not st.session_state.ck_cfg_seeded:
    _saved = chunker.load_chunking_config()
    s1m = _saved.get("stage1_method", chunker.DEFAULT_CHUNKING_CONFIG["stage1_method"])
    s1o = _saved.get("stage1_opts",   chunker.DEFAULT_CHUNKING_CONFIG["stage1_opts"])
    s2m = _saved.get("stage2_method", chunker.DEFAULT_CHUNKING_CONFIG["stage2_method"])
    s2o = _saved.get("stage2_opts",   chunker.DEFAULT_CHUNKING_CONFIG["stage2_opts"])
    lm  = _saved.get("llm_model",     "gpt-4o-mini")

    st.session_state["ck_w_s1_method"]    = s1m
    st.session_state["ck_w_s1_levels"]    = s1o.get("levels", ["#", "##"])
    st.session_state["ck_w_s1_sep"]       = s1o.get("extra_separators", [])
    st.session_state["ck_w_s1_fields"]    = s1o.get("fields", chunker.DEFAULT_POLICY_FIELDS[:10])
    st.session_state["ck_w_s1_minchar"]   = s1o.get("min_section_chars", 30)
    st.session_state["ck_w_s1_minmerge"]  = s1o.get("min_merge", 50)
    st.session_state["ck_w_s2_method"]    = s2m
    st.session_state["ck_w_s2_csize"]     = s2o.get("chunk_size", 800)
    st.session_state["ck_w_s2_coverlap"]  = s2o.get("chunk_overlap", 100)
    st.session_state["ck_w_s2_sep_pre"]   = s2o.get("separator_preset", "한국어 최적화")
    st.session_state["ck_w_s2_ttype"]     = s2o.get("threshold_type", "percentile")
    st.session_state["ck_w_s2_tamount"]   = s2o.get("threshold_amount", 95.0)
    st.session_state["ck_w_s2_emb"]       = s2o.get("embedding_model", "text-embedding-3-small")
    st.session_state["ck_w_s2_winsize"]   = s2o.get("window_size", 3)
    st.session_state["ck_w_s2_stride"]    = s2o.get("stride", 1)
    st.session_state["ck_w_s2_minmerge"]  = s2o.get("min_merge", 50)

    llm_labels  = list(LLM_MODELS.keys())
    llm_values  = list(LLM_MODELS.values())
    llm_idx     = llm_values.index(lm) if lm in llm_values else 0
    st.session_state["ck_w_llm_model"]   = llm_labels[llm_idx]

    st.session_state["ck_target_label"]   = _saved.get("target_label", "주택")
    st.session_state.ck_cfg_seeded = True


# =============================================================================
# 헬퍼
# =============================================================================
def _chunk_stats(docs: list) -> dict | None:
    if not docs:
        return None
    lengths = [len(d.page_content) for d in docs]
    return {
        "count":     len(docs),
        "avg":       sum(lengths) // len(lengths),
        "min":       min(lengths),
        "max":       max(lengths),
        "too_short": sum(1 for l in lengths if l < 100),
    }


def _render_stats(stats: dict | None, label: str = ""):
    if not stats:
        return
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("청크 수",   stats["count"])
    c2.metric("평균 길이", f"{stats['avg']:,}자")
    c3.metric("최소 길이", f"{stats['min']:,}자")
    c4.metric("최대 길이", f"{stats['max']:,}자")
    warn = stats["too_short"]
    c5.metric("100자 미만", warn,
              delta="주의" if warn else None,
              delta_color="inverse" if warn else "off")


def _show_modal(title: str, render_fn):
    if hasattr(st, "dialog"):
        @st.dialog(title, width="large")
        def _wrap():
            render_fn()
        _wrap()
    else:
        with st.expander(title, expanded=True):
            render_fn()


# =============================================================================
# ⚙️ 설정 패널
# =============================================================================
with st.expander("⚙️ 설정  —  LLM 모델 · 청킹 설정 저장/불러오기"):
    sc1, sc2 = st.columns([2, 1])
    with sc1:
        llm_label  = st.selectbox("LLM 모델 (요약 + 메타데이터 추출)",
                                   options=list(LLM_MODELS.keys()),
                                   key="ck_w_llm_model")
        ACTIVE_LLM = LLM_MODELS[llm_label]
    with sc2:
        st.markdown("&nbsp;", unsafe_allow_html=True)  # 세로 정렬용 padding
        if st.button("💾 현재 설정 저장", use_container_width=True):
            # 저장할 설정을 현재 위젯 값에서 수집
            _cfg_to_save = {
                "stage1_method": st.session_state.get("ck_w_s1_method", "건너뛰기"),
                "stage1_opts": {
                    "levels":           st.session_state.get("ck_w_s1_levels", ["#", "##"]),
                    "extra_separators": st.session_state.get("ck_w_s1_sep", []),
                    "fields":           st.session_state.get("ck_w_s1_fields", []),
                    "min_section_chars":st.session_state.get("ck_w_s1_minchar", 30),
                    "min_merge":        st.session_state.get("ck_w_s1_minmerge", 0),
                },
                "stage2_method": st.session_state.get("ck_w_s2_method", "건너뛰기"),
                "stage2_opts": {
                    "chunk_size":       st.session_state.get("ck_w_s2_csize", 800),
                    "chunk_overlap":    st.session_state.get("ck_w_s2_coverlap", 100),
                    "separator_preset": st.session_state.get("ck_w_s2_sep_pre", "한국어 최적화"),
                    "threshold_type":   st.session_state.get("ck_w_s2_ttype", "percentile"),
                    "threshold_amount": st.session_state.get("ck_w_s2_tamount", 95.0),
                    "embedding_model":  st.session_state.get("ck_w_s2_emb", "text-embedding-3-small"),
                    "window_size":      st.session_state.get("ck_w_s2_winsize", 3),
                    "stride":           st.session_state.get("ck_w_s2_stride", 1),
                    "min_merge":        st.session_state.get("ck_w_s2_minmerge", 50),
                },
                "llm_model": LLM_MODELS.get(st.session_state.get("ck_w_llm_model", ""), "gpt-4o-mini"),
                "target_label": st.session_state.get("ck_target_label", "주택"),
            }
            chunker.save_chunking_config(_cfg_to_save)
            st.success("설정이 저장됐어요. 다음 번에 이 페이지를 열면 자동으로 불러와요.")

st.markdown("---")

# =============================================================================
# 저장 대상 컬렉션
# =============================================================================
st.markdown("#### 저장 대상 컬렉션")
target_label = st.radio(
    "문서 분류를 먼저 선택하세요. (메타데이터 category 에 자동 주입)",
    options=["주택", "금융"],
    horizontal=True,
    index=0 if (st.session_state.ck_target_label or "주택") == "주택" else 1,
)
st.session_state.ck_target_label      = target_label
st.session_state.ck_target_collection = (
    vector_db.HOUSING_COLLECTION if target_label == "주택" else vector_db.FINANCE_COLLECTION
)

st.markdown("---")


# =============================================================================
# 1단계 — 파일 업로드 (로드 / 요약 분리)
# =============================================================================
st.markdown("### 1단계)  파일 업로드")
uploaded = st.file_uploader("파일을 드래그하세요", accept_multiple_files=False, type=["pdf", "txt", "md"])

b1, b2, b3 = st.columns([1.5, 1.2, 1])
run_load    = b1.button("📁 파일 로드 & 요약",      type="primary", use_container_width=True)
run_refine  = b2.button("✨ 마크다운 정제 (LLM)",   use_container_width=True)
show_summary= b3.button("결과 미리보기",             use_container_width=True)

if run_load:
    if not uploaded:
        st.warning("파일을 먼저 업로드하세요.")
    else:
        # 1. 파일 로드
        with st.spinner("문서 로드 중..."):
            try:
                text = chunker.read_uploaded_file(uploaded, uploaded.name)
            except Exception as e:
                st.error(f"파일 로드 실패: {e}")
                st.stop()
        
        st.session_state.ck_filename    = uploaded.name
        st.session_state.ck_raw_text    = text
        st.session_state.ck_full_summary= None
        for k in ("ck_stage1_docs", "ck_stage1_stats",
                  "ck_stage2_docs", "ck_stage2_stats", "ck_final_docs"):
            st.session_state[k] = None
        
        # 2. 자동으로 요약 생성
        with st.spinner("자동 전체 요약 생성 중..."):
            try:
                summary = chunker.summarize_full_document(text, model=ACTIVE_LLM)
                st.session_state.ck_full_summary = summary
                st.success(f"로드 및 요약 완료: {uploaded.name}")
            except Exception as e:
                st.error(f"요약 실패: {e}")
                st.session_state.ck_full_summary = "(요약 실패)"

if run_refine:
    if not st.session_state.ck_raw_text:
        st.warning("먼저 [📁 파일 로드] 를 눌러주세요.")
    else:
        progress_bar = st.progress(0, text="마크다운 정제 준비 중... (0%)")
        input_len = len(st.session_state.ck_raw_text[:8000])
        # 출력 길이는 대략 입력의 1.1배로 예상 (진행률 계산용)
        estimated_total = input_len * 1.1
        
        full_refined = ""
        try:
            # 1% 단위 부드러운 업데이트를 위해 스트리밍 사용
            for chunk in chunker.stream_refine_markdown_with_llm(
                st.session_state.ck_raw_text, model=ACTIVE_LLM
            ):
                full_refined += chunk
                # 진행률 계산 (최대 99%까지 표시, 완료 시 100%)
                curr_pct = min(len(full_refined) / estimated_total, 0.99)
                progress_bar.progress(curr_pct, text=f"마크다운 정제 중... ({int(curr_pct*100)}%)")
            
            progress_bar.progress(1.0, text="마크다운 정제 완료! (100%)")
            st.session_state.ck_raw_text = chunker._strip_markdown_code_blocks(full_refined)
            st.success("마크다운 정제가 완료되었습니다.")
        except Exception as e:
            st.error(f"정제 실패: {e}")


if show_summary:
    if not st.session_state.ck_full_summary:
        st.warning("먼저 [📁 파일 로드 & 요약] 을 눌러주세요.")
    else:
        def _r_summary():
            st.markdown(f"**파일:** `{st.session_state.ck_filename}`  ·  "
                        f"**원문 {len(st.session_state.ck_raw_text):,}자**")
            st.markdown("#### LLM 전체 요약")
            st.info(st.session_state.ck_full_summary)
            st.markdown("---")
            st.markdown("#### 마크다운 변환 결과 (Source)")
            st.code(st.session_state.ck_raw_text, language="markdown")
        _show_modal("1단계 결과 — 전체 요약", _r_summary)

st.markdown("---")


# =============================================================================
# 2단계 — 1차 청킹 (섹션 분할)
# =============================================================================
st.markdown("### 2단계)  1차 청킹  (섹션 분할)")

stage1_method = st.selectbox(
    "1차 청커 선택",
    options=["MarkdownSection (헤더 + 수평선 등)", "PolicyFieldChunker (정책 키워드)", "건너뛰기"],
    key="ck_w_s1_method",
)

stage1_opts: dict = {}
if stage1_method.startswith("MarkdownSection"):
    levels = st.multiselect("분할 헤더 레벨", options=["#", "##", "###", "####"],
                             key="ck_w_s1_levels")
    extra_seps = st.multiselect("추가 구분자", options=["---", "***", "___"],
                                 key="ck_w_s1_sep")
    # min_merge = st.slider(
    #     "짧은 청크 병합 기준 (자)",
    #     min_value=0, max_value=300, step=10,
    #     key="ck_w_s1_minmerge",
    #     help="이 글자 수 미만인 청크는 앞 청크에 자동 병합됩니다. 0 = 비활성",
    # )
    # stage1_opts = {"levels": levels, "extra_separators": extra_seps, "min_merge": min_merge}
    stage1_opts = {"levels": levels, "extra_separators": extra_seps, "min_merge": 0}

elif stage1_method.startswith("PolicyFieldChunker"):
    fields   = st.multiselect("정책 필드 키워드",
                               options=chunker.DEFAULT_POLICY_FIELDS,
                               key="ck_w_s1_fields")
    min_chars= st.slider("섹션 최소 길이(자)", 0, 200, key="ck_w_s1_minchar", step=5)
    stage1_opts = {"fields": fields, "min_section_chars": min_chars}

b1s1, b2s1 = st.columns(2)
run_stage1  = b1s1.button("1차 청킹 실행",       type="primary", use_container_width=True)
show_stage1 = b2s1.button("1차 청킹 결과 미리보기", use_container_width=True)

if run_stage1:
    if not st.session_state.ck_raw_text:
        st.warning("1단계를 먼저 진행하세요.")
    else:
        try:
            text = st.session_state.ck_raw_text
            fname = st.session_state.ck_filename or ""
            if stage1_method.startswith("MarkdownSection"):
                docs = chunker.chunk_with_markdown_section(
                    text,
                    headers=stage1_opts["levels"],
                    extra_separators=stage1_opts["extra_separators"],
                    extra_metadata={"source": fname},
                )
                # min_merge = stage1_opts.get("min_merge", 0)
                # if min_merge > 0:
                #     before = len(docs)
                #     docs = chunker.merge_short_chunks(docs, min_chars=min_merge)
                #     merged_cnt = before - len(docs)
                #     if merged_cnt > 0:
                #         st.info(f"짧은 청크 {merged_cnt}개를 앞 청크에 병합했어요.")
                pass
            elif stage1_method.startswith("PolicyFieldChunker"):
                docs = chunker.chunk_with_policy_field(
                    text,
                    field_keywords=stage1_opts["fields"],
                    min_section_chars=stage1_opts["min_section_chars"],
                    extra_metadata={"source": fname},
                )
            else:
                docs = [Document(page_content=text,
                                 metadata={"source": fname, "stage1": "skipped"})]

            st.session_state.ck_stage1_docs   = docs
            st.session_state.ck_stage1_method = stage1_method
            st.session_state.ck_stage1_stats  = _chunk_stats(docs)
            st.session_state.ck_stage2_docs   = None
            st.session_state.ck_stage2_stats  = None
            st.session_state.ck_final_docs    = None
            st.success(f"1차 청킹 완료 — {len(docs)}개 청크")
        except Exception as e:
            st.error(f"1차 청킹 실패: {e}")

if st.session_state.ck_stage1_stats:
    _render_stats(st.session_state.ck_stage1_stats, "1차")

if show_stage1:
    docs = st.session_state.ck_stage1_docs
    if not docs:
        st.warning("먼저 [1차 청킹 실행] 을 눌러주세요.")
    else:
        def _r_s1():
            st.markdown(f"**기법:** {st.session_state.ck_stage1_method}  ·  **총 {len(docs)}개**")
            for i, d in enumerate(docs, 1):
                with st.expander(f"청크 {i}  ·  {len(d.page_content):,}자  ·  meta: {d.metadata}",
                                 expanded=(i <= 2)):
                    st.code(d.page_content, language="markdown")
        _show_modal("2단계 결과 — 1차 청킹", _r_s1)

st.markdown("---")


# =============================================================================
# 3단계 — 2차 청킹 (세부 분할)
# =============================================================================
st.markdown("### 3단계)  2차 청킹  (세부 분할 · 선택사항)")

stage2_method = st.selectbox(
    "2차 청커 선택",
    options=["건너뛰기", "RecursiveCharacterTextSplitter",
             "SemanticChunker", "SentenceWindowChunker"],
    key="ck_w_s2_method",
)

stage2_opts: dict = {}

if stage2_method == "RecursiveCharacterTextSplitter":
    chunk_size    = st.slider("chunk_size (자)", 200, 6000, key="ck_w_s2_csize", step=50)
    # overlap 는 chunk_size 의 50% 를 초과 불가 — 실시간 경고
    max_overlap   = chunk_size // 2
    chunk_overlap = st.slider("chunk_overlap (자)", 0, max_overlap,
                               min(st.session_state.get("ck_w_s2_coverlap", 100), max_overlap),
                               step=10, key="ck_w_s2_coverlap",
                               help=f"chunk_size의 50% 이하({max_overlap}자)로 자동 제한돼요.")
    sep_preset    = st.selectbox("구분자 프리셋",
                                  options=list(chunker.SEPARATOR_PRESETS.keys()),
                                  key="ck_w_s2_sep_pre",
                                  help="한국어 문서엔 '한국어 최적화'를 권장해요.")
    stage2_opts = {
        "chunk_size":      chunk_size,
        "chunk_overlap":   chunk_overlap,
        "separator_preset": sep_preset,
    }

elif stage2_method == "SemanticChunker":
    if not chunker.is_semantic_available():
        st.error("`langchain-experimental` 미설치 — SemanticChunker 사용 불가")
    ttype  = st.selectbox("breakpoint_threshold_type",
                           ["percentile", "standard_deviation", "interquartile", "gradient"],
                           key="ck_w_s2_ttype")
    tcfg   = THRESHOLD_CFG[ttype]
    # 슬라이더 범위를 타입에 맞게 동적 조정
    saved_amount = st.session_state.get("ck_w_s2_tamount", tcfg["default"])
    clamped      = max(tcfg["min"], min(tcfg["max"], float(saved_amount)))
    tamount = st.slider(
        "threshold_amount",
        tcfg["min"], tcfg["max"], clamped, tcfg["step"],
        key="ck_w_s2_tamount",
        help=tcfg["help"],
    )
    emb_label = st.selectbox("임베딩 모델",
                              options=list(EMBEDDING_MODELS.keys()),
                              index=list(EMBEDDING_MODELS.values()).index(
                                  st.session_state.get("ck_w_s2_emb", "text-embedding-3-small")
                              ) if st.session_state.get("ck_w_s2_emb") in EMBEDDING_MODELS.values() else 0,
                              key="ck_w_s2_emb_label",
                              help="한국어 콘텐츠엔 large 모델이 더 정확하지만 비용이 높아요.")
    stage2_opts = {
        "threshold_type":   ttype,
        "threshold_amount": tamount,
        "embedding_model":  EMBEDDING_MODELS[emb_label],
    }

elif stage2_method == "SentenceWindowChunker":
    win_size = st.slider("window_size (문장 묶음 크기)", 1, 10, key="ck_w_s2_winsize")
    # stride 는 window_size 이하로 제한 — 초과 시 청크 누락 방지
    stride   = st.slider("stride (이동 문장 수)", 1, win_size,
                          min(st.session_state.get("ck_w_s2_stride", 1), win_size),
                          key="ck_w_s2_stride",
                          help=f"window_size({win_size}) 이하로 제한돼요. 초과하면 청크가 빠질 수 있어요.")
    stage2_opts = {"window_size": win_size, "stride": stride}

if stage2_method != "건너뛰기":
    s2_min_merge = st.slider(
        "짧은 청크 병합 기준 (자)",
        min_value=0, max_value=500, step=10,
        key="ck_w_s2_minmerge",
        help="이 글자 수 미만인 청크는 인접 청크에 자동 병합됩니다. 0 = 비활성",
    )
    stage2_opts["min_merge"] = s2_min_merge

b1s2, b2s2 = st.columns(2)
run_stage2  = b1s2.button("2차 청킹 실행",        type="primary", use_container_width=True)
show_stage2 = b2s2.button("2차 청킹 결과 미리보기", use_container_width=True)

if run_stage2:
    if stage2_method == "건너뛰기":
        if not st.session_state.ck_stage1_docs:
            st.warning("1차 청킹부터 진행하거나 2차 청커를 선택하세요.")
        else:
            st.session_state.ck_stage2_docs   = list(st.session_state.ck_stage1_docs)
            st.session_state.ck_stage2_method = "(건너뛰기 — 1차 결과 그대로)"
            st.session_state.ck_stage2_stats  = _chunk_stats(st.session_state.ck_stage2_docs)
            st.session_state.ck_final_docs    = None
            st.info("2차 청킹 건너뜀.")
    else:
        base_docs = (
            st.session_state.ck_stage1_docs or (
                [Document(page_content=st.session_state.ck_raw_text,
                          metadata={"source": st.session_state.ck_filename or ""})]
                if st.session_state.ck_raw_text else None
            )
        )
        if base_docs is None:
            st.warning("1단계부터 진행해주세요.")
        else:
            method_key = {
                "RecursiveCharacterTextSplitter": "recursive",
                "SemanticChunker":                "semantic",
                "SentenceWindowChunker":          "sentence_window",
            }[stage2_method]
            try:
                with st.spinner("2차 청킹 진행 중..."):
                    docs2 = chunker.apply_secondary_chunking(base_docs, method_key, stage2_opts)
                st.session_state.ck_stage2_docs   = docs2
                st.session_state.ck_stage2_method = stage2_method
                st.session_state.ck_stage2_stats  = _chunk_stats(docs2)
                st.session_state.ck_final_docs    = None
                st.success(f"2차 청킹 완료 — {len(docs2)}개 청크")
            except Exception as e:
                st.error(f"2차 청킹 실패: {e}")

if st.session_state.ck_stage2_stats:
    _render_stats(st.session_state.ck_stage2_stats, "2차")

if show_stage2:
    docs = st.session_state.ck_stage2_docs
    if not docs:
        st.warning("먼저 [2차 청킹 실행] 을 눌러주세요.")
    else:
        def _r_s2():
            st.markdown(f"**기법:** {st.session_state.ck_stage2_method}  ·  **총 {len(docs)}개**")
            show_n = min(20, len(docs))
            st.caption(f"앞에서 최대 {show_n}개 표시")
            for i, d in enumerate(docs[:show_n], 1):
                with st.expander(f"청크 {i}  ·  {len(d.page_content):,}자  ·  meta: {d.metadata}", expanded=(i <= 2)):
                    st.code(d.page_content, language="markdown")
        _show_modal("3단계 결과 — 2차 청킹", _r_s2)

st.markdown("---")


# =============================================================================
# 4단계 — 메타데이터 추출 + 결합
# =============================================================================
st.markdown("### 4단계)  LLM 메타데이터 추출 + 결합")
st.caption("각 청크에서 정책 메타데이터(JSON)를 추출하고, [전체 요약 + 메타데이터 + 원본]으로 Page Content를 만듭니다.")
if not st.session_state.ck_full_summary:
    st.info("💡 전체 요약이 없어도 진행할 수 있지만, 요약이 있으면 검색 품질이 높아져요.")

b1s4, b2s4 = st.columns(2)
run_combine = b1s4.button("메타데이터 추출 + 결합 실행", type="primary", use_container_width=True)
show_final  = b2s4.button("최종 청크 미리보기",           use_container_width=True)

if run_combine:
    base = (
        st.session_state.ck_stage2_docs
        or st.session_state.ck_stage1_docs
        or ([Document(page_content=st.session_state.ck_raw_text,
                      metadata={"source": st.session_state.ck_filename or ""})]
            if st.session_state.ck_raw_text else None)
    )
    if base is None:
        st.warning("최소 1단계는 진행해주세요.")
    else:
        progress = st.progress(0.0, text="메타데이터 추출 중...")

        def _cb(done: int, total: int):
            try:
                progress.progress(done / max(1, total), text=f"메타데이터 추출 {done}/{total}")
            except Exception:
                pass

        try:
            final = chunker.build_final_documents(
                chunks=base,
                full_summary=st.session_state.ck_full_summary or "",
                category=st.session_state.ck_target_label or "주택",
                extra_metadata={"source": st.session_state.ck_filename or ""},
                model=ACTIVE_LLM,
                progress_cb=_cb,
            )
            st.session_state.ck_final_docs = final
            progress.progress(1.0, text="완료")
            st.success(f"최종 청크 {len(final)}개 생성 완료")
        except Exception as e:
            st.error(f"실패: {e}")

if show_final:
    final = st.session_state.ck_final_docs
    if not final:
        st.warning("먼저 [메타데이터 추출 + 결합 실행] 을 눌러주세요.")
    else:
        def _r_final():
            st.markdown(f"**총 {len(final)}개**")
            show_n = min(15, len(final))
            st.caption(f"앞에서 최대 {show_n}개 표시")
            for i, d in enumerate(final[:show_n], 1):
                with st.expander(f"청크 {i}", expanded=(i <= 1)):
                    st.markdown("**Metadata:**")
                    st.json(d.metadata)
                    st.markdown("**Page Content:**")
                    st.code(d.page_content[:5000] + ("…" if len(d.page_content) > 5000 else ""))
        _show_modal("4단계 결과 — 최종 청크", _r_final)

st.markdown("---")


# =============================================================================
# 5단계 — ChromaDB 저장
# =============================================================================
st.markdown(
    f"### 5단계)  ChromaDB 저장 <span style='font-size: 0.5em; color: gray; font-weight: normal; margin-left: 10px;'>(Model: {vector_db.DEFAULT_EMBED_MODEL})</span>",
    unsafe_allow_html=True
)

if st.button("🚀 ChromaDB 에 저장 (현재 최종 청크 전체)", type="primary", use_container_width=True):
    final = st.session_state.ck_final_docs
    if not final:
        st.warning("4단계까지 진행한 뒤 저장하세요.")
    else:
        target = st.session_state.ck_target_collection
        try:
            n = vector_db.add_documents(target, final)
            st.success(f"{n}개 청크를 `{target}` 컬렉션에 저장했습니다.")
            st.session_state.ck_final_docs = None
        except Exception as e:
            st.error(f"저장 실패: {e}")

st.markdown("---")

stats = vector_db.collections_status()
m1, m2 = st.columns(2)
m1.metric("주택 정책 (policy_housing)", f"{stats['housing']:,} chunks")
m2.metric("금융 정책 (policy_finance)", f"{stats['finance']:,} chunks")

# 저장된 파일 목록 표시
with st.expander("📄 DB에 저장된 문서 목록 확인 (Unique Sources)"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**주택 정책 문서**")
        h_sources = vector_db.get_all_sources(vector_db.HOUSING_COLLECTION)
        if h_sources:
            for s in h_sources:
                st.markdown(f"- {s}")
        else:
            st.caption("저장된 문서 없음")
    with c2:
        st.markdown("**금융 정책 문서**")
        f_sources = vector_db.get_all_sources(vector_db.FINANCE_COLLECTION)
        if f_sources:
            for s in f_sources:
                st.markdown(f"- {s}")
        else:
            st.caption("저장된 문서 없음")

@st.dialog("⚠️ 컬렉션 초기화 확인")
def confirm_reset_dialog():
    target_label = st.session_state.ck_target_label
    st.warning(f"정말로 **{target_label}** 컬렉션의 모든 데이터를 삭제하시겠습니까?\n이 작업은 되돌릴 수 없으며, 저장된 모든 청크가 영구 삭제됩니다.")
    
    col1, col2 = st.columns(2)
    if col1.button("🔥 네, 전체 삭제", type="primary", use_container_width=True):
        target = st.session_state.ck_target_collection
        vector_db.reset_collection(target)
        st.success(f"`{target}` 컬렉션 초기화 완료")
        st.rerun()
    if col2.button("취소", use_container_width=True):
        st.rerun()

if st.button("🗑️ 현재 선택된 컬렉션 비우기", use_container_width=True):
    confirm_reset_dialog()

st.markdown("---")

# =============================================================================
# 🔄 메타데이터 마이그레이션
# =============================================================================
with st.expander("🔄 메타데이터 마이그레이션  —  기존 DB에 새 메타데이터 구조 적용"):
    st.caption(
        "기존 Chroma DB 청크의 **메타데이터 dict만** 새 METADATA_PROMPT 기준으로 재추출합니다.  \n"
        "임베딩 재생성 없이 LLM 호출만 사용하므로, Self-Query 필터를 쓰기 전 한 번만 실행하면 됩니다."
    )

    mg1, mg2 = st.columns([2, 1])
    with mg1:
        mg_target = st.radio(
            "대상 컬렉션",
            options=["주택 (policy_housing)", "금융 (policy_finance)", "전체"],
            horizontal=True,
            key="mg_target",
        )
    with mg2:
        mg_llm_label = st.selectbox(
            "LLM 모델",
            options=list(LLM_MODELS.keys()),
            key="mg_llm_label",
        )
        MG_LLM = LLM_MODELS[mg_llm_label]

    if st.button("🚀 메타데이터 재추출 시작", type="primary",
                  use_container_width=True, key="mg_run"):

        if mg_target == "주택 (policy_housing)":
            targets = [(vector_db.HOUSING_COLLECTION, "주택")]
        elif mg_target == "금융 (policy_finance)":
            targets = [(vector_db.FINANCE_COLLECTION, "금융")]
        else:
            targets = [
                (vector_db.HOUSING_COLLECTION, "주택"),
                (vector_db.FINANCE_COLLECTION, "금융"),
            ]

        total_updated = 0
        CHUNK_MARKER  = "[원본 청크]"

        for col_name, category in targets:
            try:
                vs       = vector_db.get_vectorstore(col_name)
                chroma_col = vs._collection
                result   = chroma_col.get(include=["documents", "metadatas"])
                ids      = result["ids"]
                doc_texts = result["documents"]
            except Exception as e:
                st.error(f"{col_name} 로드 실패: {e}")
                continue

            if not ids:
                st.info(f"{col_name}: 저장된 청크 없음, 건너뜁니다.")
                continue

            st.markdown(f"**{col_name}** — {len(ids)}개 청크")
            prog     = st.progress(0.0, text="준비 중...")
            errors   = 0
            new_metas = []

            for i, (doc_id, page_content) in enumerate(zip(ids, doc_texts)):
                # [원본 청크] 구분자가 있으면 해당 부분만, 없으면 전체 사용
                if CHUNK_MARKER in (page_content or ""):
                    raw_chunk = page_content.split(CHUNK_MARKER, 1)[1].strip()
                else:
                    raw_chunk = page_content or ""

                try:
                    meta = chunker.extract_chunk_metadata(
                        raw_chunk, category=category, model=MG_LLM
                    )
                except Exception:
                    meta = {"category": category}
                    errors += 1

                # Chroma는 None 값 저장 불가 — 제거
                meta_clean = {k: v for k, v in meta.items() if v is not None}
                new_metas.append(meta_clean)

                prog.progress(
                    (i + 1) / len(ids),
                    text=f"추출 중... {i + 1}/{len(ids)}"
                )

            # 일괄 메타데이터 업데이트 (임베딩 유지)
            try:
                chroma_col.update(ids=ids, metadatas=new_metas)
                total_updated += len(ids)
                msg = f"✅ {col_name}: {len(ids)}개 완료"
                if errors:
                    msg += f"  (LLM 오류 {errors}개 → category만 저장)"
                st.success(msg)
            except Exception as e:
                st.error(f"{col_name} 저장 실패: {e}")

        if total_updated:
            st.success(f"🎉 마이그레이션 완료 — 총 {total_updated}개 메타데이터 업데이트")
            # DB 상태 갱신
            stats2 = vector_db.collections_status()
            c1, c2 = st.columns(2)
            c1.metric("주택 정책", f"{stats2['housing']:,} chunks")
            c2.metric("금융 정책", f"{stats2['finance']:,} chunks")
