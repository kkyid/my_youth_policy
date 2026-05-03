import streamlit as st
st.set_page_config(page_title="Evaluation", layout="wide", initial_sidebar_state="collapsed")

import sys
import os
import time
import logging
from pathlib import Path
import pandas as pd

# ChromaDB 텔레메트리 에러 방지 (환경 변수 + 로깅 차단)
os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ui import inject_ui, render_page_title
from core import rag_engine, evaluator, retrievers as retr, prompts as prompt_store
from core import logger as exp_logger

inject_ui()
render_page_title("Evaluation")

st.markdown("""
<style>
.retr-badge {
    display:inline-block; background:#F1F5F9; border:1px solid #CBD5E1;
    border-radius:5px; padding:2px 10px; font-size:0.78rem; color:#475569;
}
.rerank-on {
    display:inline-block; background:#EEF2FF; border:1px solid #C7D2FE;
    border-radius:5px; padding:2px 10px; font-size:0.78rem; color:#4F46E5; font-weight:600;
}
.rerank-off {
    display:inline-block; background:#F8FAFC; border:1px solid #E2E8F0;
    border-radius:5px; padding:2px 10px; font-size:0.78rem; color:#94A3B8;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 평가4종 개선방법 모달
# =============================================================================
@st.dialog("📊 평가 4종 개선 방법", width="large")
def show_improvement_guide():
    st.markdown("""
### 1. Faithfulness (충실도)가 낮다면?
**"모델이 문서에 없는 말을 지어냄 (환각 현상)"**

원인: LLM이 가지고 있는 사전 지식과 검색된 문서가 충돌하거나, 모델 성능 부족.

**해결 방안:**
- **System Prompt 수정** — "반드시 주어진 문서 안에서만 답변해라", "모르면 모른다고 해라"라고 강력하게 제약.
- **Temperature 낮추기** — LLM의 창의성을 줄여서 사실 위주로만 답변하도록 설정 (예: 0.0 ~ 0.2).
- **Chain of Thought** — 답변 전 근거 문장을 먼저 추출하게 하는 등 사고 과정을 추가.

---

### 2. Answer Relevance (답변 관련성)가 낮다면?
**"대답은 하는데 질문의 의도에서 벗어남"**

원인: 답변 형식이 잘못되었거나, 불필요한 사족이 너무 김.

**해결 방안:**
- **Few-shot Prompting** — 원하는 답변의 예시(질문-답변 세트)를 프롬프트에 몇 개 넣어줌.
- **출력 형식 고정** — JSON이나 마크다운 등 특정 형식을 강제하여 답변의 가독성과 명확성을 높임.
- **모델 업그레이드** — 더 추론 능력이 뛰어난 상위 모델(예: GPT-4o, Claude 3.5 등)로 교체 검토.

---

### 3. Context Precision (맥락 정밀도)가 낮다면?
**"검색 결과에 쓰레기 정보가 너무 많음"**

원인: 질문과 상관없는 문서가 상위에 노출됨.

**해결 방안:**
- **Reranker 도입** — 임베딩 모델로 검색한 결과를 다시 한번 순위를 매겨서 정밀도를 높임.
- **임베딩 모델 교체** — 도메인 특화(예: 한국어 법률, 의료 등) 임베딩 모델로 변경.
- **쿼리 변환** — 질문을 검색에 더 적합한 형태로 LLM을 통해 재작성 (Query Rewriting).

---

### 4. Context Recall (맥락 재현율)이 낮다면?
**"답을 내기 위해 꼭 필요한 정보가 검색에서 빠짐"**

원인: 데이터베이스 자체에 내용이 없거나, 검색 방식이 한계가 있음.

**해결 방안:**
- **Chunk Size 조정** — 문서를 너무 작게 잘랐다면 문맥이 끊길 수 있으니 크기를 키움.
- **Hybrid Search** — 벡터 검색(Semantic)뿐만 아니라 키워드 검색(BM25)을 섞어서 사용.
- **데이터 추가** — 검색 대상 문서가 최신 정보인지, 혹은 누락된 데이터가 없는지 확인.
""")


# =============================================================================
# ⚙️ 설정
# =============================================================================
with st.expander("⚙️ 설정  —  평가 모델"):
    eval_model_label = st.selectbox(
        "평가 모델",
        options=list(evaluator.EVAL_MODELS.keys()),
        key="eval_model_label",
        help="채점에 사용할 LLM. 높은 모델일수록 정확하지만 비용이 높아요.",
    )
    ACTIVE_EVAL_MODEL = evaluator.EVAL_MODELS[eval_model_label]

    if evaluator.ragas_available():
        st.success("✅ ragas 라이브러리 감지 — 참고 정답 입력 시 실제 RAGAS로 평가합니다.")
    else:
        st.info("ℹ️ ragas 미설치 — LLM 시뮬레이션으로 평가합니다. (`pip install ragas datasets`로 설치 가능)")



# =============================================================================
# 테스트셋 관리
# =============================================================================
st.markdown("#### 테스트셋")
st.caption(
    "testset.json 에 JSON형태로 저장하고 불러오면 편합니다."
)

# 세션에 테스트셋 최초 로드
if "eval_testset_df" not in st.session_state:
    st.session_state.eval_testset_df = pd.DataFrame(evaluator.load_testset())

edited_df = st.data_editor(
    st.session_state.eval_testset_df,
    column_config={
        "question":     st.column_config.TextColumn(
            "질문",
            width="medium",
            help="추상적인 질문도 괜찮아요. 예) 서울사는 23세인데 월세가 너무 부담돼.",
        ),
        "ground_truth": st.column_config.TextColumn(
            "참고 정답 (ground truth)",
            width="large",
            help="이상적인 답변에 반드시 포함되어야 할 핵심 정보를 작성하세요. context_recall 채점에 사용됩니다.",
        ),
    },
    num_rows="dynamic",
    use_container_width=True,
    height=260,
    key="eval_testset_editor",
)

ts1, ts2 = st.columns(2)
with ts1:
    if st.button("💾 파일에 저장", use_container_width=True, help="현재 편집 중인 내용을 testset.json 파일에 영구 저장합니다."):
        rows = edited_df.to_dict("records")
        rows = [r for r in rows if str(r.get("question", "")).strip()]
        evaluator.save_testset(rows)
        st.session_state.eval_testset_df = pd.DataFrame(rows)
        # 위젯 편집 상태 초기화
        if "eval_testset_editor" in st.session_state:
            del st.session_state["eval_testset_editor"]
        st.success("파일 저장 완료!")
        st.rerun()

with ts2:
    if st.button("🔄 파일에서 불러오기", use_container_width=True, help="testset.json 파일에 저장된 내용을 다시 읽어옵니다."):
        st.session_state.eval_testset_df = pd.DataFrame(evaluator.load_testset())
        # 위젯 편집 상태 초기화
        if "eval_testset_editor" in st.session_state:
            del st.session_state["eval_testset_editor"]
        st.success("파일 로드 완료!")
        st.rerun()

# 참고 정답 없는 항목 경고
no_gt = sum(1 for _, row in edited_df.iterrows() if not str(row.get("ground_truth", "")).strip())
if no_gt > 0:
    st.warning(
        f"참고 정답이 비어있는 항목 **{no_gt}개** — "
        "해당 항목은 context_recall 이 0.0 으로 처리됩니다. "
        "정확한 평가를 위해 정답을 채워주세요."
    )

st.markdown("---")


# =============================================================================
# 현재 리트리버 설정 뱃지
# =============================================================================
try:
    _cfg   = retr.load_retriever_config()
    _units = [u for u in _cfg.get("units", []) if u.get("active") and u.get("type") not in (None, "", "미설정")]
    _rerank = _cfg.get("reranker", {})
    _parts  = []
    for u in _units:
        _parts.append(
            f'<span class="retr-badge">{u.get("type","?")} &nbsp;k={u.get("k","?")} &nbsp;{u.get("search_type","")}</span>'
        )
    if _rerank.get("enabled"):
        _parts.append(f'<span class="rerank-on">Reranker ● ON &nbsp;·&nbsp; Top {_rerank.get("final_k", 3)}</span>')
    else:
        _parts.append('<span class="rerank-off">Reranker ○ OFF</span>')
    _rows = "".join(f'<div style="margin-bottom:3px;">{p}</div>' for p in _parts)

except Exception:
    pass


# =============================================================================
# 평가 실행
# =============================================================================
ec1, ec2 = st.columns([3, 1], vertical_alignment="bottom")
with ec1:
    run_tag = st.text_input(
        "실험 태그",
        placeholder="예: 실험01_VectorStore_k5  (비워두면 날짜로 자동 생성)",
        help="같은 태그로 묶인 실험들이 Logs 탭 2에서 한 그룹으로 집계됩니다.",
        key="eval_run_tag",
    )
with ec2:
    run_eval = st.button("▶ 평가 실행", type="primary", use_container_width=True)

if run_eval:
    test_cases = [
        r for r in edited_df.to_dict("records")
        if str(r.get("question", "")).strip()
    ]
    if not test_cases:
        st.warning("질문을 1개 이상 입력하세요.")
        st.stop()

    import concurrent.futures

    cfg          = retr.load_retriever_config()
    prompts_dict = prompt_store.load_prompts()
    results      = []
    progress     = st.progress(0.0, text="평가 진행 중...")

    def evaluate_single_case(i, case):
        q  = str(case.get("question", "")).strip()
        gt = str(case.get("ground_truth", "")).strip()
        
        max_retries = 3
        last_error = ""

        for attempt in range(1, max_retries + 1):
            try:
                # 1. RAG 파이프라인 실행
                res = rag_engine.run_pipeline(q, cfg, prompts_dict)
                
                # 2. 메트릭 채점
                metrics = evaluator.evaluate_rag(
                    question=q,
                    contexts=res["contexts_text"],
                    answer=res["report"],
                    ground_truth=gt,
                    model=ACTIVE_EVAL_MODEL,
                )
                
                result_data = {
                    "question":     q,
                    "ground_truth": gt,
                    "top3":         res.get("top3", []),
                    "report":       res.get("report", ""),
                    **metrics,
                }

                # 3. Logs 기록
                try:
                    exp_logger.log_experiment({
                        "question":      q,
                        "ground_truth":  gt,  # 정답 데이터 추가
                        "retriever":     cfg,
                        "top3":          res["top3"],
                        "report":        res["report"],
                        "metrics": {
                            k: metrics[k]
                            for k in ["faithfulness", "answer_relevance", "context_precision", "context_recall"]
                        },
                        "tag": (run_tag.strip() or f"eval_{__import__('datetime').datetime.now().strftime('%m%d_%H%M')}"),
                    })
                except Exception:
                    pass

                return {"error": False, "index": i, "result": result_data}

            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    time.sleep(attempt * 2)  # 지수 백오프 (2초, 4초...)
                    continue
                else:
                    return {"error": True, "index": i, "question": q, "message": f"질문 #{i} 최종 실패 (3회 시도): {last_error}"}

    total_cases = len(test_cases)
    completed_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        futures = [executor.submit(evaluate_single_case, i, case) for i, case in enumerate(test_cases, start=1)]
        
        # Update progress as tasks complete
        for _ in concurrent.futures.as_completed(futures):
            completed_count += 1
            progress.progress(completed_count / total_cases, text=f"[{completed_count}/{total_cases}] 채점 완료")

        # Collect results in the original order
        for future in futures:
            res_data = future.result()
            if res_data["error"]:
                st.error(res_data["message"])
            else:
                results.append(res_data["result"])

    progress.progress(1.0, text="완료")

    if not results:
        st.warning("결과가 없습니다.")
        st.stop()

    # ── 결과 표 ────────────────────────────────────────────────────
    metric_cols = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]
    df = pd.DataFrame(results)

    st.markdown("#### 결과 요약")
    st.dataframe(
        df[["question"] + metric_cols + ["comment"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "question":          st.column_config.TextColumn("질문", width="large"),
            "faithfulness":      st.column_config.NumberColumn("Faithful",  format="%.3f"),
            "answer_relevance":  st.column_config.NumberColumn("Ans Rel",   format="%.3f"),
            "context_precision": st.column_config.NumberColumn("Ctx Prec",  format="%.3f"),
            "context_recall":    st.column_config.NumberColumn("Ctx Rec",   format="%.3f"),
            "comment":           st.column_config.TextColumn("총평", width="large"),
        },
    )

    # ── 평균 지표 ──────────────────────────────────────────────────
    st.markdown("#### 평균 지표")
    avg  = df[metric_cols].mean().to_dict()
    cols = st.columns(4)
    labels = {
        "faithfulness":      "Faithfulness",
        "answer_relevance":  "Answer Relevance",
        "context_precision": "Context Precision",
        "context_recall":    "Context Recall",
    }
    for col, key in zip(cols, metric_cols):
        with col:
            st.metric(labels[key], f"{avg[key]:.3f}")


else:
    st.info("테스트셋을 입력하고 실험 태그를 설정한 뒤 [▶ 평가 실행] 을 눌러 주세요. 결과는 Logs 페이지에 태그별로 기록됩니다.")
