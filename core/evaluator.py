"""RAGAS 지표 평가.

우선순위:
1. ragas 라이브러리 (ground_truth 있을 때, 설치된 경우)
2. LLM 시뮬레이션 — ground_truth 있으면 4종 전부 채점, 없으면 context_recall = 0

테스트셋(질문 + 정답 쌍) 저장/불러오기 포함.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# ── ragas 라이브러리 감지 ────────────────────────────────────────
_RAGAS_AVAILABLE = False
_RAGAS_NEW_API   = False  # True = ragas >= 0.2, False = 구 API

try:
    from ragas import evaluate as _ragas_eval  # type: ignore
    try:
        # ragas >= 0.2 새 API
        from ragas import EvaluationDataset, SingleTurnSample   # type: ignore
        from ragas.metrics import (                              # type: ignore
            Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall,
        )
        _RAGAS_AVAILABLE = True
        _RAGAS_NEW_API   = True
    except ImportError:
        # ragas < 0.2 구 API
        from ragas.metrics import (                              # type: ignore
            faithfulness as _r_faith,
            answer_relevancy as _r_ans,
            context_precision as _r_prec,
            context_recall as _r_rec,
        )
        from datasets import Dataset as _HFDataset              # type: ignore
        _RAGAS_AVAILABLE = True
        _RAGAS_NEW_API   = False
except ImportError:
    pass


# ── 상수 ────────────────────────────────────────────────────────
EVAL_MODELS: Dict[str, str] = {
    "gpt-4o-mini  (빠름 · 저렴)": "gpt-4o-mini",
    "gpt-4o       (정확 · 고비용)": "gpt-4o",
}

_DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
TESTSET_FILE = _DATA_DIR / "testset.json"

DEFAULT_TESTSET: List[Dict[str, str]] = [
    {
        "question":     "서울사는 23세인데 월세가 너무 부담돼.",
        "ground_truth": (
            "서울시 청년월세지원 사업을 통해 월 최대 20만원을 최대 12개월간 지원받을 수 있습니다. "
            "만 19~39세 무주택 청년이 대상이며 기준 중위소득 150% 이하 조건이 있습니다. "
            "서울주거포털(housing.seoul.go.kr)에서 신청할 수 있습니다."
        ),
    },
    {
        "question":     "신혼부부인데 서울에서 집 구하기 너무 어려워요.",
        "ground_truth": (
            "신혼부부 대상으로 장기안심주택, 신혼부부 전세자금 대출 보증 지원 등이 있습니다. "
            "결혼 7년 이내 무주택 부부가 대상이며, 소득 기준을 충족해야 합니다. "
            "SH공사 장기안심주택은 시세의 80% 이하로 최대 10년 거주가 가능합니다."
        ),
    },
    {
        "question":     "취업준비생이라 돈이 없는데 서울에서 살 수 있는 방법이 있을까요?",
        "ground_truth": (
            "청년 매입임대주택과 청년 전세임대주택을 통해 시세의 30~50% 수준으로 거주할 수 있습니다. "
            "또한 청년 전세자금 대출 보증을 통해 저금리 전세자금을 마련할 수 있습니다. "
            "소득이 없는 경우 부모 소득을 기준으로 심사하는 경우도 있으니 확인이 필요합니다."
        ),
    },
]


# ── 테스트셋 I/O ─────────────────────────────────────────────────
def load_testset() -> List[Dict[str, str]]:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    if TESTSET_FILE.exists():
        try:
            with TESTSET_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # 파일이 존재하면 비어있더라도(empty list) 그 데이터를 존중합니다.
            if isinstance(data, list):
                return data
        except Exception:
            pass
    # 파일이 아예 없거나 에러가 날 때만 기본셋 반환
    return [dict(t) for t in DEFAULT_TESTSET]


def save_testset(testset: List[Dict[str, str]]) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with TESTSET_FILE.open("w", encoding="utf-8") as f:
        json.dump(testset, f, ensure_ascii=False, indent=2)


def ragas_available() -> bool:
    return _RAGAS_AVAILABLE


# ── 평가 프롬프트 ─────────────────────────────────────────────────
# ground_truth 있을 때 — context_recall 포함 4종 정밀 채점
_EVAL_PROMPT_WITH_GT = """\
당신은 RAG 시스템 평가 전문가입니다. 아래 정보를 바탕으로 4개 지표를 채점하세요.

[지표 정의]
- faithfulness (충실도): 답변의 각 주장이 검색 컨텍스트에 근거하는 비율.
  컨텍스트에 없는 내용이 답변에 들어있을수록 점수가 낮아집니다.
- answer_relevance (답변 관련성): 답변이 질문의 핵심 의도를 충족하는 정도.
  동문서답이거나 불필요한 내용이 많으면 낮아집니다.
- context_precision (컨텍스트 정밀도): 검색된 컨텍스트 중 실제로 유용한 비율.
  무관한 문서가 상위에 포함될수록 낮아집니다.
- context_recall (컨텍스트 재현율): 참고 정답의 핵심 정보가 컨텍스트에서 찾아지는 비율.
  참고 정답의 내용이 컨텍스트에 없으면 낮아집니다.

[질문]
{question}

[검색된 컨텍스트]
{contexts}

[생성된 답변]
{answer}

[참고 정답 (ground truth)]
{ground_truth}

[출력 형식 - JSON 만, 다른 텍스트 금지]
{{
  "faithfulness": 0.0,
  "faithfulness_reason": "근거 1줄",
  "answer_relevance": 0.0,
  "answer_relevance_reason": "근거 1줄",
  "context_precision": 0.0,
  "context_precision_reason": "근거 1줄",
  "context_recall": 0.0,
  "context_recall_reason": "근거 1줄",
  "comment": "전체 총평 1문장"
}}
"""

# ground_truth 없을 때 — context_recall 채점 불가 (0.0 고정)
_EVAL_PROMPT_NO_GT = """\
당신은 RAG 시스템 평가 전문가입니다. 참고 정답 없이 3개 지표를 채점합니다.
(context_recall 은 참고 정답이 없어 채점 불가 — 자동으로 0.0 처리)

[지표 정의]
- faithfulness: 답변의 주장이 컨텍스트에 근거하는 비율
- answer_relevance: 답변이 질문의 핵심 의도를 충족하는 정도
- context_precision: 검색 컨텍스트 중 실제로 유용한 비율

[질문]
{question}

[검색된 컨텍스트]
{contexts}

[생성된 답변]
{answer}

[출력 형식 - JSON 만, 다른 텍스트 금지]
{{
  "faithfulness": 0.0,
  "faithfulness_reason": "근거 1줄",
  "answer_relevance": 0.0,
  "answer_relevance_reason": "근거 1줄",
  "context_precision": 0.0,
  "context_precision_reason": "근거 1줄",
  "context_recall": 0.0,
  "context_recall_reason": "참고 정답 없음 — 채점 불가",
  "comment": "전체 총평 1문장"
}}
"""


# ── 유틸 ────────────────────────────────────────────────────────
def _extract_json(text: str) -> Dict[str, Any]:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def _safe_float(val: Any) -> float:
    try:
        return max(0.0, min(1.0, float(val)))
    except Exception:
        return 0.0


# ── ragas 라이브러리 직접 호출 ───────────────────────────────────
def _evaluate_with_ragas(
    question: str,
    contexts: List[str],
    answer: str,
    ground_truth: str,
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    
    # 평가용 LLM 및 임베딩 명시적 설정 (JSON 모드 강제)
    eval_llm = ChatOpenAI(
        model=model, 
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    eval_embeddings = OpenAIEmbeddings()

    if _RAGAS_NEW_API:
        sample  = SingleTurnSample(
            user_input=question,
            retrieved_contexts=contexts,
            response=answer,
            reference=ground_truth,
        )
        dataset = EvaluationDataset(samples=[sample])
        
        # 메트릭 객체 생성 시 LLM 주입
        m_faith = Faithfulness(llm=eval_llm)
        m_rel   = AnswerRelevancy(llm=eval_llm, embeddings=eval_embeddings)
        m_prec  = ContextPrecision(llm=eval_llm)
        m_recall = ContextRecall(llm=eval_llm)

        result  = _ragas_eval(
            dataset,
            metrics=[m_faith, m_rel, m_prec, m_recall],
        )
        row = result.to_pandas().iloc[0]
    else:
        ds = _HFDataset.from_dict({
            "question":     [question],
            "answer":       [answer],
            "contexts":     [contexts],
            "ground_truth": [ground_truth],
        })
        
        # 구 API 대응: 메트릭에 llm 직접 할당
        _r_faith.llm = eval_llm
        _r_ans.llm   = eval_llm
        _r_ans.embeddings = eval_embeddings
        _r_prec.llm  = eval_llm
        _r_rec.llm   = eval_llm

        result = _ragas_eval(
            ds,
            metrics=[_r_faith, _r_ans, _r_prec, _r_rec],
        )
        row = result.to_pandas().iloc[0]

    return {
        "faithfulness":               _safe_float(row.get("faithfulness", 0.0)),
        "faithfulness_reason":        "(ragas 라이브러리 평가)",
        "answer_relevance":           _safe_float(row.get("answer_relevancy", 0.0)),
        "answer_relevance_reason":    "(ragas 라이브러리 평가)",
        "context_precision":          _safe_float(row.get("context_precision", 0.0)),
        "context_precision_reason":   "(ragas 라이브러리 평가)",
        "context_recall":             _safe_float(row.get("context_recall", 0.0)),
        "context_recall_reason":      "(ragas 라이브러리 평가)",
        "comment":                    f"ragas 라이브러리 직접 평가 (모델: {model})",
    }


# ── LLM 시뮬레이션 평가 ──────────────────────────────────────────
def _evaluate_with_llm(
    question: str,
    contexts: List[str],
    answer: str,
    ground_truth: str,
    model: str,
) -> Dict[str, Any]:
    llm           = ChatOpenAI(model=model, temperature=0)
    context_block = "\n---\n".join(contexts) if contexts else "(컨텍스트 없음)"
    has_gt        = bool(ground_truth and ground_truth.strip())

    if has_gt:
        template = _EVAL_PROMPT_WITH_GT
        inputs   = {
            "question":     question,
            "contexts":     context_block,
            "answer":       answer or "(답변 없음)",
            "ground_truth": ground_truth,
        }
    else:
        template = _EVAL_PROMPT_NO_GT
        inputs   = {
            "question": question,
            "contexts": context_block,
            "answer":   answer or "(답변 없음)",
        }

    chain  = ChatPromptTemplate.from_template(template) | llm | StrOutputParser()
    raw    = chain.invoke(inputs)
    parsed = _extract_json(raw)

    return {
        "faithfulness":               _safe_float(parsed.get("faithfulness")),
        "faithfulness_reason":        parsed.get("faithfulness_reason", ""),
        "answer_relevance":           _safe_float(parsed.get("answer_relevance")),
        "answer_relevance_reason":    parsed.get("answer_relevance_reason", ""),
        "context_precision":          _safe_float(parsed.get("context_precision")),
        "context_precision_reason":   parsed.get("context_precision_reason", ""),
        "context_recall":             _safe_float(parsed.get("context_recall")),
        "context_recall_reason":      parsed.get("context_recall_reason", "참고 정답 없음 — 채점 불가" if not has_gt else ""),
        "comment":                    parsed.get("comment", ""),
    }


# ── 메인 진입점 ──────────────────────────────────────────────────
def evaluate_rag(
    question: str,
    contexts: List[str],
    answer: str,
    ground_truth: str = "",
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """RAGAS 지표 평가. ragas 라이브러리 우선, 불가 시 LLM 시뮬레이션."""
    has_gt = bool(ground_truth and ground_truth.strip())

    if _RAGAS_AVAILABLE and has_gt:
        try:
            return _evaluate_with_ragas(question, contexts, answer, ground_truth, model=model)
        except Exception:
            pass  # fallback to LLM

    return _evaluate_with_llm(question, contexts, answer, ground_truth, model)
