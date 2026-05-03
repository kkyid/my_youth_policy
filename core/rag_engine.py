"""LCEL 기반 RAG 파이프라인.

흐름:
0. 전처리 (Pre-processing)
   - Self-Query: 사용자 질문에서 메타데이터 필터 추출
   - Query Transform: HyDE / Multi-Query / Decomposition
1. Ask LLM → 정보 부족 시 되묻기 (status=ASK / READY)
2. 질문 분해 (주택 vs 금융)
3. 각 DB 에서 검색 (Self-Query 필터 적용)
4. Selection LLM 으로 종합 Top 3 선정
5. Report LLM 으로 종합 보고서 생성
"""
from __future__ import annotations

import json
import re
import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from . import vector_db
from . import retrievers as retr
from . import prompts as prompt_store

LLM_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _llm(temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=temperature)


def _extract_json(text: str) -> Any:
    """문자열에서 첫번째 JSON object/array 추출."""
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _format_docs(docs: List[Document]) -> str:
    out: List[str] = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        head = meta.get("policy_name") or meta.get("title") or meta.get("source") or f"doc_{i}"
        out.append(f"[{i}] {head}\n{d.page_content}")
    return "\n---\n".join(out)


def _dedup_docs(docs: List[Document]) -> List[Document]:
    """page_content 앞 80자 기준 중복 제거."""
    seen: set = set()
    result: List[Document] = []
    for d in docs:
        key = (d.page_content or "")[:80]
        if key not in seen:
            seen.add(key)
            result.append(d)
    return result


# ---------------------------------------------------------------------------
# 0. 전처리 — 기본 프롬프트 상수
# ---------------------------------------------------------------------------
DEFAULT_SELF_QUERY_PROMPT = """\
사용자 질문에서 정책 DB 필터 조건을 추출하세요.

[추출 규칙]
- user_age: 언급된 나이 (int). 없으면 null.
- user_income_monthly: 월소득/월급 (만원, int). 없으면 null.
- user_income_annual: 연봉/연소득 (만원, int). 없으면 null.
  ※ 월소득만 있으면 ×12 해서 annual도 채우세요.
- user_income_pct: 중위소득 % 언급 시 (int). 없으면 null.
- user_household: 가구형태. "1인가구"|"신혼부부"|"한부모가족"|"청년" 중 하나, 없으면 null.
- user_housing: 희망 주거형태. "전세"|"월세"|"매입"|"임대주택" 중 하나, 없으면 null.
- user_district: 언급된 서울 자치구명 (예: "강남구"). 없으면 null.

[사용자 질문]
{question}

[출력 - JSON만]
{{
  "user_age": null,
  "user_income_monthly": null,
  "user_income_annual": null,
  "user_income_pct": null,
  "user_household": null,
  "user_housing": null,
  "user_district": null
}}
"""

DEFAULT_HYDE_PROMPT = """\
아래 사용자 질문에 대한 답이 될 만한 가상의 정책 문서 단락을 작성하세요.
서울시 청년/신혼부부 주택·금융 정책 형식을 따르세요.
실제로 존재하는 것처럼 구체적으로 작성하되, 300자 이내로 작성하세요.

[사용자 질문]
{question}

[가상 정책 문서]
"""

DEFAULT_MULTI_QUERY_PROMPT = """\
아래 사용자 질문을 검색 관점이 다른 3가지 질의로 재작성하세요.
서울시 청년/신혼부부 주택·금융 정책 DB 검색에 최적화하세요.
원본 질문과 겹치지 않도록 다양한 표현·키워드를 활용하세요.

[원본 질문]
{question}

[출력 - JSON 배열, 정확히 3개]
["질의1", "질의2", "질의3"]
"""

DEFAULT_DECOMPOSITION_PROMPT = """\
복잡한 사용자 질문을 독립적으로 검색 가능한 하위 질문들로 분해하세요.
서울시 청년/신혼부부 정책 DB에서 각각 검색될 수 있어야 합니다.

[사용자 질문]
{question}

[출력 - JSON 배열, 2~4개]
["하위질문1", "하위질문2"]
"""


# ---------------------------------------------------------------------------
# 0-A. Self-Query: 메타데이터 필터 추출
# ---------------------------------------------------------------------------
def apply_self_query(
    question: str,
    prompt: str | None = None,
) -> Optional[Dict[str, Any]]:
    """사용자 질문 → Chroma where 필터 dict (적용 가능한 조건 없으면 None)."""
    template = prompt or DEFAULT_SELF_QUERY_PROMPT
    chain = ChatPromptTemplate.from_template(template) | _llm(0.0) | StrOutputParser()
    raw = chain.invoke({"question": question})
    parsed = _extract_json(raw)
    if not isinstance(parsed, dict):
        return None

    conditions: List[Dict] = []

    age = parsed.get("user_age")
    if isinstance(age, (int, float)) and age > 0:
        a = int(age)
        conditions.append({"age_min": {"$lte": a}})
        conditions.append({"age_max": {"$gte": a}})

    # 연소득 (monthly × 12 우선, 없으면 annual 직접)
    income_annual = parsed.get("user_income_annual")
    if income_annual is None and parsed.get("user_income_monthly"):
        m = parsed.get("user_income_monthly")
        if isinstance(m, (int, float)) and m > 0:
            income_annual = int(m) * 12
    if isinstance(income_annual, (int, float)) and income_annual > 0:
        # income_max=0 이면 "제한 없음" 정책이므로 필터에서 제외하면 놓칠 수 있음
        # → income_max >= annual OR income_max == 0 은 Chroma에서 $or 로 처리
        conditions.append({"$or": [
            {"income_max": {"$eq": 0}},
            {"income_max": {"$gte": int(income_annual)}},
        ]})

    income_pct = parsed.get("user_income_pct")
    if isinstance(income_pct, (int, float)) and income_pct > 0:
        conditions.append({"$or": [
            {"income_pct": {"$eq": 0}},
            {"income_pct": {"$gte": int(income_pct)}},
        ]})

    household = parsed.get("user_household")
    if household and isinstance(household, str):
        conditions.append({"household_type": {"$in": [household, "무관"]}})

    housing = parsed.get("user_housing")
    if housing and isinstance(housing, str):
        conditions.append({"housing_type": {"$in": [housing, "무관"]}})

    district = parsed.get("user_district")
    if district and isinstance(district, str):
        conditions.append({"$or": [
            {"district": {"$eq": district}},
            {"district": {"$eq": "서울특별시"}},
        ]})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ---------------------------------------------------------------------------
# 0-B. HyDE: 가상 문서 생성
# ---------------------------------------------------------------------------
def apply_hyde(
    question: str,
    prompt: str | None = None,
) -> str:
    """질문 → 가상 정책 문서 텍스트 (검색 쿼리 대체용)."""
    template = prompt or DEFAULT_HYDE_PROMPT
    chain = ChatPromptTemplate.from_template(template) | _llm(0.3) | StrOutputParser()
    result = chain.invoke({"question": question})
    return result.strip() or question


# ---------------------------------------------------------------------------
# 0-C. Multi-Query: 질의 다변화
# ---------------------------------------------------------------------------
def apply_multi_query(
    question: str,
    prompt: str | None = None,
) -> List[str]:
    """원본 질문 + LLM 생성 3개 질의 = 총 4개 반환."""
    template = prompt or DEFAULT_MULTI_QUERY_PROMPT
    chain = ChatPromptTemplate.from_template(template) | _llm(0.3) | StrOutputParser()
    raw = chain.invoke({"question": question})
    parsed = _extract_json(raw)
    variants: List[str] = []
    if isinstance(parsed, list):
        variants = [str(q) for q in parsed if q]
    return [question] + variants[:3]  # 원본 포함


# ---------------------------------------------------------------------------
# 0-D. Decomposition: 복합 질문 분해
# ---------------------------------------------------------------------------
def apply_decomposition(
    question: str,
    prompt: str | None = None,
) -> List[str]:
    """복잡한 질문 → 하위 질문 리스트."""
    template = prompt or DEFAULT_DECOMPOSITION_PROMPT
    chain = ChatPromptTemplate.from_template(template) | _llm(0.0) | StrOutputParser()
    raw = chain.invoke({"question": question})
    parsed = _extract_json(raw)
    subs: List[str] = []
    if isinstance(parsed, list):
        subs = [str(q) for q in parsed if q]
    return subs if subs else [question]


# ---------------------------------------------------------------------------
# 0-E. 전처리 통합 실행
# ---------------------------------------------------------------------------
def apply_preprocessing(
    question: str,
    preproc_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """전처리 설정에 따라 queries 리스트 + metadata_filter 반환.

    반환:
        {
          "queries": ["쿼리1", ...],   # 검색에 사용할 쿼리들
          "metadata_filter": {...} | None  # Chroma where 필터
        }
    """
    # ── Query Transform ───────────────────────────────────────────
    qt_cfg = preproc_cfg.get("query_transform", {})
    method = qt_cfg.get("method", "없음")
    prompts_map = qt_cfg.get("prompts", {})

    if method == "HyDE":
        hyde_text = apply_hyde(question, prompts_map.get("HyDE"))
        queries = [hyde_text]
    elif method == "Multi-Query":
        queries = apply_multi_query(question, prompts_map.get("Multi-Query"))
    elif method == "Decomposition":
        queries = apply_decomposition(question, prompts_map.get("Decomposition"))
    else:  # "없음"
        queries = [question]

    # ── Self-Query ────────────────────────────────────────────────
    metadata_filter = None
    sq_cfg = preproc_cfg.get("self_query", {})
    if sq_cfg.get("enabled"):
        metadata_filter = apply_self_query(question, sq_cfg.get("prompt"))

    return {"queries": queries, "metadata_filter": metadata_filter}


# ---------------------------------------------------------------------------
# 1. Ask LLM (정보 부족 점검)
# ---------------------------------------------------------------------------
def ask_or_ready(
    question: str,
    ask_prompt: str | None = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """결과: {status: 'ASK'|'READY', missing: [...], question: '...'}"""
    template = ask_prompt or prompt_store.load_prompts()["ask"]
    chain = ChatPromptTemplate.from_template(template) | _llm(temperature) | StrOutputParser()
    raw = chain.invoke({"question": question})
    parsed = _extract_json(raw)
    if not isinstance(parsed, dict):
        return {"status": "READY", "missing": [], "question": ""}
    status = parsed.get("status", "READY")
    if status not in ("ASK", "READY"):
        status = "READY"
    return {
        "status": status,
        "missing": parsed.get("missing", []) or [],
        "question": parsed.get("question", "") or "",
    }


# ---------------------------------------------------------------------------
# 2. 질문 분해 (주택 vs 금융)
# ---------------------------------------------------------------------------
DECOMPOSE_PROMPT = """\
사용자 질문을 두 갈래로 분해하세요.
- housing_query: 주택/임대/공급/입주 관련 검색용 질의
- finance_query: 대출/금융/이자/자금/지원금 관련 검색용 질의

[질문]
{question}

[출력 - JSON]
{{"housing_query": "...", "finance_query": "..."}}
"""


def decompose_question(question: str) -> Tuple[str, str]:
    chain = ChatPromptTemplate.from_template(DECOMPOSE_PROMPT) | _llm(0.0) | StrOutputParser()
    raw = chain.invoke({"question": question})
    parsed = _extract_json(raw) or {}
    return (
        parsed.get("housing_query") or question,
        parsed.get("finance_query") or question,
    )


# ---------------------------------------------------------------------------
# 3. 검색 (metadata_filter 지원)
# ---------------------------------------------------------------------------
def retrieve_candidates(
    housing_q: str,
    finance_q: str,
    config: Dict[str, Any],
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Document], List[Document]]:
    cfg = dict(config or {})

    housing_vs = vector_db.get_vectorstore(vector_db.HOUSING_COLLECTION)
    finance_vs = vector_db.get_vectorstore(vector_db.FINANCE_COLLECTION)

    housing_retriever = retr.build_retriever(housing_vs, cfg, metadata_filter=metadata_filter)
    finance_retriever = retr.build_retriever(finance_vs, cfg, metadata_filter=metadata_filter)

    # housing / finance 병렬 검색
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        h_fut = ex.submit(housing_retriever.invoke, housing_q) if housing_q else None
        f_fut = ex.submit(finance_retriever.invoke, finance_q) if finance_q else None
        housing_docs = h_fut.result() if h_fut else []
        finance_docs = f_fut.result() if f_fut else []

    for d in housing_docs:
        d.metadata = {**(d.metadata or {}), "category": "주택"}
    for d in finance_docs:
        d.metadata = {**(d.metadata or {}), "category": "금융"}

    return housing_docs, finance_docs


# ---------------------------------------------------------------------------
# 4. Selection (Top 3)
# ---------------------------------------------------------------------------
def select_top3(
    question: str,
    housing_docs: List[Document],
    finance_docs: List[Document],
    selection_prompt: str | None = None,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    template = selection_prompt or prompt_store.load_prompts()["selection"]
    contexts = _format_docs(housing_docs + finance_docs) or "(컨텍스트 없음)"
    chain = ChatPromptTemplate.from_template(template) | _llm(temperature) | StrOutputParser()
    raw = chain.invoke({"question": question, "contexts": contexts})
    parsed = _extract_json(raw)

    if isinstance(parsed, list) and parsed:
        result = []
        for i, item in enumerate(parsed[:3], start=1):
            if not isinstance(item, dict):
                continue
            result.append({
                "rank":        item.get("rank", i),
                "policy_name": item.get("policy_name", f"정책 {i}"),
                "category":    item.get("category", "기타"),
                "reason":      item.get("reason", ""),
                "url":         item.get("url", ""),
                "summary":     item.get("summary", ""),
            })
        if result:
            return result

    fallback: List[Dict[str, Any]] = []
    for i, d in enumerate((housing_docs + finance_docs)[:3], start=1):
        meta = d.metadata or {}
        fallback.append({
            "rank":        i,
            "policy_name": meta.get("policy_name") or meta.get("title") or f"정책 {i}",
            "category":    meta.get("category", "기타"),
            "reason":      "후보 컨텍스트에서 자동 선정",
            "url":         meta.get("url", ""),
            "summary":     (d.page_content or "")[:120],
        })
    return fallback


# ---------------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------------
def make_report(
    question: str,
    top3: List[Dict[str, Any]],
    contexts: List[str] | None = None,
    report_prompt: str | None = None,
    temperature: float = 0.3,
) -> str:
    template = report_prompt or prompt_store.load_prompts()["report"]
    chain = ChatPromptTemplate.from_template(template) | _llm(temperature) | StrOutputParser()
    contexts_str = "\n---\n".join(contexts) if contexts else "(컨텍스트 없음)"
    return chain.invoke({
        "question": question,
        "top3":     json.dumps(top3, ensure_ascii=False, indent=2),
        "contexts": contexts_str,
    })


# ---------------------------------------------------------------------------
# 5-B. Report 스트리밍 (app.py 실시간 출력용)
# ---------------------------------------------------------------------------
def stream_report(
    question: str,
    top3: List[Dict[str, Any]],
    contexts: List[str] | None = None,
    prompts_dict: Dict[str, Any] | None = None,
):
    """Report LLM 출력을 chunk 단위로 yield하는 제너레이터."""
    from langchain_openai import ChatOpenAI as _ChatOpenAI
    prompts_dict = prompts_dict or prompt_store.load_prompts()
    template     = prompts_dict.get("report") or prompt_store.load_prompts()["report"]
    temperature  = float(prompts_dict.get("report_temp", 0.3))

    llm  = _ChatOpenAI(model=LLM_MODEL, temperature=temperature, streaming=True)
    chain = ChatPromptTemplate.from_template(template) | llm | StrOutputParser()
    contexts_str = "\n---\n".join(contexts) if contexts else "(컨텍스트 없음)"

    yield from chain.stream({
        "question": question,
        "top3":     json.dumps(top3, ensure_ascii=False, indent=2),
        "contexts": contexts_str,
    })


# ---------------------------------------------------------------------------
# 6-A. 전처리 → 분해 → 검색 → Top3 (app.py 단계별 UX용)
# ---------------------------------------------------------------------------
def run_pipeline_top3(
    question: str,
    retriever_config: Dict[str, Any],
    prompts_dict: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """전처리 → 질문 분해 → 검색 → Top3 선정.

    전처리 설정은 retriever_config["preprocessing"] 에서 읽는다.
    Multi-Query / Decomposition 이면 여러 쿼리 각각 검색 후 병합.
    """
    prompts_dict = prompts_dict or prompt_store.load_prompts()
    preproc_cfg  = retriever_config.get("preprocessing", {})

    # ── 0. 전처리 ──────────────────────────────────────────────────
    preproc_result  = apply_preprocessing(question, preproc_cfg)
    queries         = preproc_result["queries"]          # 1~4개
    metadata_filter = preproc_result["metadata_filter"]  # Chroma where or None

    # ── 1~3. 각 쿼리별 분해 + 검색, 병합 ─────────────────────────
    all_housing_docs: List[Document] = []
    all_finance_docs: List[Document] = []
    housing_queries: List[str] = []
    finance_queries: List[str] = []

    for q in queries:
        hq, fq = decompose_question(q)
        housing_queries.append(hq)
        finance_queries.append(fq)
        h_docs, f_docs = retrieve_candidates(hq, fq, retriever_config, metadata_filter)
        all_housing_docs.extend(h_docs)
        all_finance_docs.extend(f_docs)

    # 중복 제거
    all_housing_docs = _dedup_docs(all_housing_docs)
    all_finance_docs = _dedup_docs(all_finance_docs)

    # ── 4. Top3 선정 ──────────────────────────────────────────────
    top3 = select_top3(
        question,
        all_housing_docs,
        all_finance_docs,
        selection_prompt=prompts_dict.get("selection"),
        temperature=float(prompts_dict.get("selection_temp", 0.0)),
    )

    return {
        "housing_query":   housing_queries[0] if housing_queries else question,
        "finance_query":   finance_queries[0] if finance_queries else question,
        "all_queries":     queries,
        "housing_docs":    all_housing_docs,
        "finance_docs":    all_finance_docs,
        "metadata_filter": metadata_filter,
        "top3":            top3,
        "contexts_text":   [d.page_content for d in all_housing_docs + all_finance_docs],
    }


# ---------------------------------------------------------------------------
# 6-B. 종합 보고서 생성
# ---------------------------------------------------------------------------
def run_pipeline_report(
    question: str,
    top3: List[Dict[str, Any]],
    contexts: List[str] | None = None,
    prompts_dict: Dict[str, Any] | None = None,
) -> str:
    prompts_dict = prompts_dict or prompt_store.load_prompts()
    return make_report(
        question,
        top3,
        contexts=contexts,
        report_prompt=prompts_dict.get("report"),
        temperature=float(prompts_dict.get("report_temp", 0.3)),
    )


# ---------------------------------------------------------------------------
# 6-C. 전체 파이프라인 wrapper (Evaluation 페이지 등 단일 호출용)
# ---------------------------------------------------------------------------
def run_pipeline(
    question: str,
    retriever_config: Dict[str, Any],
    prompts_dict: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """6-A + 6-B 를 순서대로 실행하는 wrapper."""
    prompts_dict = prompts_dict or prompt_store.load_prompts()

    stage1 = run_pipeline_top3(question, retriever_config, prompts_dict)
    report = run_pipeline_report(question, stage1["top3"], stage1["contexts_text"], prompts_dict)

    housing_docs = stage1["housing_docs"]
    finance_docs = stage1["finance_docs"]

    return {
        "housing_query":  stage1["housing_query"],
        "finance_query":  stage1["finance_query"],
        "all_queries":    stage1.get("all_queries", [question]),
        "metadata_filter": stage1.get("metadata_filter"),
        "housing_docs": [
            {"content": d.page_content, "metadata": d.metadata} for d in housing_docs
        ],
        "finance_docs": [
            {"content": d.page_content, "metadata": d.metadata} for d in finance_docs
        ],
        "top3":   stage1["top3"],
        "contexts_text": stage1["contexts_text"],
        "report":        report,
    }
