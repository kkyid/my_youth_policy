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

# ── LangSmith traceable (없으면 no-op 데코레이터로 폴백) ──────────
try:
    from langsmith import traceable as _traceable
except ImportError:
    try:
        from langsmith.run_helpers import traceable as _traceable
    except ImportError:
        def _traceable(*, name="", run_type="chain", **kw):  # type: ignore
            def _deco(fn): return fn
            return _deco

# LLM_MODEL 전역 상수는 제거하고 retr.get_global_model()을 사용합니다.


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _llm(temperature: float = 0.2, model: str | None = None) -> ChatOpenAI:
    if not model:
        model = retr.get_global_model()
    return ChatOpenAI(model=model, temperature=temperature, max_retries=3, request_timeout=60)


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
- user_marital_status: 혼인 상태. "미혼"|"기혼"|"신혼부부" 중 하나, 없으면 null.
- user_housing_type: 희망 주거/지원 형태. "전세"|"월세"|"임대주택"|"이사비" 중 하나, 없으면 null.
- user_region: 언급된 서울 자치구명 (예: "도봉구", "강남구"). 없으면 null.
- user_is_homeless: 무주택자 여부 언급 시 (bool). 언급 없으면 null.

[사용자 질문]
{question}

[출력 - JSON만]
{{
  "user_age": null,
  "user_income_monthly": null,
  "user_income_annual": null,
  "user_income_pct": null,
  "user_marital_status": null,
  "user_housing_type": null,
  "user_region": null,
  "user_is_homeless": null
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

[중요 규칙]
1. 원본 질문의 의도를 넓히지 마세요.
   - "최대 한도"를 물으면 반드시 한도/대출한도 중심 질의만 만드세요.
   - "신청 기간"을 물으면 기간/마감/접수일 중심 질의만 만드세요.
   - "가능 여부"를 물으면 자격/조건 중심 질의만 만드세요.
2. 전세자금대출, 청약통장, 월세지원, 이사비지원처럼 상품/정책 유형이 명확하면 같은 유형 안에서만 표현을 바꾸세요.
3. 추천형 질문이 아닌데 여러 정책 추천 질의로 바꾸지 마세요.
4. RAPTOR summary 검색을 돕기 위해 핵심 키워드와 구체 조건(나이, 신혼부부, 전세자금, 한도 등)을 각 질의에 유지하세요.

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
    model: str | None = None,
) -> Optional[Dict[str, Any]]:
    """사용자 질문 → Chroma where 필터 dict (적용 가능한 조건 없으면 None)."""
    template = prompt or DEFAULT_SELF_QUERY_PROMPT
    chain = ChatPromptTemplate.from_template(template) | _llm(0.0, model=model) | StrOutputParser()
    raw = chain.invoke({"question": question})
    parsed = _extract_json(raw)
    if not isinstance(parsed, dict):
        return None

    conditions: List[Dict] = []

    # 1. 연령
    age = parsed.get("user_age")
    if isinstance(age, (int, float)) and age > 0:
        a = int(age)
        conditions.append({"age_min": {"$lte": a}})
        conditions.append({"age_max": {"$gte": a}})

    # 2. 소득 (income_max_man)
    income_annual = parsed.get("user_income_annual")
    if income_annual is None and parsed.get("user_income_monthly"):
        m = parsed.get("user_income_monthly")
        if isinstance(m, (int, float)) and m > 0:
            income_annual = int(m) * 12
    
    if isinstance(income_annual, (int, float)) and income_annual > 0:
        # DB의 income_max_man 필드와 비교 (단위: 만원)
        # ChromaDB는 None 값을 where 조건에 사용할 수 없으므로 $gte만 사용
        conditions.append({"$or": [
            {"income_max_man": {"$eq": 0}},
            {"income_max_man": {"$gte": int(income_annual)}},
        ]})

    # 3. 혼인 상태 (marital_status)
    marital = parsed.get("user_marital_status")
    if marital and isinstance(marital, str):
        conditions.append({"marital_status": {"$in": [marital, "무관"]}})

    # 4. 주거/지원 형태 (housing_type)
    housing = parsed.get("user_housing_type")
    if housing and isinstance(housing, str):
        conditions.append({"$or": [
            {"housing_type": {"$eq": housing}},
            {"loan_type": {"$eq": housing}},
            {"tags": {"$eq": housing}}
        ]})

    # 5. 지역 (region)
    region = parsed.get("user_region")
    if region and isinstance(region, str):
        # "도봉구" -> "서울 도봉구" 처럼 부분 매칭 지원을 위해 $contains 사용 권장하나 
        # Chroma 버전에 따라 다를 수 있음. 여기서는 일단 유연하게 처리.
        conditions.append({"region": {"$eq": region}})

    # 6. 무주택 여부
    homeless = parsed.get("user_is_homeless")
    if homeless is True:
        conditions.append({"$or": [
            {"is_homeless": {"$eq": True}},
            {"requires_no_house": {"$eq": True}}
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
    model: str | None = None,
) -> str:
    """질문 → 가상 정책 문서 텍스트 (검색 쿼리 대체용)."""
    template = prompt or DEFAULT_HYDE_PROMPT
    chain = ChatPromptTemplate.from_template(template) | _llm(0.3, model=model) | StrOutputParser()
    result = chain.invoke({"question": question})
    return result.strip() or question


# ---------------------------------------------------------------------------
# 0-C. Multi-Query: 질의 다변화
# ---------------------------------------------------------------------------
def apply_multi_query(
    question: str,
    prompt: str | None = None,
    model: str | None = None,
) -> List[str]:
    """원본 질문 + LLM 생성 3개 질의 = 총 4개 반환."""
    template = prompt or DEFAULT_MULTI_QUERY_PROMPT
    chain = ChatPromptTemplate.from_template(template) | _llm(0.3, model=model) | StrOutputParser()
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
    model: str | None = None,
) -> List[str]:
    """복잡한 질문 → 하위 질문 리스트."""
    template = prompt or DEFAULT_DECOMPOSITION_PROMPT
    chain = ChatPromptTemplate.from_template(template) | _llm(0.0, model=model) | StrOutputParser()
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
    model: str | None = None,
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
        hyde_text = apply_hyde(question, prompts_map.get("HyDE"), model=model)
        queries = [hyde_text]
    elif method == "Multi-Query":
        queries = apply_multi_query(question, prompts_map.get("Multi-Query"), model=model)
    elif method == "Decomposition":
        queries = apply_decomposition(question, prompts_map.get("Decomposition"), model=model)
    else:  # "없음"
        queries = [question]

    # ── Self-Query ────────────────────────────────────────────────
    metadata_filter = None
    sq_cfg = preproc_cfg.get("self_query", {})
    if sq_cfg.get("enabled"):
        metadata_filter = apply_self_query(question, sq_cfg.get("prompt"), model=model)

    return {"queries": queries, "metadata_filter": metadata_filter}


# ---------------------------------------------------------------------------
# 1. Ask + Decompose (정보 부족 점검 + 질문 분해 — 1회 LLM 호출)
# ---------------------------------------------------------------------------
ASK_AND_DECOMPOSE_PROMPT = """\
당신은 서울시 청년/신혼부부 주택 정책 상담사입니다.
두 가지 작업을 동시에 수행하세요.

━━━ 작업 1: 정보 충분성 판단 ━━━

[필수 정보 — 검색을 위해 반드시 필요한 3가지]
1. 연령대 (예: 만 27세, 20대 후반, 신혼부부 등)
2. 소득 또는 자산 정보 (예: 연봉 4000만원, 무직, 자산 1억 이하 등)
3. 가구 형태 (1인가구 / 신혼부부 / 자녀 유무)

[정보 인식 규칙]
1. 연령대: "살", "세", "대", "년생", "나이", "청년", "사회초년생" 등
2. 소득/자산: "연봉", "소득", "월급", "자산", "재산", "벌어", "수입", "만원", "억", "무직", "학생" 등 구체적 수치 포함 시
3. 가구 형태: "1인", "혼자", "미혼", "독신", "신혼", "부부", "결혼", "자녀", "아이", "가족" 등

[판단 규칙]
1. 전체 텍스트에서 정보를 취합하세요 (추가 정보 포함).
2. 필수 정보 3가지가 모두 확인되면 즉시 status: "READY".
3. ASK일 때만 빠진 항목 나열.

[응답 양식 - status가 ASK일 때의 question 필드]
"📋 필수 정보가 필요해요
아래 항목을 알려주시면 바로 검색해드릴게요:
• [빠진 항목명]

💡 아래 선택 정보도 함께 알려주시면 더 정확한 정책을 추천해드릴 수 있어요:
- 지역 (예: 영등포구, 금천구 등)
- 주거 형태 선호 (전세 / 월세 / 매입 / 임대주택)
- 자금 상황 (보유 자금, 대출 가능 여부)"

━━━ 작업 2: 검색 쿼리 분해 ━━━
사용자 질문을 주택 DB와 금융 DB 검색용으로 각각 분리하세요.
- housing_query: 주택/임대/공급/입주 관련 검색용 질의
- finance_query: 대출/금융/이자/자금/지원금 관련 검색용 질의
status가 ASK이면 housing_query와 finance_query는 빈 문자열로.

━━━ 출력 형식 — JSON만 ━━━
{{
  "status": "ASK" 또는 "READY",
  "missing": ["빠진 항목"],
  "question": "사용자에게 보낼 메시지 (READY면 빈 문자열)",
  "housing_query": "주택 검색 쿼리 (ASK면 빈 문자열)",
  "finance_query": "금융 검색 쿼리 (ASK면 빈 문자열)"
}}

[사용자 질문]
{question}
"""


def ask_and_decompose(
    question: str,
    ask_prompt: str | None = None,
    temperature: float = 0.0,
    model: str | None = None,
) -> Dict[str, Any]:
    """ask_or_ready + decompose_question을 1회 LLM 호출로 처리.

    반환: {status, missing, question, housing_query, finance_query}
    """
    template = ask_prompt or ASK_AND_DECOMPOSE_PROMPT
    chain = ChatPromptTemplate.from_template(template) | _llm(temperature, model=model) | StrOutputParser()
    raw = chain.invoke({"question": question})
    parsed = _extract_json(raw)
    if not isinstance(parsed, dict):
        return {
            "status": "READY", "missing": [], "question": "",
            "housing_query": question, "finance_query": question,
        }
    status = parsed.get("status", "READY")
    if status not in ("ASK", "READY"):
        status = "READY"
    return {
        "status":        status,
        "missing":       parsed.get("missing", []) or [],
        "question":      parsed.get("question", "") or "",
        "housing_query": parsed.get("housing_query") or question,
        "finance_query": parsed.get("finance_query") or question,
    }


# 하위 호환용 — Evaluation 페이지 등에서 단독으로 쓸 경우를 위해 유지
def ask_or_ready(
    question: str,
    ask_prompt: str | None = None,
    temperature: float = 0.0,
    model: str | None = None,
) -> Dict[str, Any]:
    """결과: {status: 'ASK'|'READY', missing: [...], question: '...'}"""
    result = ask_and_decompose(question, ask_prompt, temperature, model)
    return {
        "status":  result["status"],
        "missing": result["missing"],
        "question": result["question"],
    }


# ---------------------------------------------------------------------------
# 2. 질문 분해 (주택 vs 금융) — 하위 호환용
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


def decompose_question(question: str, model: str | None = None) -> Tuple[str, str]:
    chain = ChatPromptTemplate.from_template(DECOMPOSE_PROMPT) | _llm(0.0, model=model) | StrOutputParser()
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

    housing_vs = vector_db.get_cached_vectorstore(vector_db.HOUSING_COLLECTION)
    finance_vs = vector_db.get_cached_vectorstore(vector_db.FINANCE_COLLECTION)

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
    model: str | None = None,
) -> List[Dict[str, Any]]:
    template = selection_prompt or prompt_store.load_prompts()["selection"]
    contexts = _format_docs(housing_docs + finance_docs) or "(컨텍스트 없음)"
    chain = ChatPromptTemplate.from_template(template) | _llm(temperature, model=model) | StrOutputParser()
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
    model: str | None = None,
) -> str:
    template = report_prompt or prompt_store.load_prompts()["report"]
    chain = ChatPromptTemplate.from_template(template) | _llm(temperature, model=model) | StrOutputParser()
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
    model: str | None = None,
):
    """Report LLM 출력을 chunk 단위로 yield하는 제너레이터."""
    from langchain_openai import ChatOpenAI as _ChatOpenAI
    prompts_dict = prompts_dict or prompt_store.load_prompts()
    template     = prompts_dict.get("report") or prompt_store.load_prompts()["report"]
    temperature  = float(prompts_dict.get("report_temp", 0.3))

    if not model:
        model = retr.get_global_model()

    llm  = _ChatOpenAI(model=model, temperature=temperature, streaming=True)
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
@_traceable(name="pipeline_top3", run_type="chain")
def run_pipeline_top3(
    question: str,
    retriever_config: Dict[str, Any],
    prompts_dict: Dict[str, Any] | None = None,
    model: str | None = None,
    predecomposed: Dict[str, str] | None = None,
    pretrieved: Optional[Tuple[List[Document], List[Document]]] = None,
) -> Dict[str, Any]:
    """전처리 → 질문 분해 → 검색 → Top3 선정.

    predecomposed: ask_and_decompose()에서 미리 얻은 {housing_query, finance_query}.
    pretrieved:    투기적 검색으로 미리 얻은 (housing_docs, finance_docs).
                   둘 다 전달되면 분해 + 검색 LLM/IO를 모두 건너뜀.
    """
    prompts_dict = prompts_dict or prompt_store.load_prompts()
    preproc_cfg  = retriever_config.get("preprocessing", {})

    # ── 0. 전처리 ──────────────────────────────────────────────────
    preproc_result  = apply_preprocessing(question, preproc_cfg, model=model)
    queries         = preproc_result["queries"]          # 1~4개
    metadata_filter = preproc_result["metadata_filter"]  # Chroma where or None

    # ── 1~3. 각 쿼리별 분해 + 검색, 병합 ─────────────────────────
    all_housing_docs: List[Document] = []
    all_finance_docs: List[Document] = []
    housing_queries: List[str] = []
    finance_queries: List[str] = []

    for i, q in enumerate(queries):
        if i == 0 and predecomposed:
            hq = predecomposed.get("housing_query") or q
            fq = predecomposed.get("finance_query") or q
        else:
            hq, fq = decompose_question(q, model=model)
        housing_queries.append(hq)
        finance_queries.append(fq)

        # 첫 번째 쿼리는 투기적 검색 결과 재사용 (있을 경우)
        if i == 0 and pretrieved is not None:
            h_docs, f_docs = pretrieved
        else:
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
        model=model,
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
@_traceable(name="pipeline_report", run_type="chain")
def run_pipeline_report(
    question: str,
    top3: List[Dict[str, Any]],
    contexts: List[str] | None = None,
    prompts_dict: Dict[str, Any] | None = None,
    model: str | None = None,
) -> str:
    prompts_dict = prompts_dict or prompt_store.load_prompts()
    return make_report(
        question,
        top3,
        contexts=contexts,
        report_prompt=prompts_dict.get("report"),
        temperature=float(prompts_dict.get("report_temp", 0.3)),
        model=model,
    )


# ---------------------------------------------------------------------------
# 7. 후속 대화 의도 감지
# ---------------------------------------------------------------------------
FOLLOWUP_INTENT_PROMPT = """\
사용자가 Top3 정책 추천 결과를 받은 뒤 추가 메시지를 보냈습니다.
아래 메시지의 의도를 파악하세요.

[Top3 정책 목록]
{top3_names}

[사용자 메시지]
{message}

[의도 분류]
- "qualify": 자격요건/조건 추가 정보 제공 (나이, 소득, 지역, 무주택 여부, 혼인상태 등)
- "detail": 특정 정책에 대해 더 자세히 알고 싶다는 요청
- "other": 그 외

[출력 - JSON만]
{{
  "intent": "qualify" | "detail" | "other",
  "policies": ["정책명1", "정책명2"],
  "extra": "추출된 자격요건 설명 (qualify일 때)"
}}
"""


def detect_followup_intent(
    message: str,
    top3: List[Dict[str, Any]],
    model: str | None = None,
) -> Dict[str, Any]:
    """후속 메시지 의도 감지: qualify / detail / other."""
    top3_names = ", ".join(p.get("policy_name", "") for p in top3)
    chain = (
        ChatPromptTemplate.from_template(FOLLOWUP_INTENT_PROMPT)
        | _llm(0.0, model=model)
        | StrOutputParser()
    )
    raw = chain.invoke({"top3_names": top3_names, "message": message})
    parsed = _extract_json(raw)
    if not isinstance(parsed, dict):
        return {"intent": "other", "policies": [], "extra": ""}
    return {
        "intent":   parsed.get("intent", "other"),
        "policies": parsed.get("policies", []) or [],
        "extra":    parsed.get("extra", "") or "",
    }


# ---------------------------------------------------------------------------
# 8. 자격요건 기반 정책 필터링
# ---------------------------------------------------------------------------
QUALIFY_CHECK_PROMPT = """\
사용자가 제공한 자격요건 정보를 바탕으로, Top3 정책 각각에 대해
해당 사용자가 자격이 되는지 판단하세요.

[사용자 자격요건 정보]
{user_info}

[Top3 정책 및 요약]
{top3_json}

[참고 컨텍스트]
{contexts}

[판단 기준]
- 자격요건이 명확히 충족되지 않는 경우에만 "disqualified"로 표시
- 정보가 불충분해서 판단 불가한 경우는 "qualified"로 처리
- 판단 근거를 간단히 설명

[출력 - JSON만]
[
  {{
    "policy_name": "정책명",
    "status": "qualified" | "disqualified",
    "reason": "판단 근거 1줄"
  }}
]
"""


def check_qualification(
    user_info: str,
    top3: List[Dict[str, Any]],
    contexts: List[str] | None = None,
    model: str | None = None,
) -> List[Dict[str, Any]]:
    """사용자 자격요건 vs Top3 정책 → qualified/disqualified 리스트."""
    chain = (
        ChatPromptTemplate.from_template(QUALIFY_CHECK_PROMPT)
        | _llm(0.0, model=model)
        | StrOutputParser()
    )
    raw = chain.invoke({
        "user_info":  user_info,
        "top3_json":  json.dumps(top3, ensure_ascii=False, indent=2),
        "contexts":   "\n---\n".join(contexts or []) or "(컨텍스트 없음)",
    })
    parsed = _extract_json(raw)
    if isinstance(parsed, list):
        return parsed
    return [{"policy_name": p.get("policy_name", ""), "status": "qualified", "reason": ""} for p in top3]


# ---------------------------------------------------------------------------
# 9. 정책 상세 정보 생성
# ---------------------------------------------------------------------------
POLICY_DETAIL_PROMPT = """\
아래 정책에 대해 사용자 관점의 상세 정보를 충분히 길고 구체적으로 작성하세요.
수치, 금액, 조건, 절차가 컨텍스트에 있다면 반드시 포함하세요.

[사용자 원래 질문]
{question}

[정책명]
{policy_name}

[참고 컨텍스트]
{contexts}

[작성 지침]
- overview: 정책 목적, 지원 대상, 지원 규모, 운영 기관을 포함해 4~5문장으로 상세히
- eligibility: 나이·소득·자산·가구형태·무주택 여부 등 각 요건을 수치 포함해 5~7개 항목으로
- benefits: 지원 금액, 금리, 기간, 한도 등 구체적 수치가 담긴 혜택 4~6개 항목으로
- how_to_apply: 신청 채널, 단계별 절차, 담당 기관을 구체적으로
- deadline: 신청 기간, 모집 일정, 선발 방식(선착순/추첨/심사)까지 포함
- required_docs: 필요 서류를 구체적으로 5~8개 항목으로
- caution: 자격 박탈 조건, 중복 수혜 제한, 놓치기 쉬운 유의사항을 3~4문장으로

[출력 형식 - JSON만]
{{
  "policy_name": "정책명",
  "category": "주택" 또는 "금융",
  "overview": "목적·대상·규모·기관 포함 4~5문장",
  "eligibility": ["나이 조건 (수치 포함)", "소득 조건 (수치 포함)", "자산 조건", "가구형태 조건", "무주택 조건", "기타 조건"],
  "benefits": ["혜택1 (금액/금리 수치 포함)", "혜택2", "혜택3", "혜택4"],
  "how_to_apply": "단계별 신청 방법 (채널·절차·담당기관 포함)",
  "deadline": "신청기간 + 선발방식 포함",
  "required_docs": ["서류1", "서류2", "서류3", "서류4", "서류5"],
  "caution": "자격박탈·중복제한·유의사항 3~4문장",
  "url": "공식 URL (없으면 빈 문자열)"
}}
"""


def get_policy_detail(
    question: str,
    policy_name: str,
    contexts: List[str] | None = None,
    model: str | None = None,
) -> Dict[str, Any]:
    """특정 정책의 상세 정보 생성."""
    chain = (
        ChatPromptTemplate.from_template(POLICY_DETAIL_PROMPT)
        | _llm(0.2, model=model)
        | StrOutputParser()
    )
    raw = chain.invoke({
        "question":    question,
        "policy_name": policy_name,
        "contexts":    "\n---\n".join(contexts or []) or "(컨텍스트 없음)",
    })
    parsed = _extract_json(raw)
    if isinstance(parsed, dict):
        return parsed
    return {"policy_name": policy_name, "overview": raw[:300], "eligibility": [], "benefits": [],
            "how_to_apply": "", "deadline": "", "required_docs": [], "caution": "", "url": ""}


def prefetch_policy_details(
    question: str,
    top3: List[Dict[str, Any]],
    contexts: List[str] | None = None,
    model: str | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Top3 정책 상세 정보를 병렬로 미리 fetch. {policy_name: detail_dict}"""
    def _fetch(p: Dict) -> tuple:
        name = p.get("policy_name", "")
        return name, get_policy_detail(question, name, contexts, model)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(_fetch, p): p for p in top3}
        result: Dict[str, Dict[str, Any]] = {}
        for fut in concurrent.futures.as_completed(futures):
            try:
                name, detail = fut.result()
                result[name] = detail
            except Exception:
                pass
    return result


# ---------------------------------------------------------------------------
# 6-C. 전체 파이프라인 wrapper (Evaluation 페이지 등 단일 호출용)
# ---------------------------------------------------------------------------
@_traceable(name="run_pipeline", run_type="chain")
def run_pipeline(
    question: str,
    retriever_config: Dict[str, Any],
    prompts_dict: Dict[str, str] | None = None,
    model: str | None = None,
) -> Dict[str, Any]:
    """6-A + 6-B 를 순서대로 실행하는 wrapper."""
    prompts_dict = prompts_dict or prompt_store.load_prompts()

    # ── LangSmith Run에 리트리버 설정 메타데이터 태깅 ──────────────
    try:
        from langsmith.run_helpers import get_current_run_tree as _grt
        _rt = _grt()
        if _rt:
            _units   = retriever_config.get("units", [])
            _active  = [u for u in _units if u.get("active") and u.get("type") not in (None, "", "미설정")]
            _rerank  = retriever_config.get("reranker", {})
            _alias   = retriever_config.get("alias", "unknown")
            _rt.metadata = {
                "retriever_alias":   _alias,
                "retriever_units":   " + ".join(u.get("type", "?") for u in _active),
                "reranker":          "ON" if _rerank.get("enabled") else "OFF",
                "reranker_top_n":    _rerank.get("final_k", "-"),
                "llm_model":         model or retriever_config.get("llm_model", "gpt-4o-mini"),
            }
    except Exception:
        pass

    stage1 = run_pipeline_top3(question, retriever_config, prompts_dict, model=model)
    report = run_pipeline_report(question, stage1["top3"], stage1["contexts_text"], prompts_dict, model=model)

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
