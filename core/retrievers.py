"""Retriever 빌더 + 설정 영속화."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.retrievers import (
    ContextualCompressionRetriever,
    MultiQueryRetriever,
    EnsembleRetriever,
)
from langchain.retrievers.document_compressors import CrossEncoderReranker, LLMChainExtractor
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from pydantic import Field
import streamlit as st


# ── 한국어 토크나이저 ────────────────────────────────────────────
@st.cache_resource
def get_kiwi_tokenizer():
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        return lambda text: [token.form for token in kiwi.tokenize(text)]
    except ImportError:
        return None


@st.cache_resource
def load_cross_encoder(model_name: str):
    """Cross-Encoder 모델 캐싱 로드 (model_name 별로 캐싱)."""
    return HuggingFaceCrossEncoder(model_name=model_name)


# ── 경로 ─────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RETRIEVER_CONFIG_FILE = DATA_DIR / "retriever_config.json"

# ── 기본 설정 ─────────────────────────────────────────────────────
DEFAULT_CONFIG: Dict[str, Any] = {
    "alias":         "combined_search",
    "llm_model":     "gpt-4o-mini",
    "units": [
        {
            "type": "VectorStore", "k": 5, "search_type": "similarity",
            "active": True, "weight": 1.0,
            "lambda_mult": 0.5, "score_threshold": 0.5,
        },
        {
            "type": "미설정", "k": 5, "search_type": "similarity",
            "active": False, "weight": 1.0,
            "lambda_mult": 0.5, "score_threshold": 0.5,
        },
        {
            "type": "미설정", "k": 5, "search_type": "similarity",
            "active": False, "weight": 1.0,
            "lambda_mult": 0.5, "score_threshold": 0.5,
        },
    ],
    "reranker": {
        "enabled": True,
        "type": "Cross-Encoder",
        "model": "bongsoo/kpf-cross-encoder-v1",   # 한국어 모델 기본값
        "final_k": 3,
    },
    "extractor": {
        "enabled": False,
        "model": "gpt-4o-mini",
    },
    "preprocessing": {
        "self_query": {
            "enabled": False,
            "prompt": "",
        },
        "query_transform": {
            "method": "없음",
            "prompts": {
                "HyDE": "",
                "Multi-Query": "",
                "Decomposition": "",
            },
        },
    },
}


# ── Config I/O ────────────────────────────────────────────────────
def load_retriever_config() -> Dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RETRIEVER_CONFIG_FILE.exists():
        try:
            with RETRIEVER_CONFIG_FILE.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            if "units" not in cfg:
                return dict(DEFAULT_CONFIG)
            return cfg
        except Exception:
            return dict(DEFAULT_CONFIG)
    return dict(DEFAULT_CONFIG)


def save_retriever_config(cfg: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with RETRIEVER_CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def get_global_model() -> str:
    """전체 프로젝트에서 사용할 글로벌 LLM 모델명을 반환 (기본값: gpt-4o-mini)."""
    cfg = load_retriever_config()
    # 누락된 경우 기본값 gpt-4o-mini 반환
    return cfg.get("llm_model", DEFAULT_CONFIG["llm_model"])


def set_global_model(model_name: str) -> None:
    """글로벌 LLM 모델명을 설정 파일에 영구 저장."""
    cfg = load_retriever_config()
    cfg["llm_model"] = model_name
    save_retriever_config(cfg)


# ── 유틸 ─────────────────────────────────────────────────────────
def get_all_documents_from_vs(vs: Chroma) -> List[Document]:
    try:
        data = vs.get()
        return [
            Document(page_content=data["documents"][i], metadata=data["metadatas"][i])
            for i in range(len(data["ids"]))
        ]
    except Exception as e:
        logging.error(f"Error fetching documents from Chroma: {e}")
        return []


# ── Custom Retriever ──────────────────────────────────────────────
class CustomParentDocumentRetriever(BaseRetriever):
    """source 단위로 청크를 묶어 Parent Document처럼 반환하는 Retriever."""
    vectorstore: Any = Field(description="Chroma vectorstore")
    k: int = Field(default=3)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        docs = self.vectorstore.similarity_search(query, k=self.k)
        sources = {d.metadata.get("source") for d in docs if d.metadata.get("source")}
        if not sources:
            return docs
        parent_docs = []
        for source in sources:
            source_docs = self.vectorstore.similarity_search(" ", k=100, filter={"source": source})
            if source_docs:
                combined = "\n\n".join(d.page_content for d in source_docs)
                parent_docs.append(Document(
                    page_content=combined,
                    metadata={"source": source, "type": "parent_document"},
                ))
        return parent_docs


# ── 메인 빌더 ─────────────────────────────────────────────────────
def build_retriever(vectorstore: Chroma, config: Dict[str, Any], metadata_filter: Dict[str, Any] | None = None) -> BaseRetriever:
    """설정에 따라 리트리버를 조립해 반환."""
    cfg = config or load_retriever_config()

    active_retrievers: list = []
    weights: list = []

    for unit in cfg.get("units", DEFAULT_CONFIG["units"]):
        if not unit.get("active") or unit.get("type") == "미설정":
            continue

        utype   = unit.get("type")
        k       = int(unit.get("k", 5))
        stype   = unit.get("search_type", "similarity")
        weight  = float(unit.get("weight", 1.0))
        lmult   = float(unit.get("lambda_mult", 0.5))
        thresh  = float(unit.get("score_threshold", 0.5))

        # search_kwargs 조립
        if stype == "mmr":
            search_kwargs = {"k": k, "lambda_mult": lmult}
        elif stype == "similarity_score_threshold":
            search_kwargs = {"k": k, "score_threshold": thresh}
        else:
            search_kwargs = {"k": k}

        retriever = None

        if utype == "VectorStore":
            if metadata_filter:
                search_kwargs["filter"] = metadata_filter
            retriever = vectorstore.as_retriever(
                search_type=stype,
                search_kwargs=search_kwargs,
            )

        elif utype == "BM25":
            all_docs = get_all_documents_from_vs(vectorstore)
            if all_docs:
                tokenizer = get_kiwi_tokenizer()
                retriever = (
                    BM25Retriever.from_documents(all_docs, preprocess_func=tokenizer)
                    if tokenizer
                    else BM25Retriever.from_documents(all_docs)
                )
                retriever.k = k

        elif utype == "Multi-Query Retriever":
            base = vectorstore.as_retriever(search_type=stype, search_kwargs=search_kwargs)
            llm  = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            retriever = MultiQueryRetriever.from_llm(retriever=base, llm=llm)

        elif utype == "Parent Document Retriever":
            retriever = CustomParentDocumentRetriever(vectorstore=vectorstore, k=k)

        elif utype == "Self-Querying Retriever":
            metadata_field_info = [
                AttributeInfo(name="title",             description="정책/상품명",                               type="string"),
                AttributeInfo(name="category",          description="정책 분류: '주택' 또는 '금융'",              type="string"),
                AttributeInfo(name="region",            description="지원 가능 지역 (예: '서울 도봉구', '서울')",  type="string"),
                AttributeInfo(name="target",            description="지원 대상자: '청년', '신혼부부' 등",         type="string"),
                AttributeInfo(name="age_min",           description="지원 가능 최소 연령",                       type="integer"),
                AttributeInfo(name="age_max",           description="지원 가능 최대 연령",                       type="integer"),
                AttributeInfo(name="marital_status",    description="혼인 조건: '미혼', '기혼', '예비신혼부부', '무관'", type="string"),
                AttributeInfo(name="requires_no_house", description="무주택 조건 여부 (금융)",                   type="boolean"),
                AttributeInfo(name="is_homeless",       description="무주택 필수 여부 (주택)",                   type="boolean"),
                AttributeInfo(name="income_max_man",    description="소득 상한 (단위: 만원)",                    type="integer"),
                AttributeInfo(name="loan_limit_man",    description="최대 대출/지원 한도 (단위: 만원)",           type="integer"),
                AttributeInfo(name="asset_max_man",     description="자산 상한 (단위: 만원)",                    type="integer"),
                AttributeInfo(name="is_first_purchase", description="생애 최초 구매 조건 여부",                   type="boolean"),
                AttributeInfo(name="housing_type",      description="주택 유형 (예: '국민임대주택', '전세')",     type="string"),
            ]
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            retriever = SelfQueryRetriever.from_llm(
                llm, vectorstore,
                "청년 및 신혼부부를 위한 주택/금융 정책 정보",
                metadata_field_info,
                search_kwargs={"k": k},
            )

        # 하위 호환: 구형 "Ensemble Retriever" 타입 처리
        elif utype == "Ensemble Retriever":
            base_vs  = vectorstore.as_retriever(search_type=stype, search_kwargs=search_kwargs)
            all_docs = get_all_documents_from_vs(vectorstore)
            if all_docs:
                tokenizer = get_kiwi_tokenizer()
                base_bm25 = (
                    BM25Retriever.from_documents(all_docs, preprocess_func=tokenizer)
                    if tokenizer
                    else BM25Retriever.from_documents(all_docs)
                )
                base_bm25.k = k
                retriever = EnsembleRetriever(retrievers=[base_vs, base_bm25], weights=[0.5, 0.5])
            else:
                retriever = base_vs

        if retriever:
            active_retrievers.append(retriever)
            weights.append(weight)

    # fallback
    if not active_retrievers:
        return vectorstore.as_retriever(search_kwargs={"k": 3})

    # 1단계: 병렬 앙상블 (유닛별 가중치 적용)
    if len(active_retrievers) > 1:
        total = sum(weights)
        norm  = [w / total for w in weights]
        base_retriever = EnsembleRetriever(retrievers=active_retrievers, weights=norm)
    else:
        base_retriever = active_retrievers[0]

    # 2단계: 리랭커 (Cross-Encoder 순위 재조정)
    rerank_cfg = cfg.get("reranker", DEFAULT_CONFIG["reranker"])
    if rerank_cfg.get("enabled"):
        try:
            model_name = rerank_cfg.get("model", "bongsoo/kpf-cross-encoder-v1")
            model      = load_cross_encoder(model_name)
            compressor = CrossEncoderReranker(model=model, top_n=int(rerank_cfg.get("final_k", 3)))
            base_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever,
            )
        except Exception as e:
            logging.error(f"Reranker 초기화 실패: {e}")

    # 3단계: LLM-Extractor (핵심 내용 추출 — 리랭커 이후 적용)
    extractor_cfg = cfg.get("extractor", DEFAULT_CONFIG["extractor"])
    if extractor_cfg.get("enabled"):
        try:
            ext_model  = extractor_cfg.get("model", "gpt-4o-mini")
            ext_llm    = ChatOpenAI(model=ext_model, temperature=0)
            ext_compressor = LLMChainExtractor.from_llm(ext_llm)
            return ContextualCompressionRetriever(
                base_compressor=ext_compressor,
                base_retriever=base_retriever,
            )
        except Exception as e:
            logging.error(f"LLM-Extractor 초기화 실패: {e}")

    return base_retriever
