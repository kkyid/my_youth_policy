"""ChromaDB 매니저.

- housing / finance 두 컬렉션을 분리 관리한다.
- 메타데이터 포함 임베딩 저장 / 검색 / 통계 조회 기능 제공.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma
import chromadb
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# ROOT 경로 설정 및 .env 로드
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ROOT_DIR = DATA_DIR.parent
load_dotenv(ROOT_DIR / ".env")

# 주택·금융 DB는 별도 디렉토리에 분리 저장
HOUSING_DB_DIR = DATA_DIR / "db" / "chroma_db_v2"
FINANCE_DB_DIR  = DATA_DIR / "db" / "chroma_finance"
HOUSING_DB_DIR.mkdir(parents=True, exist_ok=True)
FINANCE_DB_DIR.mkdir(parents=True, exist_ok=True)

HOUSING_COLLECTION = "youth_housing_policy"
FINANCE_COLLECTION = "youth_finance_policy"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


def get_embeddings(model: Optional[str] = None) -> OpenAIEmbeddings:
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAIEmbeddings(model=model or DEFAULT_EMBED_MODEL, api_key=api_key)


def _db_dir(collection: str) -> Path:
    """컬렉션 이름으로 해당 DB 디렉토리를 반환."""
    if collection == FINANCE_COLLECTION:
        return FINANCE_DB_DIR
    return HOUSING_DB_DIR


def get_vectorstore(collection: str, embeddings: Optional[OpenAIEmbeddings] = None) -> Chroma:
    from chromadb.config import Settings
    embeddings = embeddings or get_embeddings()
    client = chromadb.PersistentClient(
        path=str(_db_dir(collection)),
        settings=Settings(anonymized_telemetry=False),
        tenant="default_tenant",
        database="default_database",
    )
    return Chroma(
        client=client,
        collection_name=collection,
        embedding_function=embeddings,
    )


try:
    import streamlit as st

    @st.cache_resource(show_spinner=False)
    def get_cached_vectorstore(collection: str) -> Chroma:
        """앱 시작 시 한 번만 로드하고 세션 간 재사용."""
        return get_vectorstore(collection)

except ImportError:
    def get_cached_vectorstore(collection: str) -> Chroma:
        return get_vectorstore(collection)


def add_documents(collection: str, docs: Iterable[Document]) -> int:
    docs = list(docs)
    if not docs:
        return 0
    vs = get_vectorstore(collection)
    vs.add_documents(docs)
    return len(docs)


def reset_collection(collection: str) -> None:
    """컬렉션을 완전히 비웁니다."""
    try:
        vs = get_vectorstore(collection)
        # 1. 모든 ID를 가져와서 삭제 (가장 확실한 방법)
        existing = vs._collection.get()
        if existing and existing['ids']:
            vs._collection.delete(ids=existing['ids'])
        
        # 2. 컬렉션 자체 삭제 시도 (필요한 경우)
        vs.delete_collection()
    except Exception as e:
        print(f"Error resetting collection {collection}: {e}")


def count_documents(collection: str) -> int:
    try:
        vs = get_vectorstore(collection)
        return vs._collection.count()
    except Exception as e:
        # 에러 원인을 파악하기 위해 무조건 출력합니다.
        print(f"!!! DB 조회 에러 ({collection}): {e}")
        return 0


def get_all_sources(collection: str) -> List[str]:
    """컬렉션에 저장된 문서들의 유니크한 source(파일명) 목록을 반환합니다."""
    try:
        vs = get_vectorstore(collection)
        # 대량의 데이터일 경우를 대비해 metadata만 가져옵니다.
        result = vs._collection.get(include=["metadatas"])
        metadatas = result.get("metadatas", [])
        
        sources = set()
        for meta in metadatas:
            if meta and "source" in meta:
                sources.add(meta["source"])
        
        return sorted(list(sources))
    except Exception as e:
        print(f"Error getting sources for {collection}: {e}")
        return []


def similarity_search(
    collection: str,
    query: str,
    k: int = 2,
) -> List[Document]:
    vs = get_vectorstore(collection)
    return vs.similarity_search(query, k=k)


def collections_status() -> dict:
    return {
        "housing": count_documents(HOUSING_COLLECTION),
        "finance": count_documents(FINANCE_COLLECTION),
    }
