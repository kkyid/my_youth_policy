"""실험 기록 로거.

매 RAG 실행마다 (chunker / prompts / retriever / 결과) 를 JSONL + CSV 로 기록.
Logs 페이지에서 페이징 조회한다.
"""
from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOG_DIR = DATA_DIR / "logs"
JSONL_FILE = LOG_DIR / "experiments.jsonl"
CSV_FILE = LOG_DIR / "experiments.csv"

CSV_FIELDS = [
    "id",
    "ts",
    "question",
    "retriever_type",
    "k",
    "report_excerpt",
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
]


def _ensure_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_experiment(record: Dict[str, Any]) -> str:
    """실험 1건 기록. ID 반환."""
    _ensure_dirs()

    rec = {
        "id": record.get("id") or uuid.uuid4().hex[:12],
        "ts": record.get("ts") or datetime.now().isoformat(timespec="seconds"),
        **record,
    }

    # JSONL (전체 데이터)
    with JSONL_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # CSV (요약)
    write_header = not CSV_FILE.exists()
    with CSV_FILE.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        report = (rec.get("report") or "")[:120].replace("\n", " ")
        metrics = rec.get("metrics") or {}

        # retriever 정보를 units 배열에서 추출 (config 구조: {"units": [...], "reranker": {...}})
        retriever_cfg = rec.get("retriever") or {}
        active_units = [
            u for u in retriever_cfg.get("units", [])
            if u.get("active") and u.get("type") not in (None, "", "미설정")
        ]
        retriever_type = "+".join(u.get("type", "?") for u in active_units) if active_units else ""
        k_val = str(active_units[0].get("k", "")) if active_units else ""

        w.writerow(
            {
                "id": rec["id"],
                "ts": rec["ts"],
                "question": (rec.get("question") or "")[:200],
                "retriever_type": retriever_type,
                "k": k_val,
                "report_excerpt": report,
                "faithfulness": metrics.get("faithfulness", ""),
                "answer_relevance": metrics.get("answer_relevance", ""),
                "context_precision": metrics.get("context_precision", ""),
                "context_recall": metrics.get("context_recall", ""),
            }
        )

    return rec["id"]


def load_logs() -> List[Dict[str, Any]]:
    _ensure_dirs()
    if not JSONL_FILE.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with JSONL_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def clear_logs() -> None:
    _ensure_dirs()
    if JSONL_FILE.exists():
        JSONL_FILE.unlink()
    if CSV_FILE.exists():
        CSV_FILE.unlink()
