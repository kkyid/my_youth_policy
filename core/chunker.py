"""파일 → Markdown 변환 → 단계별 청킹 유틸.

지원 청커:
[1차 / 섹션 분할]
1. MarkdownSectionSplitter   - 헤더(#, ##, ###, ####) + 수평선(---, ***, ___) + 인용(>) 등 다양한 마크다운 구분자
2. PolicyFieldChunker         - 정책 도메인 키워드(자격, 지원 내용, 신청 기간 등) 기준 분할

[2차 / 세부 분할]
3. RecursiveCharacterTextSplitter
4. SemanticChunker            - 임베딩 변화점 기반
5. SentenceWindowChunker      - 문장 단위 슬라이딩 윈도우

추가 유틸:
- summarize_full_document     - LLM 으로 문서 전체 요약
- extract_chunk_metadata      - LLM 으로 청크별 정책 메타데이터 JSON 추출
- combine_chunk_payload       - 전체요약 + 메타데이터 요약 + 원본 결합
- apply_secondary_chunking    - 1차 결과 청크 리스트에 2차 청커 적용
- build_final_documents       - 메타데이터 추출 + 결합을 일괄 수행
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

try:
    from langchain_experimental.text_splitter import SemanticChunker
    _SEMANTIC_AVAILABLE = True
except Exception:
    SemanticChunker = None  # type: ignore
    _SEMANTIC_AVAILABLE = False


LLM_MODEL = "gpt-4o-mini"

METADATA_FIELDS = [
    # ── 정책 기본 ──────────────────────────────
    "category",          # "주택" | "금융"
    "title",             # 정책 공식 명칭
    "source_url",        # 공식 신청 URL
    "application_period",# 신청 기간 (텍스트)
    "support_type",      # 지원 내용·금액 요약 (텍스트)
    "housing_limit",     # 보증금/월세 상한 (텍스트)
    "is_repeat_allowed", # 재신청 가능 여부 (텍스트)
    # ── Self-Query 필터용 정규화 필드 ──────────
    "age_min",           # 최소 연령 (int, 없으면 0)
    "age_max",           # 최대 연령 (int, 없으면 99)
    "income_pct",        # 기준 중위소득 % 상한 (int, 없으면 0)
    "income_max",        # 연소득 상한 만원 (int, 없으면 0)
    "asset_max",         # 순자산 상한 만원 (int, 없으면 0)
    "household_type",    # "1인가구" | "신혼부부" | "한부모가족" | "청년" | "무관"
    "housing_type",      # "전세" | "월세" | "매입" | "임대주택" | "무관"
    "district",          # 서울 자치구 (예: "강남구"), 전체면 "서울특별시"
]

DEFAULT_POLICY_FIELDS = [
    "지원 대상", "지원 자격", "신청 자격", "자격 요건",
    "지원 내용", "지원 금액", "지원 한도", "지원 기간",
    "신청 방법", "신청 기간", "접수 기간",
    "선정 방법", "선정 기준",
    "제외 사항", "유의 사항",
    "문의처", "필요 서류", "사업 개요",
]

DEFAULT_EXTRA_SEPARATORS = ["---", "***", "___"]

# ── 구분자 프리셋 ──────────────────────────────────────────────────
SEPARATOR_PRESETS: Dict[str, List[str]] = {
    "한국어 최적화":   ["\n\n", "\n", "다. ", "요. ", "니다. ", "습니다. ", ". ", " ", ""],
    "기본 (영문)":     ["\n\n", "\n", ". ", " ", ""],
    "단락 우선":       ["\n\n\n", "\n\n", "\n", " ", ""],
    "문장 단위":       ["다. ", "요. ", "니다. ", "습니다. ", ". ", "! ", "? ", " ", ""],
}

# ── 청킹 설정 저장 경로 ────────────────────────────────────────────
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNKING_CONFIG_FILE = _DATA_DIR / "chunking_config.json"

DEFAULT_CHUNKING_CONFIG: Dict[str, Any] = {
    "stage1_method": "MarkdownSection (헤더 + 수평선 등)",
    "stage1_opts": {"levels": ["#", "##"], "extra_separators": []},
    "stage2_method": "건너뛰기",
    "stage2_opts": {
        "chunk_size": 800,
        "chunk_overlap": 100,
        "separator_preset": "한국어 최적화",
        "threshold_type": "percentile",
        "threshold_amount": 95.0,
        "embedding_model": "text-embedding-3-small",
        "window_size": 3,
        "stride": 1,
    },
    "llm_model": "gpt-4o-mini",
}


def load_chunking_config() -> Dict[str, Any]:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CHUNKING_CONFIG_FILE.exists():
        try:
            with CHUNKING_CONFIG_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return dict(DEFAULT_CHUNKING_CONFIG)


def save_chunking_config(cfg: Dict[str, Any]) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with CHUNKING_CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


# =============================================================================
# 1) 파일 -> 텍스트(Markdown)
# =============================================================================
def read_uploaded_file(file_obj, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in (".txt", ".md"):
        return file_obj.read().decode("utf-8", errors="ignore")
    if suffix == ".pdf":
        return _pdf_to_markdown(file_obj)
    raise ValueError(f"지원하지 않는 확장자: {suffix}")


def _pdf_to_markdown(file_obj) -> str:
    raw_bytes = file_obj.read()

    # ── 1순위: PyMuPDF 폰트 크기 기반 직접 추출 (OCR 불필요, 가장 정확) ──
    try:
        import pymupdf
        doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
        md = _fitz_to_markdown(doc)
        # 헤더(#)가 포함되어 있거나 최소한의 실질적 내용이 있을 때만 1순위 결과 채택
        if md and len(md.strip()) > 200 and ("#" in md or "공고" in md or "지원" in md):
            return _normalize_korean_headings(md)
    except Exception:
        pass

    # ── 2순위: pymupdf4llm (OCR 포함, 1순위 실패 시) ────────────────────
    try:
        import pymupdf4llm
        import pymupdf
        doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
        md = pymupdf4llm.to_markdown(doc)
        return _normalize_korean_headings(md)
    except Exception:
        pass

    # ── 3순위: pypdf 기본 텍스트 추출 ────────────────────────────────────
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw_bytes))
        chunks: List[str] = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            chunks.append(f"# Page {i}\n\n{text}")
        md = "\n\n".join(chunks)
        return _normalize_korean_headings(md)
    except Exception as e:
        raise RuntimeError(f"PDF 파싱 실패: {e}")


def _fitz_to_markdown(doc) -> str:
    """PyMuPDF word 좌표를 사용해 텍스트를 재조립하고 폰트 크기로 헤더를 감지합니다.

    이 PDF처럼 디자인 툴로 만들어 텍스트 요소가 쪼개진 경우에도
    같은 y 좌표의 단어들을 묶어 한 줄로 재조립하므로
    "1. 사업 개요" 같은 섹션 제목이 정확히 복원됩니다.
    """
    import statistics

    Y_TOLERANCE = 4.0  # 같은 줄로 볼 y 좌표 허용 오차(pt)

    # ── PASS 1: 전체 문서에서 폰트 크기 수집 ────────────────────────────
    all_sizes: List[float] = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():
                        all_sizes.append(span["size"])

    if not all_sizes:
        return ""

    body_size = statistics.median(all_sizes)
    h1_threshold = body_size * 2.0
    h2_threshold = body_size * 1.25   # 조금 더 낮춤 (1.35 -> 1.25)
    h3_threshold = body_size * 1.02   # 조금 더 낮춤 (1.05 -> 1.02)

    # ── PASS 2: 페이지별 단어(word) 좌표 기반 줄 재조립 ─────────────────
    # get_text("words", sort=True) → (x0,y0,x1,y1, text, block_no, line_no, word_no)
    # get_text("rawdict")로 폰트 크기도 함께 수집
    md_lines: List[str] = []

    for page in doc:
        # 단어별 좌표 (sort=True → y 오름차순 → x 오름차순)
        words = page.get_text("words", sort=True)

        # span 정보로 각 단어의 폰트 크기·볼드 여부 매핑 (bbox 겹침으로 근사)
        span_info: List[dict] = []
        for block in page.get_text("rawdict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    for char in span.get("chars", []):
                        span_info.append({
                            "x": char["origin"][0],
                            "y": char["origin"][1],
                            "size": span["size"],
                            "bold": bool(span.get("flags", 0) & (1 << 4))
                                    or "Bold" in span.get("font", ""),
                        })

        def get_span_attrs(wx: float, wy: float) -> Tuple[float, bool]:
            """단어 위치에 가장 가까운 span의 (size, is_bold) 반환."""
            best, best_dist = None, float("inf")
            for s in span_info:
                d = abs(s["x"] - wx) + abs(s["y"] - wy) * 2
                if d < best_dist:
                    best_dist = d
                    best = s
            if best:
                return best["size"], best["bold"]
            return body_size, False

        # y 좌표 기준으로 단어를 줄(row)로 묶기
        rows: List[List] = []  # [[y_center, [(x, text, size, bold), ...]], ...]
        for w in words:
            x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
            text = text.strip()
            if not text:
                continue
            y_center = (y0 + y1) / 2
            size, bold = get_span_attrs(x0, y_center)

            # 같은 y 좌표 줄이 있으면 거기에 추가
            matched = False
            for row in rows:
                if abs(row[0] - y_center) <= Y_TOLERANCE:
                    row[1].append((x0, text, size, bold))
                    matched = True
                    break
            if not matched:
                rows.append([y_center, [(x0, text, size, bold)]])

        # 각 줄 내에서 x 순으로 정렬 후 텍스트 합치기
        prev_y = None
        for row in sorted(rows, key=lambda r: r[0]):
            y_center = row[0]
            items = sorted(row[1], key=lambda i: i[0])  # x 정렬

            line_text = " ".join(i[1] for i in items).strip()
            if not line_text:
                continue

            max_size = max(i[2] for i in items)
            is_bold = any(i[3] for i in items)

            # 빈 줄 삽입 (줄 간격이 넓으면 단락 구분)
            if prev_y is not None and (y_center - prev_y) > body_size * 2.5:
                md_lines.append("")
            prev_y = y_center

            # "서 울 특 별 시 장" 같이 한 글자씩 공백으로 나뉜 장식 텍스트는
            # 폰트가 커도 섹션 헤더가 아니므로 일반 텍스트로 처리
            # (단, 20자 이상의 긴 문장은 제외)
            spaced_chars = len(line_text) < 20 and re.match(r"^[가-힣](?: [가-힣])+$", line_text)

            if max_size >= h1_threshold and not spaced_chars:
                md_lines.append(f"# {line_text}")
            elif max_size >= h2_threshold and not spaced_chars:
                md_lines.append(f"## {line_text}")
            elif is_bold and max_size >= h3_threshold and not spaced_chars:
                md_lines.append(f"### {line_text}")
            else:
                md_lines.append(line_text)

        md_lines.append("")  # 페이지 구분

    return "\n".join(md_lines)


def _normalize_korean_headings(text: str) -> str:
    """기존 텍스트에서 한국 정책 문서 특유의 계층 구조를 찾아 마크다운 헤더로 변환."""
    # ── 추가: 주요 정책 키워드로 시작하는 줄을 헤더로 처리 ──
    POLICY_KEYWORDS = [
        "대출대상", "대출금리", "대출한도", "대출기간", "상환방법", "담보평가", "고객부담비용",
        "지원대상", "지원내용", "신청방법", "신청기한", "준비서류", "유의사항", "문의처",
        "지원 내용", "신청 자격", "대상자 선정", "신청 방법", "구비 서류", "대출 대상"
    ]

    # ── PASS 0: 한 줄에 여러 키워드가 뭉쳐있는 경우 강제 줄바꿈 (웹 인쇄 PDF 대응) ──
    for kw in POLICY_KEYWORDS:
        # 단어 앞에 공백이 있고 뒤에 내용이 오는 패턴을 찾아 줄바꿈 삽입
        text = text.replace(f" {kw}", f"\n{kw}")
        text = text.replace(f"  {kw}", f"\n{kw}")

    lines = text.split("\n")
    lines_pass1 = []

    # ── PASS 1: OCR로 인해 한 줄에 붙어버린 섹션 번호 분리 ──────────────
    # 예) "## 기존내용 ․ 2. 신청 자격요건"  →  두 줄로 분리
    split_pattern = re.compile(
        r"\s[․·ㆍ・･·ㆍ]\s+(\d+[\.\)]\s*[가-힣][가-힣\w\s()·,\-]{1,30})\s*$"
    )
    for line in lines:
        m = split_pattern.search(line)
        if m and m.start() > 0:
            before = line[: m.start()].rstrip()
            header_part = m.group(1).strip()
            lines_pass1.append(before)
            lines_pass1.append(f"## {header_part}")
        else:
            lines_pass1.append(line)

    # ── PASS 2: 구체적인 정규식 기반 변환 ──────────────────────────────
    result = []
    for line in lines_pass1:
        stripped = line.strip()
        if not stripped:
            result.append("")
            continue

        # ── 1순위: 숫자. 제목 (대분류) ──
        # 예) **1. 사업개요** 또는 1. 사업개요 (단독 줄)
        m_big = re.match(r"^(?:(?:\*\*)|(?:))(\d+[\.\)]\s+.+?)(?:(?:\*\*)|(?::?))\s*$", stripped)
        if m_big:
            result.append(f"## {m_big.group(1).strip()}")
            continue

        # ── 2순위: 가. 제목 (중분류) ──
        m_mid = re.match(r"^(?:(?:\*\*)|(?:))([가-힣][\.\)]\s+.+?)(?:(?:\*\*)|(?::?))\s*$", stripped)
        if m_mid:
            result.append(f"#### {m_mid.group(1).strip()}")
            continue

        # ── 3순위: ① 항목 / (1) 항목 => (####) ──
        m_small = re.match(r"^(?:(?:\*\*)|(?:))([①-⑳\(\d+\)]\s*.+?)(?:(?:\*\*)|(?::?))\s*$", stripped)
        if m_small:
            result.append(f"#### {m_small.group(1).strip()}")
            continue

        # ── 4순위: 정책 키워드 매칭 (예: 대출대상) ──
        matched_kw = False
        for kw in POLICY_KEYWORDS:
            if stripped.startswith(kw) and len(stripped) < 100:
                content = stripped[len(kw):].strip()
                if content.startswith(":") or content.startswith(" "):
                    content = content[1:].strip()
                
                if content:
                    result.append(f"### {kw}")
                    result.append(content)
                else:
                    result.append(f"### {kw}")
                matched_kw = True
                break
        if matched_kw:
            continue

        # ── 특수 기호 불렛 ──
        if stripped.startswith("- ") or stripped.startswith("* ") or stripped.startswith("• "):
            result.append(line)
            continue
        
        # ── (추가) 문장 중간의 기호들을 마크다운 불렛으로 변환 ──
        # 예) "※ 내용" 또는 "○ 내용" 또는 " - 내용"
        if stripped.startswith("※") or stripped.startswith("○") or stripped.startswith("ㅇ") or stripped.startswith("-"):
            clean_s = stripped.lstrip("※○ㅇ- ").strip()
            result.append(f"- {clean_s}")
            continue

        result.append(line)

    return "\n".join(result)


def _strip_markdown_code_blocks(text: str) -> str:
    """LLM이 응답 앞뒤에 붙인 ```markdown ... ``` 기호를 제거합니다."""
    text = text.strip()
    # 시작 부분의 ```markdown 또는 ``` 제거 (대소문자 무시)
    text = re.sub(r"^```(?:markdown)?\n?", "", text, flags=re.IGNORECASE)
    # 끝 부분의 ``` 제거
    text = re.sub(r"\n?```$", "", text, flags=re.IGNORECASE)
    return text.strip()


def refine_markdown_with_llm(text: str, model: str = "gpt-4o-mini") -> str:
    """LLM을 사용하여 텍스트를 구조화된 마크다운으로 재작성합니다."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_template("""\
아래 텍스트는 PDF에서 추출된 가공되지 않은 문장들입니다.
이를 읽기 좋은 마크다운(Markdown) 형식으로 재구성해 주세요.

[지침]
1. 정책명은 # 헤더, 주요 섹션(대상, 금리, 한도 등)은 ## 헤더를 반드시 사용하세요.
2. 섹션 제목에 **굵게** 표시만 하지 말고, 반드시 ## 기호를 줄 시작에 붙여야 합니다.
3. 세부 항목은 불렛 포인트(- )를 사용해 정렬하세요.
4. 표 형태의 데이터는 가급적 마크다운 테이블이나 리스트로 깔끔하게 정리하세요.
5. 출력물에 ```markdown 같은 코드 블록 기호를 절대 붙이지 말고 순수 마크다운 내용만 출력하세요.

[텍스트]
{text}

[마크다운 출력]
""")
    llm = ChatOpenAI(model=model, temperature=0.1)
    chain = prompt | llm | StrOutputParser()
    
    input_text = text[:8000]
    raw_output = chain.invoke({"text": input_text})
    return _strip_markdown_code_blocks(raw_output)


def stream_refine_markdown_with_llm(text: str, model: str = "gpt-4o-mini"):
    """LLM을 사용하여 텍스트를 마크다운으로 정제하며 결과를 스트리밍합니다."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_template("""\
아래 텍스트는 PDF에서 추출된 가공되지 않은 문장들입니다.
이를 읽기 좋은 마크다운(Markdown) 형식으로 재구성해 주세요.

[지침]
1. 정책명은 # 헤더, 주요 섹션(대상, 금리, 한도 등)은 ## 헤더를 반드시 사용하세요.
2. 섹션 제목에 **굵게** 표시만 하지 말고, 반드시 ## 기호를 줄 시작에 붙여야 합니다.
3. 세부 항목은 불렛 포인트(- )를 사용해 정렬하세요.
4. 표 형태의 데이터는 가급적 마크다운 테이블이나 리스트로 깔끔하게 정리하세요.
5. 출력물에 ```markdown 같은 코드 블록 기호를 절대 붙이지 말고 순수 마크다운 내용만 출력하세요.

[텍스트]
{text}

[마크다운 출력]
""")
    llm = ChatOpenAI(model=model, temperature=0.1, streaming=True)
    chain = prompt | llm | StrOutputParser()
    
    input_text = text[:8000]
    yield from chain.stream({"text": input_text})



# =============================================================================
# 2) LLM 전체 요약
# =============================================================================
SUMMARY_PROMPT = """\
당신은 서울시 청년/신혼부부 주택·금융 정책 분석가입니다.
아래 문서 전체를 살펴 4~6문장으로 한국어 요약을 작성하세요.

요약에는 반드시 다음 정보를 포함하세요:
- 정책의 핵심 목적
- 주요 지원 대상 (연령/소득/가구)
- 핵심 혜택 및 금액·기간
- 신청 시 주의사항이나 제외 조건

[문서]
{text}

[전체 요약 - 한국어, 평이한 서술체]
"""


def summarize_full_document(text: str, model: str = LLM_MODEL) -> str:
    """문서 전체를 LLM 으로 요약. 너무 긴 경우 앞부분 + 뒷부분 발췌."""
    from langchain_openai import ChatOpenAI

    if len(text) > 14000:
        head = text[:9000]
        tail = text[-4000:]
        truncated = head + "\n\n[...중략...]\n\n" + tail
    else:
        truncated = text

    llm = ChatOpenAI(model=model, temperature=0)
    chain = ChatPromptTemplate.from_template(SUMMARY_PROMPT) | llm | StrOutputParser()
    return chain.invoke({"text": truncated}).strip()


# =============================================================================
# 3) 1차 청킹 - 섹션 분할
# =============================================================================
def chunk_with_markdown_header(
    text: str,
    headers_to_split_on: Optional[List[Tuple[str, str]]] = None,
    extra_metadata: Optional[dict] = None,
) -> List[Document]:
    """기본 MarkdownHeaderTextSplitter (호환 유지용)."""
    headers_to_split_on = headers_to_split_on or [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    docs = splitter.split_text(text)
    if extra_metadata:
        for d in docs:
            d.metadata.update(extra_metadata)
    return docs


def chunk_with_markdown_section(
    text: str,
    headers: Optional[List[str]] = None,
    extra_separators: Optional[List[str]] = None,
    split_blockquote: bool = False,
    extra_metadata: Optional[dict] = None,
) -> List[Document]:
    """헤더 + 추가 마크다운 구분자(수평선 등) 기반 섹션 분할."""
    headers = headers or ["#", "##", "###", "####"]
    extra_separators = extra_separators if extra_separators is not None else DEFAULT_EXTRA_SEPARATORS

    work = text
    for i, sep in enumerate(extra_separators):
        pattern = r"(?m)^[\t ]*" + re.escape(sep) + r"[\t ]{0,}$"
        token = f"## __SEP_{i}__"
        work = re.sub(pattern, token, work)

    if split_blockquote:
        work = re.sub(r"(?m)^>\s+(?=\S)", "### __QUOTE__\n> ", work)

    headers_pairs: List[Tuple[str, str]] = []
    for h in headers:
        level = len(h)
        headers_pairs.append((h, f"h{level}"))
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_pairs)
    docs = splitter.split_text(work)

    for d in docs:
        for k in list(d.metadata.keys()):
            v = d.metadata[k]
            if isinstance(v, str) and v.startswith("__SEP_"):
                d.metadata[k] = "(horizontal-rule)"
            if isinstance(v, str) and v.startswith("__QUOTE__"):
                d.metadata[k] = "(blockquote)"
        if extra_metadata:
            d.metadata.update(extra_metadata)
    return docs


def chunk_with_policy_field(
    text: str,
    field_keywords: Optional[List[str]] = None,
    min_section_chars: int = 30,
    extra_metadata: Optional[dict] = None,
) -> List[Document]:
    """정책 도메인 키워드 기반 분할."""
    field_keywords = field_keywords or DEFAULT_POLICY_FIELDS
    sorted_keys = sorted(field_keywords, key=len, reverse=True)
    pattern = "(" + "|".join(re.escape(k) for k in sorted_keys) + ")"
    line_pattern = r"(?m)^(?:#{1,6}\s*|[0-9]+[.)]\s*|[-*]\s*)?\b" + pattern + r"\b"

    matches = list(re.finditer(line_pattern, text))
    docs: List[Document] = []
    if not matches:
        docs.append(
            Document(
                page_content=text.strip(),
                metadata={**(extra_metadata or {}), "field": "(전체)"},
            )
        )
        return docs

    spans: List[Tuple[int, int, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        field = m.group(1)
        spans.append((start, end, field))

    if matches[0].start() > min_section_chars:
        intro = text[: matches[0].start()].strip()
        if intro:
            docs.append(
                Document(
                    page_content=intro,
                    metadata={**(extra_metadata or {}), "field": "(개요)"},
                )
            )

    for start, end, field in spans:
        body = text[start:end].strip()
        if len(body) < min_section_chars:
            continue
        docs.append(
            Document(
                page_content=body,
                metadata={**(extra_metadata or {}), "field": field},
            )
        )
    return docs


# =============================================================================
# 4) 2차 청킹 - 세부 분할
# =============================================================================
def chunk_with_recursive(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    separators: Optional[List[str]] = None,
    extra_metadata: Optional[dict] = None,
) -> List[Document]:
    # chunk_overlap 은 chunk_size 의 절반을 초과할 수 없음
    safe_overlap = min(chunk_overlap, chunk_size // 2)
    seps = separators or SEPARATOR_PRESETS["한국어 최적화"]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=safe_overlap,
        separators=seps,
    )
    pieces = splitter.split_text(text)
    return [Document(page_content=p, metadata=dict(extra_metadata or {})) for p in pieces]


def chunk_with_semantic(
    text: str,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float = 95.0,
    embedding_model: str = "text-embedding-3-small",
    extra_metadata: Optional[dict] = None,
) -> List[Document]:
    if not _SEMANTIC_AVAILABLE:
        raise RuntimeError("SemanticChunker 미설치. langchain-experimental 설치 필요.")
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model=embedding_model)
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )
    return splitter.create_documents([text], metadatas=[extra_metadata or {}])


def chunk_with_sentence_window(
    text: str,
    window_size: int = 3,
    stride: int = 1,
    extra_metadata: Optional[dict] = None,
) -> List[Document]:
    """문장 슬라이딩 윈도우. 한국어/영어 종결 패턴 모두 처리."""
    sentences = re.split(
        r"(?<=[\.!?])\s+|(?<=다\.)\s+|(?<=요\.)\s+|(?<=죠\.)\s+|(?<=까\?)\s+",
        text,
    )
    sentences = [s.strip() for s in sentences if s and s.strip()]
    if not sentences:
        return []

    docs: List[Document] = []
    if len(sentences) <= window_size:
        docs.append(
            Document(
                page_content=" ".join(sentences),
                metadata={
                    **(extra_metadata or {}),
                    "window_start": 0,
                    "window_end": len(sentences),
                },
            )
        )
        return docs

    for i in range(0, len(sentences) - window_size + 1, max(1, stride)):
        window = sentences[i : i + window_size]
        docs.append(
            Document(
                page_content=" ".join(window),
                metadata={
                    **(extra_metadata or {}),
                    "window_start": i,
                    "window_end": i + window_size,
                },
            )
        )
    return docs


def is_semantic_available() -> bool:
    return _SEMANTIC_AVAILABLE


def merge_short_chunks(
    docs: List[Document],
    min_chars: int = 50,
) -> List[Document]:
    """짧은 청크를 인접 청크에 병합한다.

    처리 순서:
    1. 빈 청크 제거
    2. 첫 청크가 짧으면 → 다음 청크 앞에 붙임 (앞이 없으므로)
    3. 이후 청크가 min_chars 미만이면 → 바로 앞 청크 뒤에 붙임
    4. 메타데이터는 수신 청크(병합 대상) 기준으로 유지하되, 
       병합되는 청크의 헤더 정보가 다르면 텍스트에 삽입한다.
    """
    if not docs or min_chars <= 0:
        return docs

    def _get_header_str(meta: dict) -> str:
        parts = [meta.get(k) for k in ["h1", "h2", "h3", "h4"] if meta.get(k)]
        return " > ".join(parts) if parts else ""

    # 1) 빈 청크 제거
    cleaned = [
        Document(page_content=d.page_content.strip(), metadata=dict(d.metadata))
        for d in docs
        if d.page_content.strip()
    ]
    if not cleaned:
        return []

    # 2) 첫 청크가 짧으면 다음 청크 앞에 선행 병합
    if len(cleaned) >= 2 and len(cleaned[0].page_content) < min_chars:
        first = cleaned.pop(0)
        second = cleaned[0]
        
        h_first = _get_header_str(first.metadata)
        h_second = _get_header_str(second.metadata)
        
        if h_first and h_first != h_second:
            new_content = (
                f"[SECTION: {h_first}]\n{first.page_content}\n\n"
                f"---\n"
                f"[SECTION: {h_second}]\n{second.page_content}"
            )
        else:
            new_content = first.page_content + "\n\n" + second.page_content

        cleaned[0] = Document(
            page_content=new_content,
            metadata=second.metadata,
        )

    # 3) 나머지 순회 — 짧으면 앞 청크에 후행 병합
    result: List[Document] = []
    for doc in cleaned:
        text = doc.page_content
        if result and len(text) < min_chars:
            prev = result[-1]
            
            h_prev = _get_header_str(prev.metadata)
            h_curr = _get_header_str(doc.metadata)
            
            # 헤더 정보가 다르면 양쪽 모두에 이름표를 달아줌 (기존에 없다면)
            if h_curr and h_curr != h_prev:
                # prev 본문에 아직 이름표가 없다면 추가
                p_text = prev.page_content
                if not p_text.startswith("[SECTION:"):
                    p_text = f"[SECTION: {h_prev}]\n{p_text}"
                
                content_to_add = f"\n\n---\n[SECTION: {h_curr}]\n{text}"
                result[-1] = Document(
                    page_content=p_text.rstrip() + content_to_add,
                    metadata=prev.metadata,
                )
            else:
                result[-1] = Document(
                    page_content=prev.page_content.rstrip() + "\n\n" + text,
                    metadata=prev.metadata,
                )
        else:
            result.append(doc)

    return result


# =============================================================================
# 5) 1차 -> 2차 청킹 적용
# =============================================================================
def apply_secondary_chunking(
    primary_docs: List[Document],
    method: str,
    opts: Dict[str, Any],
) -> List[Document]:
    """method: 'recursive' | 'semantic' | 'sentence_window'
    opts 에 'min_merge' 키가 있으면 2차 청킹 결과에도 짧은 청크 병합을 적용한다.
    """
    out: List[Document] = []
    for d in primary_docs:
        carry_meta = dict(d.metadata or {})
        text = d.page_content
        if not text or not text.strip():
            continue

        if method == "recursive":
            preset_name = opts.get("separator_preset", "한국어 최적화")
            seps = SEPARATOR_PRESETS.get(preset_name, SEPARATOR_PRESETS["한국어 최적화"])
            sub = chunk_with_recursive(
                text,
                chunk_size=int(opts.get("chunk_size", 800)),
                chunk_overlap=int(opts.get("chunk_overlap", 100)),
                separators=seps,
                extra_metadata=carry_meta,
            )
        elif method == "semantic":
            sub = chunk_with_semantic(
                text,
                breakpoint_threshold_type=str(opts.get("threshold_type", "percentile")),
                breakpoint_threshold_amount=float(opts.get("threshold_amount", 95.0)),
                embedding_model=str(opts.get("embedding_model", "text-embedding-3-small")),
                extra_metadata=carry_meta,
            )
        elif method == "sentence_window":
            sub = chunk_with_sentence_window(
                text,
                window_size=int(opts.get("window_size", 3)),
                stride=int(opts.get("stride", 1)),
                extra_metadata=carry_meta,
            )
        else:
            sub = [d]

        out.extend(sub)

    min_merge = int(opts.get("min_merge", 0))
    if min_merge > 0:
        out = merge_short_chunks(out, min_chars=min_merge)

    return out


# =============================================================================
# 6) LLM 메타데이터 추출 + 결합
# =============================================================================
METADATA_PROMPT = """\
당신은 서울시 청년/신혼부부 정책 데이터 추출 전문가입니다.
아래 정책 청크에서 메타데이터를 JSON 으로 **정확하고 정규화하여** 추출하세요.

[공통 규칙]
- 청크에 명시된 정보만 사용하세요. 추측·창작 금지.
- 모든 키를 반드시 포함해야 합니다.
- 숫자 필드(age_min/age_max/income_pct/income_max/asset_max)는 반드시 정수(int)로 출력하세요.
- 범주형 필드는 정해진 값 중 하나만 사용하세요.

[필드별 추출 규칙]

▶ 기본 정보 (텍스트)
- title: 정책의 공식 명칭 (예: "2025년 서울시 청년월세지원")
- source_url: 공식 신청 URL (없으면 "")
- application_period: 신청 기간 텍스트 (예: "2025.06.11~06.24", 없으면 "")
- support_type: 지원 내용·금액 요약 (예: "월 최대 20만원, 최대 12개월", 없으면 "")
- housing_limit: 보증금/월세 상한 텍스트 (예: "보증금 8천만원 이하 / 월세 60만원 이하", 없으면 "")
- is_repeat_allowed: 재신청 가능 여부 (예: "생애 1회", "기수혜자 불가", 없으면 "")

▶ 연령 (int)
- age_min: 지원 가능 최소 연령. "만 19세 이상" → 19. 없으면 0.
- age_max: 지원 가능 최대 연령. "만 39세 이하" → 39. 없으면 99.
  ※ "청년(19~39세)" 표현이면 age_min=19, age_max=39

▶ 소득·자산 (int)
- income_pct: 기준 중위소득 상한 %. "중위소득 150% 이하" → 150. 없으면 0.
- income_max: 연소득 상한 (단위: 만원). "연소득 5천만원 이하" → 5000. 없으면 0.
  ※ 중위소득 % 기준이면 income_max=0
- asset_max: 순자산 상한 (단위: 만원). "순자산 3억 4500만원 이하" → 34500. 없으면 0.

▶ 가구 유형 (범주형)
- household_type: 다음 중 하나만 선택.
  "1인가구" | "신혼부부" | "한부모가족" | "청년" | "무관"
  ※ 명시 없거나 제한 없으면 "무관"
  ※ 청년 1인가구처럼 중복이면 "청년" 우선

▶ 주거 유형 (범주형)
- housing_type: 다음 중 하나만 선택.
  "전세" | "월세" | "매입" | "임대주택" | "무관"
  ※ 전·월세 모두면 "월세" (더 제한적인 것)
  ※ 명시 없으면 "무관"

▶ 지역 (텍스트)
- district: 특정 자치구 지정이면 그 구명 (예: "강남구"). 서울 전체면 "서울특별시".

[정책 청크]
{chunk}

[출력 - JSON 만, 다른 텍스트 금지]
{{
  "title": "",
  "source_url": "",
  "application_period": "",
  "support_type": "",
  "housing_limit": "",
  "is_repeat_allowed": "",
  "age_min": 0,
  "age_max": 99,
  "income_pct": 0,
  "income_max": 0,
  "asset_max": 0,
  "household_type": "무관",
  "housing_type": "무관",
  "district": "서울특별시"
}}
"""

def _extract_json_obj(text: str) -> Optional[dict]:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def extract_chunk_metadata(
    chunk_text: str,
    category: str,
    model: str = LLM_MODEL,
) -> Dict[str, str]:
    """청크 1개에서 정책 메타데이터 JSON 추출. category 강제 주입."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, temperature=0)
    chain = ChatPromptTemplate.from_template(METADATA_PROMPT) | llm | StrOutputParser()
    raw = chain.invoke({"chunk": (chunk_text or "")[:6000]})
    parsed = _extract_json_obj(raw) or {}

    INT_FIELDS = {"age_min", "age_max", "income_pct", "income_max", "asset_max"}
    DEFAULTS   = {"age_min": 0, "age_max": 99, "income_pct": 0,
                  "income_max": 0, "asset_max": 0,
                  "household_type": "무관", "housing_type": "무관",
                  "district": "서울특별시"}

    out: Dict[str, Any] = {"category": category}
    for k in METADATA_FIELDS:
        if k == "category":
            continue
        v = parsed.get(k, DEFAULTS.get(k, ""))
        if k in INT_FIELDS:
            try:
                out[k] = int(v)
            except (TypeError, ValueError):
                out[k] = DEFAULTS[k]
        else:
            out[k] = DEFAULTS.get(k, "") if v is None else str(v)
    return out


def combine_chunk_payload(
    chunk_text: str,
    full_summary: str,
    metadata: Dict[str, str],
) -> str:
    """[문서 전체 요약] + [메타데이터 요약] + [원본 청크] 결합."""
    label_map = {
        "category":          "분류",
        "title":             "정책명",
        "source_url":        "공식 URL",
        "application_period":"신청 기간",
        "support_type":      "지원 내용",
        "housing_limit":     "주거 한도",
        "is_repeat_allowed": "재신청 여부",
        "age_min":           "최소 연령",
        "age_max":           "최대 연령",
        "income_pct":        "중위소득 % 상한",
        "income_max":        "연소득 상한(만원)",
        "asset_max":         "순자산 상한(만원)",
        "household_type":    "가구 유형",
        "housing_type":      "주거 유형",
        "district":          "지역구",
    }
    meta_lines = []
    for k in METADATA_FIELDS:
        v = (metadata or {}).get(k, "")
        if not v:
            continue
        meta_lines.append(f"- {label_map.get(k, k)}: {v}")
    meta_block = "\n".join(meta_lines) if meta_lines else "- (추출된 메타데이터 없음)"

    parts = [
        "[문서 전체 요약]",
        (full_summary or "(요약 없음)").strip(),
        "",
        "[메타데이터]",
        meta_block,
        "",
        "[원본 청크]",
        (chunk_text or "").strip(),
    ]
    return "\n".join(parts)


def build_final_documents(
    chunks: List[Document],
    full_summary: str,
    category: str,
    extra_metadata: Optional[dict] = None,
    model: str = LLM_MODEL,
    progress_cb=None,
) -> List[Document]:
    """청크들에 대해 메타데이터 추출 + 결합을 모두 수행."""
    out: List[Document] = []
    n = len(chunks)
    for i, d in enumerate(chunks):
        meta_extracted = extract_chunk_metadata(
            d.page_content,
            category=category,
            model=model,
        )
        merged_meta: Dict[str, Any] = {**(extra_metadata or {}), **(d.metadata or {}), **meta_extracted}
        # ChromaDB 메타값은 str/int/float/bool 만 허용
        for k, v in list(merged_meta.items()):
            if isinstance(v, (dict, list)):
                merged_meta[k] = json.dumps(v, ensure_ascii=False)
            elif v is None:
                merged_meta[k] = ""
            elif not isinstance(v, (str, int, float, bool)):
                merged_meta[k] = str(v)

        page = combine_chunk_payload(d.page_content, full_summary, meta_extracted)
        out.append(Document(page_content=page, metadata=merged_meta))

        if progress_cb is not None:
            try:
                progress_cb(i + 1, n)
            except Exception:
                pass
    return out
