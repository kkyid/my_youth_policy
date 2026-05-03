def _normalize_korean_headings(text: str) -> str:
    """기존 텍스트에서 한국 정책 문서 특유의 계층 구조를 찾아 마크다운 헤더로 변환."""
    lines = text.split("\n")
    lines_pass1 = []

    # ── 추가: 주요 정책 키워드로 시작하는 줄을 헤더로 처리 ──
    POLICY_KEYWORDS = [
        "대출대상", "대출금리", "대출한도", "대출기간", "상환방법", "담보평가", "고객부담비용",
        "지원대상", "지원내용", "신청방법", "신청기한", "준비서류", "유의사항", "문의처",
        "지원 내용", "신청 자격", "대상자 선정", "신청 방법", "구비 서류"
    ]

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
        # 예) **가. 신청자격**
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
                # "대출대상 부부합산..." -> "### 대출대상\n부부합산..."
                content = stripped[len(kw):].strip()
                # ":" 또는 " "로 시작하는 경우 제거
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

        result.append(line)

    return "\n".join(result)
