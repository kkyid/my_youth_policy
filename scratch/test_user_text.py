import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import chunker

sample_text = """
내집마련디딤돌대출
주택도시기금의 개인상품 중 주택구입자금대출입니다.

※ 자세한 내용은 가까운 기금수탁은행을 방문하여 확인하실 수 있습니다.
정부지원 3대 서민 구입자금을 하나로 통합한 저금리의 구입자금대출
대출대상 부부합산 연소득 6천만원 이하(생애최초 주택구입자, 2자녀이상 가구는 연소득 7천만원, 신혼가구는 연소득 8.5천만원 이하), 순자산가액 5.11억원 이하 무주택 세대주 대출금리 연 2.85% ∼ 연 4.15% 대출한도 일반 2억원(생애최초 일반 2.4억원), 신혼가구 또는 2자녀 이상 가구
"""

def test_norm():
    print("Testing normalization...")
    md = chunker._normalize_korean_headings(sample_text)
    print("--- Result ---")
    print(md)
    print("--------------")

if __name__ == "__main__":
    test_norm()
