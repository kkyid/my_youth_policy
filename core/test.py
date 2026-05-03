import pymupdf

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pdf_path = os.path.join(BASE_DIR, "data", "01. 2025년 서울시 청년월세지원 모집 공고문.pdf")
doc = pymupdf.open(pdf_path)
page = doc[0]
print(page.get_text())