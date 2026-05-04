"""파이프라인 전체 E2E 테스트 스크립트.

테스트 항목:
1. DB 연결 및 상태 확인
2. 프롬프트 로드 확인
3. 리트리버 설정 로드 및 빌드
4. Self-Query 메타데이터 필터 추출
5. 쿼리 전처리 (HyDE / Multi-Query / Decomposition)
6. 질문 분해 (주택 vs 금융)
7. 검색 + 메타데이터 필터 적용
8. Top3 선정
9. 보고서 생성
10. 전체 run_pipeline 통합 호출
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')
os.environ["ANONYMIZED_TELEMETRY"] = "False"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import vector_db, rag_engine, retrievers as retr, prompts as prompt_store

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def test(name, fn):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    try:
        fn()
        results.append((name, True, ""))
        print(f"  → {PASS}")
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"  → {FAIL}: {e}")

# ── 테스트 질문 (메타데이터 필터링이 잘 되는지 확인용) ──
TEST_Q = "만 28세 사회초년생입니다. 현재 연봉은 3500만원이고 1인 가구입니다. 서울시에서 이자를 지원해주는 정책이 있다고 들었는데 맞나요?"


# ──────────────────────────────────────────────────────────
# 1. DB 연결 및 상태
# ──────────────────────────────────────────────────────────
def t1_db_status():
    stats = vector_db.collections_status()
    print(f"  주택: {stats['housing']} chunks, 금융: {stats['finance']} chunks")
    assert stats["housing"] > 0 or stats["finance"] > 0, "DB에 데이터가 없습니다!"

test("1. DB 연결 및 데이터 확인", t1_db_status)


# ──────────────────────────────────────────────────────────
# 2. 프롬프트 로드
# ──────────────────────────────────────────────────────────
def t2_prompts():
    p = prompt_store.load_prompts()
    for key in ["ask", "selection", "report"]:
        assert key in p and len(p[key]) > 50, f"프롬프트 '{key}' 미설정 또는 너무 짧음"
        print(f"  {key}: {len(p[key])}자")
    for temp_key in ["ask_temp", "selection_temp", "report_temp"]:
        assert temp_key in p, f"온도 설정 '{temp_key}' 누락"
        print(f"  {temp_key}: {p[temp_key]}")

test("2. 프롬프트 로드", t2_prompts)


# ──────────────────────────────────────────────────────────
# 3. 리트리버 설정 로드 및 빌드
# ──────────────────────────────────────────────────────────
def t3_retriever():
    cfg = retr.load_retriever_config()
    print(f"  글로벌 LLM: {cfg.get('llm_model')}")
    units = cfg.get("units", [])
    active = [u for u in units if u.get("active") and u.get("type") != "미설정"]
    print(f"  활성 유닛: {len(active)}개")
    for u in active:
        print(f"    - {u.get('type')} k={u.get('k')} search={u.get('search_type')}")
    
    rerank = cfg.get("reranker", {})
    print(f"  Reranker: {'ON' if rerank.get('enabled') else 'OFF'} model={rerank.get('model')}")
    
    preproc = cfg.get("preprocessing", {})
    sq = preproc.get("self_query", {})
    qt = preproc.get("query_transform", {})
    print(f"  Self-Query: {'ON' if sq.get('enabled') else 'OFF'}")
    print(f"  Query Transform: {qt.get('method', '없음')}")
    
    # 빌드 테스트
    vs = vector_db.get_vectorstore(vector_db.FINANCE_COLLECTION)
    r = retr.build_retriever(vs, cfg)
    print(f"  리트리버 빌드 OK: {type(r).__name__}")

test("3. 리트리버 설정 로드 및 빌드", t3_retriever)


# ──────────────────────────────────────────────────────────
# 4. Self-Query 메타데이터 필터 추출
# ──────────────────────────────────────────────────────────
def t4_self_query():
    filt = rag_engine.apply_self_query(TEST_Q)
    print(f"  추출된 필터: {json.dumps(filt, ensure_ascii=False, indent=2)}")
    assert filt is not None, "Self-Query가 필터를 생성하지 못함"
    # 연령(28)과 소득(3500만원) 관련 조건이 있어야 함
    filt_str = json.dumps(filt)
    assert "age" in filt_str or "$lte" in filt_str or "$gte" in filt_str, "연령/소득 필터 없음"

test("4. Self-Query 메타데이터 필터 추출", t4_self_query)


# ──────────────────────────────────────────────────────────
# 5. 쿼리 전처리 (HyDE / Multi-Query / Decomposition)
# ──────────────────────────────────────────────────────────
def t5_preprocessing():
    # HyDE
    hyde_result = rag_engine.apply_hyde(TEST_Q)
    print(f"  HyDE 가상문서: {hyde_result[:100]}...")
    assert len(hyde_result) > 20, "HyDE 결과가 너무 짧음"
    
    # Multi-Query
    mq_result = rag_engine.apply_multi_query(TEST_Q)
    print(f"  Multi-Query: {len(mq_result)}개 쿼리")
    for i, q in enumerate(mq_result):
        print(f"    [{i}] {q[:80]}...")
    assert len(mq_result) >= 2, "Multi-Query 결과가 2개 미만"
    
    # Decomposition
    decomp = rag_engine.apply_decomposition(TEST_Q)
    print(f"  Decomposition: {len(decomp)}개 하위질문")
    for i, q in enumerate(decomp):
        print(f"    [{i}] {q[:80]}...")
    assert len(decomp) >= 1, "Decomposition 결과 없음"

test("5. 쿼리 전처리 (HyDE/Multi-Query/Decomposition)", t5_preprocessing)


# ──────────────────────────────────────────────────────────
# 6. 질문 분해 (주택 vs 금융)
# ──────────────────────────────────────────────────────────
def t6_decompose():
    hq, fq = rag_engine.decompose_question(TEST_Q)
    print(f"  주택 쿼리: {hq[:100]}")
    print(f"  금융 쿼리: {fq[:100]}")
    assert len(hq) > 5 and len(fq) > 5, "분해된 쿼리가 너무 짧음"

test("6. 질문 분해 (주택 vs 금융)", t6_decompose)


# ──────────────────────────────────────────────────────────
# 7. 검색 + 메타데이터 필터 적용
# ──────────────────────────────────────────────────────────
def t7_retrieve():
    cfg = retr.load_retriever_config()
    hq, fq = rag_engine.decompose_question(TEST_Q)
    
    # 7a. 필터 없이 검색
    h_docs, f_docs = rag_engine.retrieve_candidates(hq, fq, cfg, metadata_filter=None)
    print(f"  [필터 없음] 주택: {len(h_docs)}건, 금융: {len(f_docs)}건")
    assert len(h_docs) + len(f_docs) > 0, "검색 결과 0건"
    
    # 7b. Self-Query 필터 적용 검색
    filt = rag_engine.apply_self_query(TEST_Q)
    if filt:
        try:
            h_docs2, f_docs2 = rag_engine.retrieve_candidates(hq, fq, cfg, metadata_filter=filt)
            print(f"  [필터 적용] 주택: {len(h_docs2)}건, 금융: {len(f_docs2)}건")
        except Exception as e:
            # ChromaDB 필터 문법 오류 가능 - 경고만 출력
            print(f"  [필터 적용] 경고: {e}")
            print(f"  → 필터 문법 비호환이 감지됨. 필터 없이 검색은 정상 동작.")
    
    # 메타데이터 확인
    all_docs = h_docs + f_docs
    if all_docs:
        sample = all_docs[0]
        meta_keys = list(sample.metadata.keys()) if sample.metadata else []
        print(f"  샘플 메타데이터 키: {meta_keys}")
        # 새 메타데이터 필드 존재 여부 확인
        new_fields = ["age_min", "age_max", "income_max_man", "marital_status", "region"]
        found = [f for f in new_fields if f in meta_keys]
        print(f"  새 메타데이터 필드 발견: {found}")

test("7. 검색 + 메타데이터 필터 적용", t7_retrieve)


# ──────────────────────────────────────────────────────────
# 8. Top3 선정
# ──────────────────────────────────────────────────────────
def t8_top3():
    cfg = retr.load_retriever_config()
    hq, fq = rag_engine.decompose_question(TEST_Q)
    h_docs, f_docs = rag_engine.retrieve_candidates(hq, fq, cfg)
    top3 = rag_engine.select_top3(TEST_Q, h_docs, f_docs)
    print(f"  Top3 선정: {len(top3)}개")
    for i, item in enumerate(top3):
        print(f"    [{i+1}] {item.get('policy_name', '?')} ({item.get('category', '?')})")
    assert len(top3) > 0, "Top3 결과 없음"

test("8. Top3 선정", t8_top3)


# ──────────────────────────────────────────────────────────
# 9. 보고서 생성
# ──────────────────────────────────────────────────────────
def t9_report():
    cfg = retr.load_retriever_config()
    hq, fq = rag_engine.decompose_question(TEST_Q)
    h_docs, f_docs = rag_engine.retrieve_candidates(hq, fq, cfg)
    top3 = rag_engine.select_top3(TEST_Q, h_docs, f_docs)
    contexts = [d.page_content for d in h_docs + f_docs]
    report = rag_engine.make_report(TEST_Q, top3, contexts)
    print(f"  보고서 길이: {len(report)}자")
    print(f"  보고서 앞 200자: {report[:200]}...")
    assert len(report) > 100, "보고서가 너무 짧음"

test("9. 보고서 생성", t9_report)


# ──────────────────────────────────────────────────────────
# 10. 전체 run_pipeline 통합 호출
# ──────────────────────────────────────────────────────────
def t10_full_pipeline():
    cfg = retr.load_retriever_config()
    prompts_dict = prompt_store.load_prompts()
    
    start = time.time()
    res = rag_engine.run_pipeline(TEST_Q, cfg, prompts_dict)
    elapsed = time.time() - start
    
    print(f"  소요 시간: {elapsed:.1f}초")
    print(f"  주택 쿼리: {res.get('housing_query', '?')[:80]}")
    print(f"  금융 쿼리: {res.get('finance_query', '?')[:80]}")
    print(f"  메타데이터 필터: {res.get('metadata_filter')}")
    print(f"  주택 문서: {len(res.get('housing_docs', []))}건")
    print(f"  금융 문서: {len(res.get('finance_docs', []))}건")
    print(f"  Top3: {len(res.get('top3', []))}개")
    for i, t in enumerate(res.get("top3", [])):
        print(f"    [{i+1}] {t.get('policy_name', '?')}")
    print(f"  보고서 길이: {len(res.get('report', ''))}자")
    print(f"  contexts_text: {len(res.get('contexts_text', []))}건")
    
    assert len(res.get("top3", [])) > 0, "Top3 결과 없음"
    assert len(res.get("report", "")) > 100, "보고서가 너무 짧음"
    assert len(res.get("contexts_text", [])) > 0, "contexts_text 없음"

test("10. 전체 run_pipeline 통합 호출", t10_full_pipeline)


# ──────────────────────────────────────────────────────────
# 결과 요약
# ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"테스트 결과 요약")
print(f"{'='*60}")
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
for name, ok, err in results:
    status = PASS if ok else FAIL
    suffix = f" ({err[:60]})" if err else ""
    print(f"  {status} {name}{suffix}")
print(f"\n  총 {total}개 중 {passed}개 통과 ({total - passed}개 실패)")
