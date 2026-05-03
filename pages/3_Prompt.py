import streamlit as st
st.set_page_config(page_title="Prompt", layout="wide", initial_sidebar_state="collapsed")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ui import inject_ui, render_page_title
from core import prompts as prompt_store

inject_ui()
render_page_title("Prompt", "Ask, Selection, Report 3종 프롬프트를 편집하고 저장합니다.")

# 각 프롬프트의 필수 변수 정의
REQUIRED_VARS = {
    "ask":       ["{question}"],
    "selection": ["{question}", "{contexts}"],
    "report":    ["{question}", "{top3}"],
}

if "prompts_state" not in st.session_state:
    st.session_state.prompts_state = prompt_store.load_prompts()

prompts = st.session_state.prompts_state

tab1, tab2, tab3 = st.tabs(["Ask LLM", "Selection LLM", "Report LLM"])

with tab1:
    st.caption("정보 부족 시 사용자에게 되묻는 1단계 프롬프트. {question} 변수 필수.")
    new_ask = st.text_area("Ask Prompt", value=prompts["ask"], height=420, key="ask_area")
    prompts["ask"] = new_ask

    st.markdown("**Temperature**")
    prompts["ask_temp"] = st.slider(
        "ask_temperature",
        min_value=0.0, max_value=1.0,
        value=float(prompts.get("ask_temp", 0.0)),
        step=0.05,
        label_visibility="collapsed",
        help="0 → 항상 일관된 판단  /  1 → 매번 다른 판단. Ask는 0에 가까울수록 안정적이에요.",
        key="ask_temp_slider",
    )
    ta_col1, ta_col2 = st.columns(2)
    ta_col1.caption(f"현재값: **{prompts['ask_temp']:.2f}**")
    ta_col2.caption("권장: 0.00 (일관성 최우선)")

with tab2:
    st.caption("Top 3 정책을 선정하는 큐레이션 프롬프트. {question}, {contexts} 필수.")
    new_sel = st.text_area("Selection Prompt", value=prompts["selection"], height=480, key="sel_area")
    prompts["selection"] = new_sel

    st.markdown("**Temperature**")
    prompts["selection_temp"] = st.slider(
        "selection_temperature",
        min_value=0.0, max_value=1.0,
        value=float(prompts.get("selection_temp", 0.0)),
        step=0.05,
        label_visibility="collapsed",
        help="0 → 항상 같은 Top3 선정  /  높을수록 매번 달라질 수 있음. 재현성이 중요하면 0을 권장해요.",
        key="sel_temp_slider",
    )
    ts_col1, ts_col2 = st.columns(2)
    ts_col1.caption(f"현재값: **{prompts['selection_temp']:.2f}**")
    ts_col2.caption("권장: 0.00 (일관성 최우선)")

with tab3:
    st.caption("최종 종합 보고서 작성 프롬프트. {question}, {top3} 필수.")
    new_rep = st.text_area("Report Prompt", value=prompts["report"], height=520, key="rep_area")
    prompts["report"] = new_rep

    st.markdown("**Temperature**")
    prompts["report_temp"] = st.slider(
        "report_temperature",
        min_value=0.0, max_value=1.0,
        value=float(prompts.get("report_temp", 0.3)),
        step=0.05,
        label_visibility="collapsed",
        help="0 → 딱딱하고 정확한 보고서  /  0.3~0.5 → 자연스러운 서술체. 너무 높으면 환각이 증가해요.",
        key="rep_temp_slider",
    )
    tr_col1, tr_col2 = st.columns(2)
    tr_col1.caption(f"현재값: **{prompts['report_temp']:.2f}**")
    tr_col2.caption("권장: 0.20 ~ 0.40")

st.markdown("---")

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    if st.button("저장", type="primary", use_container_width=True):
        # ── 필수 변수 검증 ─────────────────────────────────────────
        errors = []
        label_map = {"ask": "Ask LLM", "selection": "Selection LLM", "report": "Report LLM"}
        for key, required in REQUIRED_VARS.items():
            for var in required:
                if var not in prompts[key]:
                    errors.append(f"**[{label_map[key]}]** 프롬프트에 필수 변수 `{var}` 가 없습니다.")

        if errors:
            for msg in errors:
                st.error(msg)
            st.warning("필수 변수를 복구한 뒤 다시 저장하세요. '기본값으로 복원' 버튼을 누르면 원래 프롬프트를 확인할 수 있어요.")
        else:
            prompt_store.save_prompts(prompts)
            st.session_state.prompts_state = prompts
            st.success("저장 완료. Home / Evaluation 페이지에서 즉시 반영됩니다.")

with c2:
    if st.button("기본값으로 복원", use_container_width=True):
        st.session_state.prompts_state = prompt_store.get_default_prompts()
        prompt_store.save_prompts(st.session_state.prompts_state)
        st.success("기본값으로 복원했습니다.")
        st.rerun()

with c3:
    st.info("변경 사항은 Home / Evaluation 페이지에서 즉시 반영됩니다.")
