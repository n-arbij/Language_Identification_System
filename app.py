from __future__ import annotations

import streamlit as st

from main import CONFUSION_MATRIX_PATH, load_trained_bundle, predict_language


st.set_page_config(
    page_title="Language Identification System",
    page_icon="globe",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 48%, #eef4ff 100%);
    }
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        background: rgba(1f, 1f, 1f, 1);
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
    }
    .hero h1{
        font-size: 2.5rem;
        font-weight: 800;
        color: #000;
    }
    .result-box {
        padding: 1rem 1.2rem;
        border-radius: 16px;
        background: #0f172a;
        color: #f8fafc;
        font-size: 1.1rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading trained model...")
def get_bundle():
    return load_trained_bundle()


bundle = get_bundle()

st.markdown(
    """
    <div class="hero">
        <h1>Language Identification System</h1>
        <p style="color:#334155;">
            Enter a short text sample and the model will predict whether it is English, Swahili, Sheng, or Luo.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([2, 1])

with left_col:
    sample_text = st.text_area(
        "Text sample",
        height=160,
        placeholder="Type or paste a short message here...",
    )
    predict_clicked = st.button("Predict language", type="primary", use_container_width=True)

    if predict_clicked:
        if not sample_text.strip():
            st.warning("Enter a text sample first.")
        else:
            try:
                predicted = predict_language(sample_text, bundle)
                st.markdown(
                    f'<div class="result-box">Predicted language: {predicted}</div>',
                    unsafe_allow_html=True,
                )
            except ValueError as exc:
                st.warning(str(exc))

with right_col:
    st.metric("Best model", bundle.model_name)
    st.metric("Languages", len(bundle.labels))
    st.metric("Balanced classes", "Yes")
