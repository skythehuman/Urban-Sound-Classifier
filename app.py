"""
Urban Sound Classifier — Streamlit Dashboard
=============================================
Browser-based urban noise classification using a 4-fold ensemble
of ResNet CNN models with Mel-spectrogram and MFCC features.

Run:  streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import librosa
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from classifier import (
    load_ensemble, extract_features, ensemble_predict,
    get_mel_spectrogram, get_mfcc,
    CLASS_NAMES, DISPLAY_NAMES, TAXONOMY, get_category,
    SR, N_MELS, N_MFCC, PAD_SIZE, N_FOLDS,
)

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Urban Sound Classifier",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    h1 { font-size: 1.6rem !important; font-weight: 700 !important; }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e5ea;
        border-radius: 10px;
        padding: 12px 16px;
    }

    .category-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .cat-transportation { background: #dbeafe; color: #1e40af; }
    .cat-industry     { background: #fef3c7; color: #b45309; }
    .cat-construction   { background: #B9E5EB; color: #177380;}
    .cat-music          { background: #ede9fe; color: #6d28d9; }
    .cat-animal         { background: #dcfce7; color: #15803d; }

    .info-box {
        background: #f7f8fa;
        border: 1px solid #e2e5ea;
        border-radius: 10px;
        padding: 16px 20px;
        font-size: 0.875rem;
        line-height: 1.7;
    }
</style>
""", unsafe_allow_html=True)

CATEGORY_COLORS = {
    "Transportation": "#3b82f6",
    "Industry": "#f59e0b",
    "Construction": "#0BD6F5",
    "Music": "#8b5cf6",
    "Animal": "#22c55e",
}

# ── Load Models (cached) ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def get_models():
    return load_ensemble("models")


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Urban Sound Classifier")
    st.caption("17-class urban noise classification using ResNet + feature ensemble")
    st.divider()

    page = st.radio(
        "Navigate",
        ["Classify", "Taxonomy", "Model Info"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("4-fold ensemble · Mel + MFCC · 96.2% accuracy")


# =====================================================================
# PAGE 1 — Classify
# =====================================================================
if page == "Classify":
    st.title("Urban Sound Classifier")
    st.caption(
        "Upload a WAV audio file to classify it into one of 17 urban noise types. "
        "The ensemble model averages predictions from feature ensemble models (Mel-spectrogram + MFCC)."
    )

    # ── Upload ───────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload audio file",
        type=["wav"],
        help="WAV format, any sample rate (will be resampled to 20 kHz)",
    )

    if uploaded is not None:
        # Load audio
        audio_bytes = uploaded.read()
        audio, sr_orig = librosa.load(io.BytesIO(audio_bytes), sr=SR)
        duration = len(audio) / SR

        # Audio player
        st.audio(audio_bytes, format="audio/wav")

        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("Duration", f"{duration:.1f} sec")
        col_info2.metric("Sample Rate", f"{SR:,} Hz")
        col_info3.metric("Samples", f"{len(audio):,}")

        # ── Feature Visualization ────────────────────────────────────
        st.divider()
        st.subheader("Feature Extraction")

        tab_mel, tab_mfcc = st.tabs(["Mel-spectrogram", "MFCC"])

        with tab_mel:
            mel_raw = get_mel_spectrogram(audio, SR)
            fig_mel, ax_mel = plt.subplots(figsize=(10, 3))
            img = librosa.display.specshow(
                mel_raw, sr=SR, x_axis="time", y_axis="mel",
                ax=ax_mel, cmap="magma",
            )
            fig_mel.colorbar(img, ax=ax_mel, format="%+2.0f dB")
            ax_mel.set_title("Mel-spectrogram", fontsize=11, fontweight=600)
            st.pyplot(fig_mel, use_container_width=True)
            plt.close(fig_mel)

        with tab_mfcc:
            mfcc_raw = get_mfcc(audio, SR)
            fig_mfcc, ax_mfcc = plt.subplots(figsize=(10, 3))
            img2 = librosa.display.specshow(
                mfcc_raw, sr=SR, x_axis="time",
                ax=ax_mfcc, cmap="coolwarm",
            )
            fig_mfcc.colorbar(img2, ax=ax_mfcc)
            ax_mfcc.set_title("MFCC", fontsize=11, fontweight=600)
            st.pyplot(fig_mfcc, use_container_width=True)
            plt.close(fig_mfcc)

        # ── Classification ───────────────────────────────────────────
        st.divider()
        st.subheader("Classification Result")

        with st.spinner("Running ensemble inference (8 models)…"):
            mel_models, mfcc_models = get_models()
            mel_feat, mfcc_feat = extract_features(audio, SR)
            probs = ensemble_predict(mel_models, mfcc_models, mel_feat, mfcc_feat)

        # Top prediction
        top_idx = np.argmax(probs)
        top_class = CLASS_NAMES[top_idx]
        top_prob = probs[top_idx]
        top_display = DISPLAY_NAMES[top_class]
        cat_l1, cat_l2 = get_category(top_class)

        # Result display
        r1, r2 = st.columns([1, 2])
        with r1:
            st.metric("Predicted Class", top_display)
            st.metric("Confidence", f"{top_prob * 100:.1f}%")

            cat_cls = f"cat-{cat_l1.lower().split()[0]}"
            st.markdown(
                f'<span class="category-tag {cat_cls}">'
                f'{cat_l1} → {cat_l2}</span>',
                unsafe_allow_html=True,
            )

        with r2:
            # Top-5 bar chart
            sorted_idx = np.argsort(probs)[::-1][:7]
            top_classes = [DISPLAY_NAMES[CLASS_NAMES[i]] for i in sorted_idx]
            top_probs = [probs[i] * 100 for i in sorted_idx]
            colors = []
            for i in sorted_idx:
                c1, _ = get_category(CLASS_NAMES[i])
                colors.append(CATEGORY_COLORS.get(c1, "#888"))

            fig = go.Figure(go.Bar(
                x=top_probs[::-1],
                y=top_classes[::-1],
                orientation="h",
                marker_color=colors[::-1],
                text=[f"{p:.1f}%" for p in top_probs[::-1]],
                textposition="outside",
            ))
            fig.update_layout(
                title="Top Predictions",
                xaxis_title="Probability (%)",
                xaxis_range=[0, max(top_probs) * 1.25],
                height=300,
                margin=dict(l=10, r=20, t=40, b=30),
                template="plotly_white",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Full probability table (expandable)
        with st.expander("All 17 class probabilities"):
            df = pd.DataFrame({
                "Class": [DISPLAY_NAMES[c] for c in CLASS_NAMES],
                "Probability": probs,
            }).sort_values("Probability", ascending=False)
            df["Probability"] = df["Probability"].apply(lambda x: f"{x*100:.2f}%")
            df = df.reset_index(drop=True)
            df.index = df.index + 1
            st.dataframe(df, use_container_width=True, hide_index=False)

    else:
        # Empty state
        st.info("Upload a WAV file above to start classifying.")


# =====================================================================
# PAGE 2 — Taxonomy
# =====================================================================
elif page == "Taxonomy":
    st.title("Noise Source Taxonomy")
    st.caption("5 categories, 8 subcategories, 17 noise source classes")

    for l1, l2_dict in TAXONOMY.items():
        color = CATEGORY_COLORS.get(l1, "#888")
        st.markdown(
            f"### <span style='color:{color}'>{l1}</span>",
            unsafe_allow_html=True,
        )

        cols = st.columns(len(l2_dict))
        for col, (l2, classes) in zip(cols, l2_dict.items()):
            with col:
                st.markdown(f"**{l2}**")
                for cls in classes:
                    st.markdown(f"- {DISPLAY_NAMES[cls]}")
        st.divider()

    # Summary table
    st.subheader("Summary Table")
    rows = []
    for l1, l2_dict in TAXONOMY.items():
        for l2, classes in l2_dict.items():
            for cls in classes:
                rows.append({
                    "Category": l1,
                    "Subcategory": l2,
                    "Class": DISPLAY_NAMES[cls],
                    "Label": cls,
                })
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


# =====================================================================
# PAGE 3 — Model Info
# =====================================================================
elif page == "Model Info":
    st.title("Model Architecture & Performance")

    # Architecture overview
    st.subheader("Ensemble Architecture")
    st.markdown("""
<div class="info-box">
<b>Input</b>: Audio (WAV) → resample to 20 kHz<br><br>
<b>Feature Extraction</b>:<br>
&nbsp;&nbsp;├── Mel-spectrogram (n_mels=40) → (40 × 2344 × 1)<br>
&nbsp;&nbsp;└── MFCC (n_mfcc=40) → (40 × 2344 × 1)<br><br>
<b>Models</b>: 4 × Mel DCNN + 4 × MFCC DCNN = 8 models (4-fold CV)<br><br>
<b>Ensemble</b>: Average class probabilities → argmax → Predicted Class (17)
</div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # Model specs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Parameters", "84,755")
    c2.metric("Model Size", "~1.3 MB each")
    c3.metric("Total Models", "8")
    c4.metric("Classes", "17")

    st.divider()

    # Performance
    st.subheader("Classification Performance")

    perf_col1, perf_col2 = st.columns(2)

    with perf_col1:
        st.markdown("**Feature Comparison**")
        perf_df = pd.DataFrame({
            "Feature": ["Mel-spectrogram only", "MFCC only", "Ensemble (Mel + MFCC)"],
            "Accuracy": [90.24, 86.64, 96.20],
        })
        fig = px.bar(
            perf_df, x="Feature", y="Accuracy",
            color="Feature",
            color_discrete_sequence=["#93c5fd", "#c4b5fd", "#2563eb"],
            text="Accuracy",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(
            yaxis_range=[80, 100], showlegend=False,
            height=350, margin=dict(t=20, b=40),
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with perf_col2:
        st.markdown("**Training Configuration**")
        config_df = pd.DataFrame({
            "Parameter": [
                "Optimizer", "Learning Rate", "Loss Function",
                "Batch Size", "Validation", "Sample Rate",
                "Feature Size", "Pad Size",
            ],
            "Value": [
                "Adam", "0.0005",
                "Sparse Categorical Cross-Entropy",
                "See notebook", "4-fold Stratified CV",
                f"{SR:,} Hz", f"{N_MELS} bands / coefficients",
                f"{PAD_SIZE:,} frames",
            ],
        })
        st.dataframe(config_df, use_container_width=True, hide_index=True)

    st.divider()

    # DCNN architecture detail
    st.subheader("ResNet Block Structure")
    st.markdown("""
<div class="info-box">
<b>Block 1</b>: Conv2D(16) → BN → ReLU → MaxPool(2×2)<br>
<b>Residual Block 1</b>: [BN→ReLU→Conv(16)→BN→ReLU→Conv(16)] + skip(Conv(32)) → Add → ReLU<br>
<b>Residual Block 2</b>: MaxPool → [BN→ReLU→Conv(32)→BN→ReLU→Conv(32)] + skip(Conv(32)) → Add<br>
<b>Residual Block 3</b>: [BN→ReLU→Conv(32)→BN→ReLU→Conv(32)] + skip(Conv(64)) → Add<br>
<b>Residual Block 4</b>: MaxPool → [BN→ReLU→Conv(64)→BN→ReLU→Conv(64)] + skip(Conv(64)) → Add<br>
<b>Head</b>: GlobalAvgPool → Dense(32) → BN → ReLU → Dropout → Dense(17, softmax)
</div>
    """, unsafe_allow_html=True)

    st.divider()
    st.subheader("Data Sources")
    st.markdown("""
- [UrbanSound8K](https://urbansounddataset.weebly.com/) — Salamon et al. (2014)
- [ESC-50](https://github.com/karolpiczak/ESC-50) — Piczak (2015)
- [FSDKaggle2019](https://zenodo.org/record/3612637) — Fonseca et al. (2019)
- [Urban Sound](https://www.aihub.or.kr/) - AI Hub, Ministry of Science and ICT
    """)
