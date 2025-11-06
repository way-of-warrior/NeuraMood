import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
import tensorflow as tf
import requests
import time
import matplotlib.pyplot as plt
from streamlit.components.v1 import html as st_html
import base64, os
st.set_page_config(page_title='NeuraMood', page_icon='🧠', layout='wide',initial_sidebar_state='collapsed')

if "boot_done" not in st.session_state:
    with st.spinner("Diving into your brain"):
        time.sleep(1.0)  # tiny pause for effect
    st.session_state.boot_done = True

st.markdown(
    """
    <div style="text-align:center">
        <h1 style="color:#00ADB5;">🧠 NeuraMood</h1>
        <p style="font-size:18px; color:#A3A3A3;">
        Real-time emotion recognition from brainwave data using Deep Learning
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
art_dir = Path('artifacts')
model_path = st.sidebar.text_input('Model file', str(art_dir/'model.keras'))
scaler_path = st.sidebar.text_input('Scaler file', str(art_dir/'scaler.joblib'))
featcols_path = st.sidebar.text_input('Feature columns', str(art_dir/'feature_columns.json'))
labels_path = st.sidebar.text_input('Labels', str(art_dir/'labels.json'))

@st.cache_resource(show_spinner=False)
def load_artifacts(model_path, scaler_path, featcols_path, labels_path):
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    featcols = json.load(open(featcols_path))
    labels = json.load(open(labels_path))
    return model, scaler, featcols, labels

try:
    model, scaler, FEAT_COLS, LABELS = load_artifacts(model_path, scaler_path, featcols_path, labels_path)
    st.success('Artifacts loaded ✅')
except Exception as e:
    st.error(f'Failed to load artifacts: {e}')
    st.stop()

st.markdown('### 1) Upload data')
upload = st.file_uploader('CSV with a header row. For single prediction, upload one row. For batch, upload multiple rows.', type=['csv'])

def ensure_columns(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f'Missing {len(missing)} required columns (showing first 10): {missing[:10]}')
    return df[required_cols].copy()

if upload is not None:
    df_raw = pd.read_csv(upload)
    try:
        df = ensure_columns(df_raw, FEAT_COLS)
    except Exception as e:
        st.error(str(e)); st.stop()

    X = scaler.transform(df.values)
    y_prob = model.predict(X, verbose=0)
    y_pred = y_prob.argmax(axis=1)

    st.markdown('### 2) Predictions')
    if len(df) == 1:
        p = y_prob[0]
        pred_label = LABELS[int(np.argmax(p))] if LABELS else int(np.argmax(p))
        
        # For single predictions:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric('Predicted Emotion', pred_label)
        st.markdown("</div>", unsafe_allow_html=True)

        pdf = pd.DataFrame({'label': LABELS if LABELS else list(range(len(p))), 'prob': p})
        fig = px.bar(pdf.sort_values('prob', ascending=False), x='label', y='prob', title='Confidence by class')
        
        # For plots:
        st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander('Show input feature snippet'):
            st.dataframe(df.iloc[:, : min(20, df.shape[1])])
    else:
        out = pd.DataFrame({'prediction': [LABELS[i] if LABELS else int(i) for i in y_pred]})
        res = pd.concat([out, pd.DataFrame(y_prob, columns=[f'prob_{lbl}' for lbl in LABELS])], axis=1)
        st.success(f'Predicted {len(df)} rows')
        st.dataframe(res.head(25))
        dist = out['prediction'].value_counts().reset_index()
        dist.columns = ['label','count']
        fig = px.bar(dist, x='label', y='count', title='Batch prediction distribution')
        st.plotly_chart(fig, use_container_width=True)

st.divider()
with st.expander('🎤 Event add‑on: Confusion Matrix from labeled CSV'):
    gt_file = st.file_uploader('Upload a CSV with ground truth label column named "label" (string or int).', type=['csv'], key='gt')
    if gt_file is not None:
        dfg = pd.read_csv(gt_file)
        try:
            Xg = ensure_columns(dfg.drop(columns=['label']), FEAT_COLS)
        except Exception as e:
            st.error(str(e)); st.stop()
        yg_true = dfg['label']
        if yg_true.dtype == object and LABELS:
            map2idx = {lbl:i for i,lbl in enumerate(LABELS)}
            yg_idx = yg_true.map(map2idx)
        else:
            yg_idx = yg_true
        Xg_scaled = scaler.transform(Xg.values)
        ygp = model.predict(Xg_scaled, verbose=0).argmax(axis=1)
        from sklearn.metrics import confusion_matrix
        import plotly.figure_factory as ff
        cm = confusion_matrix(yg_idx, ygp, labels=list(range(len(LABELS))))
        fig = ff.create_annotated_heatmap(cm, x=LABELS, y=LABELS, colorscale='Blues', showscale=True)
        fig.update_layout(title='Confusion Matrix', xaxis_title='Predicted', yaxis_title='Actual')
        st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <hr>
    <p style='text-align:center; color:gray;'>
    Made with ❤️ by <b>Team CerebralConnect</b> | INNOTECH 2025
    </p>
    """,
    unsafe_allow_html=True
)
