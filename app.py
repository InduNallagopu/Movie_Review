

import os
import json
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import streamlit as st

st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

st.sidebar.title("About")

st.sidebar.info(
"""
This app predicts movie review sentiment using a **Simple RNN model** trained on the IMDB dataset.

Technologies used:
- Python
- TensorFlow
- NLP
- Streamlit
"""
)

# --- 1. COMPATIBILITY BRIDGE ---
# This prevents the 'config' attribute error in Keras 2
try:
    tf.keras.config.enable_unsafe_deserialization()
except AttributeError:
    pass 




def load_model_surgery(model_path):
    """Cleans batch_shape, ragged, DTypePolicy, and unknown TensorShapes for Keras 2."""
    with h5py.File(model_path, 'r') as f:
        model_config_raw = f.attrs.get('model_config')
        if isinstance(model_config_raw, bytes):
            model_config_raw = model_config_raw.decode('utf-8')
        model_config = json.loads(model_config_raw)
    
    def clean_config(obj):
        if isinstance(obj, dict):
            # 1. Remove keys Keras 2 doesn't understand
            for key in ['batch_shape', 'ragged', 'groups']:
                obj.pop(key, None)
            
            # 2. FIX FOR TensorShape ERROR: Force explicit input shape
            # We set it to 500 to match your padding length
            if obj.get('class_name') in ['InputLayer', 'Embedding']:
                if 'config' in obj:
                    obj['config']['batch_input_shape'] = [None, 500]

            # 3. Fix the DTypePolicy error
            if 'dtype' in obj and isinstance(obj['dtype'], dict):
                inner_config = obj['dtype'].get('config', {})
                obj['dtype'] = inner_config.get('name', 'float32')
            
            for v in obj.values():
                clean_config(v)
        elif isinstance(obj, list):
            for item in obj:
                clean_config(item)

    clean_config(model_config)
    
    # Reconstruct the model structure
    model = tf.keras.models.model_from_json(json.dumps(model_config))
    # Load the actual weights
    model.load_weights(model_path)
    return model

# --- 2. DATA LOADING (Your Original Logic) ---
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the model using our fix instead of tf.keras.models.load_model
model = load_model_surgery('simple_rnn_imdb.h5')

# --- 3. PREPROCESSING & UI ---
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


st.markdown("""
# 🎬 IMDB Movie Review Sentiment Analyzer
Analyze whether a movie review is **Positive** or **Negative** using an RNN model.
""")
if "review_text" not in st.session_state:
    st.session_state.review_text = ""
user_input = st.text_area(
    "✍️ Enter your movie review",
    height=150,
    key="review_text",
    placeholder="Example: This movie was absolutely amazing!"
)


col1, col2 = st.columns(2)

with col1:
    predict_button = st.button("Analyze Sentiment")

with col2:
    clear_button = st.button("Clear Review")

if predict_button:
    if user_input:
        processed_input = preprocess_text(user_input)
        prediction = model.predict(processed_input)
        confidence = float(prediction[0][0])
        sentiment = "Positive" if confidence > 0.5 else "Negative"
        if sentiment == "Positive":
            st.success(f"😊 Sentiment: {sentiment}")
        else:
            st.error(f"😞 Sentiment: {sentiment}")
        st.write(f"Confidence Score: {confidence:.2f}")
        st.progress(confidence)
      
    else:
        st.warning("Please enter a review first.")
if clear_button:
    st.session_state.review_text = ""
    st.rerun()
    
