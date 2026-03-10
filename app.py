

import os
import json
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import streamlit as st

# --- 1. COMPATIBILITY BRIDGE ---
# This prevents the 'config' attribute error in Keras 2
try:
    tf.keras.config.enable_unsafe_deserialization()
except AttributeError:
    pass 



def load_model_surgery(model_path):
    """Cleans batch_shape, ragged, and complex DTypePolicy for Keras 2 compatibility."""
    with h5py.File(model_path, 'r') as f:
        model_config_raw = f.attrs.get('model_config')
        if isinstance(model_config_raw, bytes):
            model_config_raw = model_config_raw.decode('utf-8')
        model_config = json.loads(model_config_raw)
    
    def clean_config(obj):
        if isinstance(obj, dict):
            # 1. Remove keys that Keras 2 doesn't understand
            for key in ['batch_shape', 'ragged', 'groups']:
                obj.pop(key, None)
            
            # 2. Fix the DTypePolicy error: Convert dict dtype to a simple string
            if 'dtype' in obj and isinstance(obj['dtype'], dict):
                # Extract the actual name (e.g., 'float32') from the policy dict
                inner_config = obj['dtype'].get('config', {})
                obj['dtype'] = inner_config.get('name', 'float32')
            
            for v in obj.values():
                clean_config(v)
        elif isinstance(obj, list):
            for item in obj:
                clean_config(item)

    clean_config(model_config)
    
    # Reconstruct the model structure from the cleaned config
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

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to predict its sentiment.')

user_input = st.text_area('Movie Review')

if st.button('Predict Sentiment'):
    if user_input:
        processed_input = preprocess_text(user_input)
        prediction = model.predict(processed_input)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
        
        st.write(f'**Sentiment:** {sentiment}')
        st.write(f'**Confidence Score:** {prediction[0][0]:.4f}')
    else:
        st.warning("Please enter a review first.")
    
