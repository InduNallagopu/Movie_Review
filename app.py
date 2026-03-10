

import os
import json
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# --- 1. THE COMPATIBILITY BRIDGE (The "Magic" lines) ---
try:
    tf.keras.config.enable_unsafe_deserialization()
except AttributeError:
    pass # We are in Keras 2, which is fine

def load_model_surgery(model_path):
    with h5py.File(model_path, 'r') as f:
        model_config_raw = f.attrs.get('model_config')
        if isinstance(model_config_raw, bytes):
            model_config_raw = model_config_raw.decode('utf-8')
        model_config = json.loads(model_config_raw)
    
    def clean_config(obj):
        if isinstance(obj, dict):
            for key in ['batch_shape', 'ragged', 'groups']:
                obj.pop(key, None)
            for v in obj.values(): clean_config(v)
        elif isinstance(obj, list):
            for item in obj: clean_config(item)

    clean_config(model_config)
    model = tf.keras.models.model_from_json(json.dumps(model_config))
    model.load_weights(model_path)
    return model

# --- 2. YOUR ORIGINAL DATASET CODE (Kept exactly the same) ---
# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the model using the surgery function instead of load_model
model = load_model_surgery('simple_rnn_imdb.h5')

def decode_review(encoded_review):
    #decode the review by mapping integers back to words
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

def preprocess_text(text):
    words=text.lower().split()
    #convert words to integers using the word index
    encoded_review=[word_index.get(word,2)+3 for word in words]
    #pad the encoded review to the same length as the training data
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

#predict the sentiment of a sample review
def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    sentiment='positive' if prediction[0][0]>0.5 else 'negative'
    return sentiment,prediction[0][0]

#streamlit app
import streamlit as st
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to predict its sentiment (positive or negative).')
user_input=st.text_area('Movie Review')
if st.button('Predict Sentiment'):
    preprocess_input=preprocess_text(user_input)
    prediction=model.predict(preprocess_input)
    sentiment='positive' if prediction[0][0]>0.5 else 'negative'

    #display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction_score: {prediction[0][0]:.2f}')
else:
    st.write('Please enter a movie review and click the button to predict its sentiment.')
    
