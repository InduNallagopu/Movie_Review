import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
# This is the "Magic" line that fixes the batch_shape/ragged errors
tf.keras.config.enable_unsafe_deserialization()

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

#load the IMDB dataset word index
word_index=imdb.get_word_index()
#reverse the word index to get the mapping from integers to words
reverse_word_index={value:key for key,value in word_index.items()}
#load the pre-trained model
#model=load_model('simple_rnn_imdb.h5')
import h5py
import json

def load_model_fixed(model_path):
    # Open the file and manually fix the config
    with h5py.File(model_path, 'r') as f:
        model_config = json.loads(f.attrs.get('model_config'))
    
    # This helper removes the 'ragged' key that causes the error
    def remove_ragged(obj):
        if isinstance(obj, dict):
            obj.pop('ragged', None)
            for key in obj:
                remove_ragged(obj[key])
        elif isinstance(obj, list):
            for item in obj:
                remove_ragged(item)

    remove_ragged(model_config)
    
    # Reconstruct the model structure from the cleaned config
    model = tf.keras.models.model_from_json(json.dumps(model_config))
    # Load the actual weights
    model.load_weights(model_path)
    return model

# NEW LOAD LINE:
model = load_model_fixed('simple_rnn_imdb.h5')

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
    
