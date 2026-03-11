# 🎬 IMDB Movie Review Sentiment Analyzer

A web application that predicts whether a movie review is **Positive** or **Negative** using a **Recurrent Neural Network (RNN)** trained on the **IMDB movie review dataset**.  
The model is deployed as an interactive web app using **Streamlit**.

---

## 🚀 Live Demo
https://moviereview-c6doruaya4ohdynouceerm.streamlit.app/

---

## 📌 Project Overview
This project uses **Natural Language Processing (NLP)** and a **Simple RNN model** to analyze movie reviews and determine their sentiment.

Users can enter any movie review and instantly receive:
- Predicted sentiment (Positive / Negative)
- Model confidence score
- Visual feedback in the interface

---

## 🧠 Model Details

- **Model Type:** Simple Recurrent Neural Network (RNN)
- **Dataset:** IMDB Movie Review Dataset
- **Vocabulary Size:** 10,000 most frequent words
- **Input Length:** 500 tokens
- **Framework:** TensorFlow / Keras

---

## 🛠️ Technologies Used

- Python  
- TensorFlow / Keras  
- Natural Language Processing (NLP)  
- NumPy  
- Streamlit  
- HDF5 (h5py)

---

## 📂 Project Structure
```movie_review/
│
├── app.py # Streamlit web application
├── simple_rnn_imdb.h5 # Trained RNN model
├── requirements.txt # Python dependencies
├── prediction.ipynb # Model inference notebook
├── simplernn.ipynb # Model training notebook
└── README.md # Project documentation
```


---

## ⚙️ Installation

### 1️⃣ Clone the repository
git clone https://github.com/InduNallagopu/Movie_Review.git
cd Movie_Review

### 2️⃣ Install dependencies
pip install -r requirements.txt

### 3️⃣ Run the Streamlit app
streamlit run app.py


---

## 💡 How It Works

1. User enters a movie review in the text box.
2. The text is **tokenized and converted into integer sequences**.
3. The sequence is **padded to a fixed length of 500**.
4. The trained **RNN model predicts sentiment probability**.
5. The result is displayed with sentiment label and confidence score.

---

## 🎯 Features

✔ Interactive Streamlit UI  
✔ Real-time sentiment prediction  
✔ Clear review button  
✔ Confidence score visualization  
✔ Responsive layout  

---

## 📊 Example

**Input Review**
This movie was amazing and the acting was fantastic!


**Prediction**
Sentiment: Positive 😊
Confidence Score: 0.94

---

## 🔮 Future Improvements

- Add sentiment gauge visualization  
- Support multi-language reviews  
- Use LSTM or Transformer models  
- Add real-time typing predictions  

---

## 👩‍💻 Author

**Indu Kumari**  
Computer Science Student | Machine Learning Enthusiast

---
