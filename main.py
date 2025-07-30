
import streamlit as st
import numpy as np
import nltk
import fitz  # PyMuPDF
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import Callback
from bs4 import BeautifulSoup
import pandas as pd
import re

from streamlit_lottie import st_lottie
import os
import pickle
import json
import requests
import time


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_coading = load_lottieurl("https://lottie.host/4dcaaa11-51ca-437d-859a-a2a25f1db47d/3V8A4vLMNq.json")
lottie_coading1 = load_lottieurl("https://lottie.host/c1b65e92-d5ff-48db-8050-e35348ab26fa/6IzO3r1va9.json")
lottie_coading2 = load_lottieurl("https://lottie.host/c1b65e92-d5ff-48db-8050-e35348ab26fa/6IzO3r1va9.json")

# Check if NLTK tokenizer model is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# Function to load text data from a PDF file and process it as HTML
def load_pdf_as_html_text(file_path):
    doc = fitz.open(file_path)
    html_content = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        html_content += page.get_text("html")
    return html_content


# Function to extract text from HTML content
def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Initialize list to hold the sequence of text
    text_sequence = []

    # Extract and maintain the sequence of paragraphs and tables
    for element in soup.find_all(['p', 'table']):
        if element.name == 'p':
            text = element.get_text()
            text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
            text_sequence.append(text)
        elif element.name == 'table':
            for row in element.find_all('tr'):
                for cell in row.find_all(['td', 'th']):
                    # Treat each cell's text as individual sentences/paragraphs
                    cell_text = cell.get_text()
                    cell_text = re.sub(r'\s+', ' ', cell_text)  # Replace multiple spaces with a single space
                    sentences = nltk.sent_tokenize(cell_text)
                    text_sequence.extend(sentences)

    return "\n".join(text_sequence)


# Function to tokenize the text
def tokenize_text(text):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1
    input_sequences = []
    for line in text.split('\n'):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, tokenizer, total_words


class StreamlitProgressBar(Callback):
    def __init__(self, total_epochs, progress_bar, status_text):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_bar = progress_bar
        self.status_text = status_text

    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.total_epochs
        self.progress_bar.progress(progress)
        self.status_text.text(f"Training: {int(progress * 100)}%")


# Function to pad sequences and create predictors and labels
def create_padded_sequences(input_sequences, total_words):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')
    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len


# Function to create the LSTM model
def create_model(predictors, label, max_sequence_len, total_words):
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1))
    model.add(LSTM(150, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Function to predict the next words with probabilities
def predict_next_words(model, tokenizer, text_seq, max_sequence_len, num_phrases, phrase_length=3):
    suggestions = set()
    while len(suggestions) < num_phrases:
        phrase = text_seq
        for _ in range(phrase_length):
            token_list = tokenizer.texts_to_sequences([phrase])[0]
            if not token_list:
                raise ValueError("The input text contains words that are not in the vocabulary.")

            token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
            predicted = model.predict(token_list, verbose=0)

            # Sample from the predicted word distribution to introduce diversity
            predicted_word_index = np.random.choice(len(predicted[0]), p=predicted[0])
            predicted_word = tokenizer.index_word.get(predicted_word_index, "")

            phrase += " " + predicted_word

        suggestions.add(phrase.strip())

    return list(suggestions)


# Streamlit app
st.title('Custom Text Autocomplete and Suggestion System')

st.lottie(lottie_coading, key="coading")

# Upload PDF or HTML file
uploaded_file = st.file_uploader("Choose a PDF or HTML file", type=["pdf", "html"])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        # Save the uploaded PDF file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button('Extract Text from PDF'):
            # Load and preprocess data from the PDF file as HTML
            html_content = load_pdf_as_html_text("temp.pdf")
            text = extract_text_from_html(html_content)
            st.session_state.text = text
            st.session_state.word_count = len(text.split())
            st.success('Text extracted successfully from PDF!')

    elif uploaded_file.type == "text/html":
        # Save the uploaded HTML file temporarily
        html_content = uploaded_file.getvalue().decode("utf-8")
        st.session_state.html_content = html_content

        if st.button('Extract Text from HTML'):
            # Extract text from the HTML file
            text = extract_text_from_html(html_content)
            st.session_state.text = text
            st.session_state.word_count = len(text.split())
            st.success('Text extracted successfully from HTML!')

if 'text' in st.session_state:
    # Display the extracted text
    st.subheader('Extracted Text:')
    st.text_area(label='Extracted Text', value=st.session_state.text, height=200)

    # Display the number of words in the extracted text
    st.write(f'Number of words in the extracted text: {st.session_state.word_count}')

    if st.button('Train Model'):
        text = st.session_state.text
        input_sequences, tokenizer, total_words = tokenize_text(text)
        predictors, label, max_sequence_len = create_padded_sequences(input_sequences, total_words)

        # Create and train the model
        progress_bar = st.progress(0)
        status_text = st.empty()
        with st.spinner('Training the model...'):
            model = create_model(predictors, label, max_sequence_len, total_words)
            epochs = 150
            progress_callback = StreamlitProgressBar(epochs, progress_bar, status_text)
            model.fit(predictors, label, epochs=epochs, callbacks=[progress_callback])

        # Save the model with a unique identifier
        model_count = len([name for name in os.listdir() if name.startswith("model_") and name.endswith(".h5")])
        model_filename = f"model_{model_count + 1}.h5"
        model.save(model_filename)

        tokenizer_filename = f"tokenizer_{model_count + 1}.pickle"
        with open(tokenizer_filename, "wb") as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        st.success(f'Model trained successfully and saved as {model_filename}!')

        # Store the model and tokenizer filenames in session state
        st.session_state.model_filename = model_filename
        st.session_state.tokenizer_filename = tokenizer_filename
        st.session_state.max_sequence_len = max_sequence_len

# Prediction section
if 'model_filename' in st.session_state and 'tokenizer_filename' in st.session_state:
    text_seq = st.text_input("Enter the input text for next word prediction:")

    if text_seq:
        with st.spinner('Generating suggestions...'):
            try:
                # Load the model and tokenizer for prediction
                model = load_model(st.session_state.model_filename)

                with open(st.session_state.tokenizer_filename, "rb") as handle:
                    tokenizer = pickle.load(handle)
                suggestions = predict_next_words(model, tokenizer, text_seq, st.session_state.max_sequence_len, 5, phrase_length=3)

                # Display suggestions
                st.write(f"Suggestions for '{text_seq}':")
                for phrase in suggestions:
                    st.write(f"{phrase}")

                st.balloons()
            except ValueError as ve:
                st.error(f"Error during prediction: {ve}")
                st.lottie(lottie_coading1, key="coading1")
                time.sleep(5)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.lottie(lottie_coading1, key="coading1")
                time.sleep(5)


