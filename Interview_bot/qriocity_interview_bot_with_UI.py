import streamlit as st
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import nltk
import speech_recognition as sr
import pyttsx3

nltk.download('stopwords')
nltk.download('punkt')

# Load the questions and answers from the CSV file
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocess text data
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_text = [word for word in word_tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_text)

# Train the chatbot using the questions and answers
def train_chatbot(data):
    vectorizer = TfidfVectorizer()
    processed_questions = [preprocess_text(question) for question in data['Question']]
    vectors = vectorizer.fit_transform(processed_questions)
    return vectors, vectorizer, data['Answer']

# Get the most relevant answer based on user input
def get_answer(user_input, vectorizer, vectors, answers):
    user_input = preprocess_text(user_input)
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, vectors)
    max_similarity_index = similarities.argmax()
    return answers.iloc[max_similarity_index]

# Parse keywords from the resume
def parse_resume_keywords(resume_path):
    keywords = set()
    global page_number
    page_number = 0

    with open(resume_path, 'rb') as resume_file:
        pdf_reader = PyPDF2.PdfReader(resume_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_number]
            text = page.extract_text().lower()
            keywords.update(word_tokenize(text))

    return keywords

# Function to convert text to speech
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Streamlit UI
def streamlit_app():
    st.title("Resume Chatbot")

    # Upload a resume
    resume_path = st.file_uploader("Upload your resume (PDF format):", type=["pdf"])

    if resume_path:
        # Load questions and answers
        file_path = (r'C:\Users\prajw\Downloads\email_spam\email_spam\Final_Dataset-500.csv')  # Change to your CSV file path
        data = load_data(file_path)

        # Train the chatbot
        vectors, vectorizer, answers = train_chatbot(data)

        # Parse keywords from the resume
        resume_keywords = parse_resume_keywords(resume_path)

        # Ask personalized questions based on resume keywords
        for keyword in resume_keywords:
            st.text(f"Tell me about your experience with {keyword}: ")

            # Use speech recognition for speech input
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                st.text("Speak now...")
                audio = recognizer.listen(source)

            try:
                user_input = recognizer.recognize_google(audio)
                st.text(f"User: {user_input}")

                # Get and print the AI's response
                answer = get_answer(user_input, vectorizer, vectors, answers)
                st.text(f"AI: {answer}")

                # Convert the AI's response to speech
                text_to_speech(answer)

            except sr.UnknownValueError:
                st.text("Sorry, I could not understand your speech.")

if __name__ == "__main__":
    streamlit_app()
