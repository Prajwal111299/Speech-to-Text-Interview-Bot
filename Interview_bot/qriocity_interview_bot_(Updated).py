import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import nltk
import random
import re
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
    if isinstance(text, str):
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text.lower())
        filtered_text = [word for word in word_tokens if word.isalnum() and word not in stop_words]
        return ' '.join(filtered_text)
    else:
        return str(text)

# Train the chatbot using the questions and answers
def train_chatbot(data):
    vectorizer = TfidfVectorizer()
    processed_questions = [preprocess_text(question) for question in data['Question']]
    vectors = vectorizer.fit_transform(processed_questions)
    return vectors, vectorizer, data['Answer']

# Get a random subset of questions
def get_random_questions(data, n=10):
    questions = data['Question'].tolist()
    random.shuffle(questions)
    return questions[:n]

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

# Remove numbers from recognized speech
def remove_numbers(text):
    return re.sub(r'\d', '', text)

# Main function
def main():
    # Load questions and answers
    file_path = (r'C:\Users\prajw\Downloads\Qriocity\Qriocity_resumes_interview_bot\Final_Dataset-500.csv')  # Change to your CSV file path
    data = load_data(file_path)
    
    # Train the chatbot
    vectors, vectorizer, _ = train_chatbot(data)

    # Ask user to upload a resume
    resume_path = input("Please upload your resume in PDF format: ")

    # Parse keywords from the resume
    resume_keywords = parse_resume_keywords(resume_path)

    # Get 10 random technical questions without numbers
    random_questions = get_random_questions(data, n=10)
    random_questions = [q for q in random_questions if not any(char.isdigit() for char in q)]

    # Ask personalized questions based on resume keywords
    for question in random_questions:
        question_text = f"Here's a question for you: {question}?"
        print(f"Chatbot: {question_text}")  # Print the chatbot's question
        text_to_speech(question_text)
        
        # Use speech recognition for speech input
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Speak now...")
            audio = recognizer.listen(source)

        try:
            user_input = recognizer.recognize_google(audio)
            
            # Remove numbers from recognized speech
            user_input = remove_numbers(user_input)
            
            print(f"User: {user_input}")

        except sr.UnknownValueError:
            print("Sorry, I could not understand your speech.")

    # Calculate and display the user's recruitment probability based on responses
    user_responses = [preprocess_text(answer) for answer in random_questions]
    optimal_answers = [preprocess_text(answer) for answer in data['Answer']]
    
    user_vector = vectorizer.transform(user_responses)
    optimal_vectors = vectorizer.transform(optimal_answers)

    similarities = cosine_similarity(user_vector, optimal_vectors)
    recruitment_probability = similarities.mean()
    print(f"\nRecruitment Probability: {recruitment_probability * 100}%")

if __name__ == "__main__":
    main()
