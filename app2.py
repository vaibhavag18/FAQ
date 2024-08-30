import pandas as pd
import nltk
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import re

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Preprocessing function
def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    text = re.sub(r'[^\w\s]', '', text)  # Remove non-alphanumeric characters
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]
    return ' '.join(stemmed_tokens)

def preprocess_with_stopwords(text):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    text = re.sub(r'[^\w\s]', '', text)  # Remove non-alphanumeric characters
    tokens = nltk.word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]
    return ' '.join(stemmed_tokens)

# Function to fetch data from the API
def fetch_data_from_api(app_name):
    api_url=f"http://127.0.0.1:5000/{app_name}"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        questions = [item['question'] for item in data]  # Assuming your API returns a list of dicts with 'question' and 'answer'
        answers = [item['answer'] for item in data]
        return questions, answers
    else:
        print(f"Error: Unable to fetch data from API. Status code: {response.status_code}")
        return None, None

# Create TF-IDF vectorizer and fit on the data
def create_vectorizer(questions_list):
    vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
    X = vectorizer.fit_transform([preprocess(q) for q in questions_list])
    return vectorizer, X

# Function to get response based on input text
def get_response(text, questions_list, answers_list, vectorizer, X):
    processed_text = preprocess_with_stopwords(text)
    vectorized_text = vectorizer.transform([processed_text])
    similarities = cosine_similarity(vectorized_text, X)
    max_similarity = np.max(similarities)
    if max_similarity > 0.6:
        high_similarity_questions = [q for q, s in zip(questions_list, similarities[0]) if s > 0.6]

        target_answers = []
        for q in high_similarity_questions:
            q_index = questions_list.index(q)
            target_answers.append(answers_list[q_index])

        Z = vectorizer.fit_transform([preprocess_with_stopwords(q) for q in high_similarity_questions])
        processed_text_with_stopwords = preprocess_with_stopwords(text)
        vectorized_text_with_stopwords = vectorizer.transform([processed_text_with_stopwords])
        final_similarities = cosine_similarity(vectorized_text_with_stopwords, Z)
        closest = np.argmax(final_similarities)
        return target_answers[closest]
    else:
        return "I can't answer this question."

# Usage
if __name__ == "__main__":
    api_url = input("Enter APP Name: ")  # Prompt the user to enter the API URL

    # Fetch data from the API
    questions_list, answers_list = fetch_data_from_api(api_url)
    print(questions_list)
    print(answers_list)

    if questions_list and answers_list:
        # Create vectorizer and fit it to the data
        vectorizer, X = create_vectorizer(questions_list)

        # Get user question
        question = input("Question: ")

        # Get the response
        response = get_response(question, questions_list, answers_list, vectorizer, X)
        print("Response:", response)
