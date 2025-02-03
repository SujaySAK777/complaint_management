# models/nlp.py

import os
import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')

def preprocess_text(text):
    """
    Preprocesses the complaint text by:
    - Lowercasing
    - Tokenizing
    - Removing stopwords
    - Lemmatizing
    """
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

def categorize_complaint(text):
    """
    Categorizes the complaint based on predefined keywords and returns the associated department and category.
    """
    categories = {
        "Sanitation": ["garbage", "trash", "waste", "overflowing"],
        "Water Supply": ["water", "leak", "pipe", "drain"],
        "Infrastructure": ["road", "pothole", "bridge", "broken"],
        "Public Safety": ["crime", "dangerous", "theft", "safety"]
    }

    # Mapping categories to departments and their respective IDs
    category_to_department = {
        "Sanitation": ("Sanitation", 1),
        "Water Supply": ("Water", 2),
        "Infrastructure": ("Infrastructure", 3),
        "Public Safety": ("Safety", 4)
    }

    for category, keywords in categories.items():
        if any(keyword in text.lower() for keyword in keywords):
            return category_to_department[category]  # Return both department name and department ID

    return ("General", 5)  # Default category and department for general complaints

def assign_priority(text):
    """
    Assigns priority to the complaint text based on:
    - Keywords in the complaint
    - Sentiment analysis
    """
    # Priority based on predefined keywords
    high_priority_keywords = ['urgent', 'dangerous', 'critical', 'leaking', 'broken', 'serious']
    for keyword in high_priority_keywords:
        if keyword in text.lower():
            return "HIGH"
    
    # Sentiment-based priority assignment
    sentiment, sentiment_priority = analyze_sentiment(text)
    
    # Return sentiment-based priority if no keyword priority is found
    if sentiment_priority == "HIGH":
        return "HIGH"
    elif sentiment_priority == "MEDIUM":
        return "MEDIUM"
    else:
        return "LOW"

def analyze_sentiment(text):
    """
    Analyzes sentiment of the complaint text and assigns priority.
    """
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    
    if sentiment['compound'] == 0.0:
        priority = "LOW"
    elif sentiment['compound'] < -0.5:
        priority = "HIGH"
    elif -0.5 <= sentiment['compound'] <= 0.5:
        priority = "MEDIUM"
    else:
        priority = "LOW"
    
    return sentiment, priority 
