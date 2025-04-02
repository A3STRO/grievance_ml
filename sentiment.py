from typing import Tuple, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import clean_text

# Sample sentiment data from Consumer Complaint Database
SAMPLE_SENTIMENTS = [
    # Positive sentiments
    "Exceptional service! Issue resolved immediately",
    "Extremely satisfied with the quick resolution",
    "Outstanding support team, solved everything perfectly",
    "Excellent handling of my complaint, very professional",
    "Incredibly helpful staff, exceeded expectations",
    "Fantastic response time and resolution",
    "Perfect handling of my case, very impressed",
    "Brilliant service recovery, totally satisfied",
    
    # Neutral sentiments
    "Need information about my account status",
    "Requesting details about the transaction fees",
    "When will my application be processed",
    "Please review my account statement",
    "What documents are required for this process",
    "How long does the verification take",
    "Need to update my contact information",
    "Checking the status of my request",
    
    # Negative sentiments
    "Absolutely outraged by the terrible service!",
    "Completely unacceptable response to my urgent issue",
    "Horrendous experience, worst service ever received",
    "Extremely frustrated with total lack of support",
    "Utterly disappointed, demanding immediate action",
    "Severely impacted by your negligent handling",
    "Totally incompetent staff, wasted hours of my time",
    "Disastrous customer service, considering legal action"
]

# Corresponding labels for the sample sentiments
SAMPLE_LABELS = [
    "positive", "positive", "positive", "positive", "positive", "positive", "positive", "positive",
    "neutral", "neutral", "neutral", "neutral", "neutral", "neutral", "neutral", "neutral",
    "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative"
]

def train_sentiment_model() -> Tuple[LogisticRegression, TfidfVectorizer]:
    """
    Train a Logistic Regression model for sentiment analysis using sample data.
    
    Returns:
        tuple: (trained_model, vectorizer)
    """
    try:
        # Initialize TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Preprocess all training texts
        processed_texts = [clean_text(text) for text in SAMPLE_SENTIMENTS]
        
        # Create feature vectors
        X = vectorizer.fit_transform(processed_texts)
        y = np.array(SAMPLE_LABELS)
        
        # Initialize and train the classifier
        classifier = LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        classifier.fit(X, y)
        
        print("Sentiment analysis model trained successfully!")
        return classifier, vectorizer
    
    except Exception as e:
        print(f"Error in training sentiment model: {str(e)}")
        raise

def predict_sentiment(text: str, classifier: LogisticRegression, vectorizer: TfidfVectorizer) -> Tuple[str, float]:
    """
    Predict the sentiment of a given grievance text.
    
    Args:
        text (str): Input grievance text
        classifier (LogisticRegression): Trained classifier
        vectorizer (TfidfVectorizer): Fitted vectorizer
        
    Returns:
        tuple: (predicted_sentiment, confidence_score)
    """
    try:
        # Preprocess the input text
        processed_text = clean_text(text)
        
        # Transform text to feature vector
        X = vectorizer.transform([processed_text])
        
        # Predict sentiment
        predicted_sentiment = classifier.predict(X)[0]
        
        # Get prediction probability
        proba = classifier.predict_proba(X)[0]
        confidence = max(proba)
        
        return predicted_sentiment, confidence
    
    except Exception as e:
        print(f"Error in predicting sentiment: {str(e)}")
        raise

def get_sentiment_priority(sentiment: str, confidence: float) -> str:
    """
    Determine priority level based on sentiment and confidence score.
    
    Args:
        sentiment (str): Predicted sentiment
        confidence (float): Confidence score of the prediction
        
    Returns:
        str: Priority level and explanation
    """
    if sentiment == "negative" and confidence > 0.45:
        return "HIGH PRIORITY: Immediate attention required due to strong negative sentiment"
    elif sentiment == "negative" and confidence > 0.35:
        return "MEDIUM PRIORITY: Address soon due to negative sentiment"
    elif sentiment == "positive" and confidence > 0.4:
        return "LOW PRIORITY: Positive feedback, respond within standard timeframe"
    else:
        return "NORMAL PRIORITY: Address within standard response time"
