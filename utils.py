import re
import spacy
from typing import Text

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model...")
    import os
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')
except Exception as e:
    print(f"Error loading spaCy model: {str(e)}")
    nlp = None

# Define keyword dictionaries for feature enhancement
URGENCY_KEYWORDS = {
    'immediate', 'urgent', 'asap', 'emergency', 'critical', 'crucial',
    'demanding', 'serious', 'vital', 'important', 'pressing'
}

NEGATIVE_KEYWORDS = {
    'outraged', 'terrible', 'horrible', 'awful', 'unacceptable',
    'disappointed', 'frustrated', 'angry', 'furious', 'worst',
    'incompetent', 'useless', 'poor', 'bad', 'worst', 'negligent',
    'disaster', 'failure', 'refuse', 'denied', 'never', 'delay'
}

POSITIVE_KEYWORDS = {
    'excellent', 'great', 'wonderful', 'fantastic', 'outstanding',
    'perfect', 'brilliant', 'exceptional', 'amazing', 'superb',
    'satisfied', 'helpful', 'professional', 'efficient', 'resolved'
}

def clean_text(text: Text) -> Text:
    """
    Clean and preprocess text using spaCy.
    
    Args:
        text (str): Input text to be preprocessed
        
    Returns:
        str: Cleaned and preprocessed text
        
    Raises:
        ValueError: If input text is empty or not a string
    """
    try:
        # Validate input
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input must be a non-empty string")

        # Convert to lowercase
        text = text.lower()

        # Process with spaCy
        doc = nlp(text)
        
        # Remove stopwords and lemmatize
        tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        
        # Add keyword-based features
        features = []
        
        # Check for urgency keywords
        if any(word in text.lower() for word in URGENCY_KEYWORDS):
            features.append("urgent_matter")
            
        # Check for strong negative sentiment
        if any(word in text.lower() for word in NEGATIVE_KEYWORDS):
            features.append("strong_negative")
            
        # Check for strong positive sentiment
        if any(word in text.lower() for word in POSITIVE_KEYWORDS):
            features.append("strong_positive")
            
        # Add exclamation mark feature
        if '!' in text:
            features.append("exclamation")
            
        # Add multiple punctuation feature
        if re.search(r'[!?]{2,}', text):
            features.append("multiple_punctuation")
            
        # Add features to the cleaned text
        tokens.extend(features)
        
        # Join tokens back into text
        cleaned_text = " ".join(tokens)
        
        return cleaned_text

    except Exception as e:
        print(f"Error in text preprocessing: {str(e)}")
        raise

def extract_features(text: Text) -> Text:
    """
    Extract additional features from text that might be useful for classification.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with extracted features
    """
    try:
        return clean_text(text)
    
    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        raise