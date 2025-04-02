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

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Process with spaCy
        doc = nlp(text)
        
        # Remove stopwords and lemmatize
        tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        
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
        # Add any custom feature extraction logic here
        # For now, we'll just return the cleaned text
        return clean_text(text)
    
    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        raise