from typing import Tuple, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import clean_text

# Sample grievance data from Consumer Complaint Database
SAMPLE_GRIEVANCES = [
    # Banking/Billing related
    "Unauthorized charges appeared on my account statement",
    "Bank charged excessive fees without prior notification",
    "Double payment was processed for my monthly bill",
    "Wrong interest rate applied to my loan account",
    "ATM withdrawal charged but money not dispensed",
    "Credit card payment was not posted to account",
    "Monthly service fee charged despite maintaining minimum balance",
    "Incorrect foreign transaction fees on my statement",
    
    # Technical issues
    "Unable to access online banking portal",
    "Mobile app keeps crashing during fund transfer",
    "Cannot reset password through the website",
    "Account balance not updating in real-time",
    "Error message when trying to view statements",
    "Two-factor authentication not working",
    "Payment gateway timeout during transaction",
    "Wrong account details showing in dashboard",
    
    # Account/Service related
    "Account closed without proper notification",
    "Delay in processing loan application",
    "Customer service representative was unhelpful",
    "Long wait times for support response",
    "Branch staff refused to help with account issues",
    "Incorrect information given about services",
    "Difficulty updating personal information",
    "Lost documents not properly handled"
]

# Corresponding labels for the sample grievances
SAMPLE_LABELS = [
    "billing", "billing", "billing", "billing", "billing", "billing", "billing", "billing",
    "technical", "technical", "technical", "technical", "technical", "technical", "technical", "technical",
    "service", "service", "service", "service", "service", "service", "service", "service"
]

def train_classification_model() -> Tuple[LogisticRegression, TfidfVectorizer]:
    """
    Train a Logistic Regression model for grievance classification using sample data.
    
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
        processed_texts = [clean_text(text) for text in SAMPLE_GRIEVANCES]
        
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
        
        print("Classification model trained successfully!")
        return classifier, vectorizer
    
    except Exception as e:
        print(f"Error in training classification model: {str(e)}")
        raise

def predict_category(text: str, classifier: LogisticRegression, vectorizer: TfidfVectorizer) -> str:
    """
    Predict the category of a given grievance text.
    
    Args:
        text (str): Input grievance text
        classifier (LogisticRegression): Trained classifier
        vectorizer (TfidfVectorizer): Fitted vectorizer
        
    Returns:
        str: Predicted category
    """
    try:
        # Preprocess the input text
        processed_text = clean_text(text)
        
        # Transform text to feature vector
        X = vectorizer.transform([processed_text])
        
        # Predict category
        predicted_category = classifier.predict(X)[0]
        
        # Get prediction probability
        proba = classifier.predict_proba(X)[0]
        confidence = max(proba)
        
        print(f"Category predicted with confidence: {confidence:.2f}")
        return predicted_category
    
    except Exception as e:
        print(f"Error in predicting category: {str(e)}")
        raise

def get_resolution_pathway(category: str) -> str:
    """
    Get recommended resolution pathway based on grievance category.
    
    Args:
        category (str): Predicted grievance category
        
    Returns:
        str: Recommended resolution pathway
    """
    resolution_pathways = {
        'billing': """
            1. Document the billing discrepancy
            2. Review transaction history and relevant charges
            3. Contact billing department for verification
            4. Process refund/adjustment if warranted
            5. Update customer on resolution timeline
        """,
        'technical': """
            1. Document technical issue details
            2. Collect error messages and system logs
            3. Escalate to technical support team
            4. Test proposed solution
            5. Follow up with user to confirm resolution
        """,
        'service': """
            1. Record service complaint details
            2. Review service history and documentation
            3. Escalate to appropriate department
            4. Implement service improvement measures
            5. Follow up with customer satisfaction check
        """
    }
    
    return resolution_pathways.get(category, "Category not recognized. Please review manually.")