from typing import Tuple, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import clean_text

# Sample grievance data from Consumer Complaint Database
SAMPLE_GRIEVANCES = [
    # Technical issues
    "Mobile app keeps crashing during transactions causing financial loss",
    "Unable to access online banking portal after multiple attempts",
    "App crashes every time I try to make a payment",
    "Website is not loading properly and shows error messages",
    "Two-factor authentication system is not working",
    "Technical glitch caused failed transactions",
    "System errors preventing account access",
    "App performance issues causing transaction failures",
    "Login authentication problems on the platform",
    "Continuous app crashes during important transactions",
    
    # Billing issues
    "Unauthorized charges appeared on my account statement",
    "Double payment was processed for my monthly bill",
    "Wrong interest rate applied to my loan account",
    "Incorrect fees charged to my account",
    "Monthly service fee charged despite maintaining minimum balance",
    "Unexplained charges on my credit card",
    "Billing discrepancy in my last statement",
    "Overcharged for banking services",
    "Wrong transaction amount debited",
    "Multiple charges for single transaction",
    
    # Service issues
    "Customer service not responding to my complaints",
    "Long wait times for support resolution",
    "Poor handling of my service request",
    "Staff was unhelpful with my inquiry",
    "No response to multiple support tickets",
    "Delayed response from customer care",
    "Inadequate support for account issues",
    "Poor communication from service team",
    "Unresolved complaints despite following up",
    "Lack of proper customer service assistance"
]

# Corresponding labels for the sample grievances
SAMPLE_LABELS = (
    ["technical"] * 10 +
    ["billing"] * 10 +
    ["service"] * 10
)

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
        'technical': """
            1. Document technical issue details and error messages
            2. Collect system logs and crash reports
            3. Escalate to technical support team for investigation
            4. Implement and test solution
            5. Follow up with user to confirm resolution
        """,
        'billing': """
            1. Document billing discrepancy details
            2. Review transaction history and charges
            3. Contact billing department for verification
            4. Process refund/adjustment if warranted
            5. Update customer on resolution timeline
        """,
        'service': """
            1. Record service complaint details
            2. Review service history and documentation
            3. Escalate to customer service management
            4. Implement service improvement measures
            5. Follow up with customer satisfaction check
        """
    }
    
    return resolution_pathways.get(category, "Category not recognized. Please review manually.")