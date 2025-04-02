from typing import Dict, Any
from model import train_classification_model, predict_category, get_resolution_pathway
from sentiment import train_sentiment_model, predict_sentiment, get_sentiment_priority

def process_grievance(text: str) -> Dict[str, Any]:
    """
    Process a grievance text through the ML pipeline.
    
    Args:
        text (str): Input grievance text
        
    Returns:
        dict: Processing results including category, sentiment, and recommendations
    """
    try:
        # Train or load models (in production, these would be pre-trained and loaded from disk)
        print("\nInitializing models...")
        classifier, vectorizer = train_classification_model()
        sentiment_classifier, sentiment_vectorizer = train_sentiment_model()
        
        print("\nProcessing grievance:", text)
        print("-" * 50)
        
        # Predict category
        category = predict_category(text, classifier, vectorizer)
        print(f"\nPredicted Category: {category.upper()}")
        
        # Get resolution pathway
        resolution = get_resolution_pathway(category)
        print("\nRecommended Resolution Pathway:")
        print(resolution)
        
        # Analyze sentiment
        sentiment, confidence = predict_sentiment(text, sentiment_classifier, sentiment_vectorizer)
        print(f"\nSentiment Analysis: {sentiment.upper()} (confidence: {confidence:.2f})")
        
        # Get priority based on sentiment
        priority = get_sentiment_priority(sentiment, confidence)
        print("\nPriority Assessment:")
        print(priority)
        
        # Return consolidated results
        return {
            "text": text,
            "category": category,
            "resolution_pathway": resolution,
            "sentiment": sentiment,
            "sentiment_confidence": confidence,
            "priority": priority
        }
        
    except Exception as e:
        print(f"\nError processing grievance: {str(e)}")
        raise

def main():
    """
    Main function to demonstrate the grievance processing pipeline.
    """
    print("=" * 50)
    print("Grievance Redressal Platform - ML Module")
    print("=" * 50)
    
    while True:
        print("\nEnter your grievance (or type 'exit' to quit):")
        user_input = input().strip()
        
        if user_input.lower() == 'exit':
            print("\nThank you for using the Grievance Redressal Platform!")
            break
        
        if not user_input:
            print("\nPlease enter a valid grievance text.")
            continue
            
        try:
            process_grievance(user_input)
            print("\n" + "=" * 50)
            
        except Exception as e:
            print(f"\nError in main execution: {str(e)}")
            print("\nPlease try again with a different input.")

if __name__ == "__main__":
    main()