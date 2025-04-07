from flask import Flask, request, jsonify, render_template
from model import train_classification_model, predict_category, get_resolution_pathway
from sentiment import train_sentiment_model, predict_sentiment, get_sentiment_priority

app = Flask(__name__)

# Initialize and train models
print("Initializing models...")
classifier, vectorizer = train_classification_model()
sentiment_classifier, sentiment_vectorizer = train_sentiment_model()
print("Models initialized successfully!")

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/process_grievance', methods=['POST'])
def process_grievance():
    """Process a grievance and return the analysis results"""
    try:
        # Get grievance text from request
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No grievance text provided'}), 400
            
        # Get category and confidence
        category = predict_category(text, classifier, vectorizer)
        
        # Get confidence score from classifier
        X = vectorizer.transform([text])
        proba = classifier.predict_proba(X)[0]
        confidence = max(proba)
        
        # Get resolution pathway
        resolution = get_resolution_pathway(category)
        
        # Analyze sentiment
        sentiment, sentiment_confidence = predict_sentiment(text, sentiment_classifier, sentiment_vectorizer)
        
        # Get priority
        priority = get_sentiment_priority(sentiment, sentiment_confidence)
        
        # Return results
        return jsonify({
            'text': text,
            'category': category,
            'confidence': float(confidence),  # Convert numpy float to Python float
            'resolution_pathway': resolution,
            'sentiment': sentiment,
            'sentiment_confidence': float(sentiment_confidence),  # Convert numpy float to Python float
            'priority': priority
        })
        
    except Exception as e:
        print(f"Error processing grievance: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Run the Flask app
    print("Starting server...")
    app.run(host='0.0.0.0', port=8000, debug=True)