# Grievance Redressal Platform - ML Module

A Machine Learning-enabled grievance redressal system for startups to address and resolve issues efficiently. This module uses classification and sentiment analysis to categorize grievances, analyze their sentiment, and recommend resolution pathways.

## Features

- **Grievance Classification**: Automatically categorizes grievances into predefined categories (billing, technical, HR)
- **Sentiment Analysis**: Analyzes the sentiment of grievances to determine priority
- **Resolution Recommendations**: Provides structured resolution pathways based on grievance category
- **Priority Assessment**: Determines priority levels based on sentiment analysis
- **Enhanced Text Processing**: Uses NLTK and spaCy for advanced text preprocessing

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Download required NLTK resources:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```
4. Download spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

Run the main script to see the ML pipeline in action:
```bash
python main.py
```

This will process sample grievances and display:
- Predicted category
- Sentiment analysis results
- Recommended resolution pathway
- Priority assessment

## Project Structure

- `main.py`: Entry point for the ML pipeline
- `model.py`: Implements grievance classification using Logistic Regression
- `sentiment.py`: Handles sentiment analysis
- `utils.py`: Contains text preprocessing utilities
- `requirements.txt`: Lists all required Python packages

## Technical Details

### ML Models Used
- **Classification**: Logistic Regression with TF-IDF features
- **Sentiment Analysis**: Logistic Regression with TF-IDF features

### Text Processing
- Tokenization using NLTK
- Stop words removal
- Lemmatization using spaCy
- TF-IDF vectorization

### Categories
- Billing
- Technical
- HR

### Sentiment Classes
- Positive
- Neutral
- Negative

## Example Output

```
Grievance: "I've been charged twice for the same service last month"

Predicted Category: BILLING
Sentiment Analysis: NEGATIVE (confidence: 0.85)
Priority: HIGH PRIORITY
Resolution Pathway:
1. Review invoice and transaction history
2. Verify charges with billing department
3. Process refund/adjustment if needed
4. Update customer on resolution timeline
```

## Future Enhancements

1. Integration with a modern, stylistic UI
2. Extended category support
3. More sophisticated ML models
4. Real-time processing capabilities
5. Integration with ticketing systems

## Note

This is the ML module of the Grievance Redressal Platform. The current implementation uses sample data for demonstration. In a production environment, you would:
1. Use real historical data for training
2. Implement model persistence
3. Add API endpoints for integration
4. Include more robust error handling
5. Add monitoring and logging