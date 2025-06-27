import sys
import joblib
from bs4 import BeautifulSoup
import re

def clean_text(text):
    """Remove HTML tags and special characters from text."""
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower().strip()

def predict_sentiment(review_text):
    try:
        # Load model and vectorizer
        model = joblib.load('models/model.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        
        # Clean input text
        review_text = clean_text(review_text)
        
        # Transform and predict
        review_tfidf = vectorizer.transform([review_text])
        prediction = model.predict(review_tfidf)[0]
        confidence = model.predict_proba(review_tfidf)[0]
        label = 'positive' if prediction == 1 else 'negative'
        confidence_score = max(confidence)
        
        return label, confidence_score
    except Exception as e:
        return f"Error: {str(e)}", 0.0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py 'Your review text'")
        sys.exit(1)
    
    review_text = sys.argv[1]
    label, confidence = predict_sentiment(review_text)
    print(f"Sentiment: {label}, Confidence: {confidence:.4f}")