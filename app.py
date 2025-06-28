from flask import Flask, request, jsonify
import joblib
from bs4 import BeautifulSoup
import re
from model import ProgressLogisticRegression

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('models/model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

def clean_text(text):
    """Remove HTML tags and special characters from text."""
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower().strip()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        review = data.get('review', '')
        if not review:
            return jsonify({'error': 'No review provided'}), 400
        
        # Clean input text
        review = clean_text(review)
        
        # Predict sentiment
        review_tfidf = vectorizer.transform([review])
        prediction = model.predict(review_tfidf)[0]
        confidence = max(model.predict_proba(review_tfidf)[0])
        label = 'positive' if prediction == 1 else 'negative'
        
        return jsonify({
            'sentiment': label,
            'confidence': float(confidence)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)