import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from bs4 import BeautifulSoup
import re
import joblib

def clean_text(text):
    """Remove HTML tags and special characters from text."""
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower().strip()

def train_model():
    # Load data
    df = pd.read_csv('data/imdb_subset.csv')
    df['review'] = df['review'].apply(clean_text)  # Clean reviews
    reviews, sentiments = df['review'].values, df['sentiment'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(reviews, sentiments, test_size=0.2, random_state=42)
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train model
    model = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))
    
    # Save model and vectorizer
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    print("Model and vectorizer saved to models/")

if __name__ == "__main__":
    train_model()