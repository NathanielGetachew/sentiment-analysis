import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from bs4 import BeautifulSoup
import re
import joblib
from tqdm import tqdm
from model import ProgressLogisticRegression

def clean_text(text):
    """Remove HTML tags and special characters from text."""
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower().strip()

def train_model():
    # Load data
    print("Loading dataset...")
    df = pd.read_csv('data/imdb_full.csv')
    print(f"Dataset size: {len(df)} samples")
    
    # Clean reviews with progress bar
    tqdm.pandas()
    df['review'] = df['review'].progress_apply(clean_text)
    reviews, sentiments = df['review'].values, df['sentiment'].values
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(reviews, sentiments, test_size=0.2, random_state=42)
    
    # Vectorize text with progress bar
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(tqdm(X_train, desc="Transforming train data"))
    X_test_tfidf = vectorizer.transform(tqdm(X_test, desc="Transforming test data"))
    
    # Train model with progress bar
    print("Training model...")
    model = ProgressLogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    print("Evaluating model...")
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