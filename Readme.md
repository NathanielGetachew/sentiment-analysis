# Sentiment Analysis Mini-Project

---

This project implements a sentiment analysis pipeline for movie reviews, designed for the Python AI Developer role. It leverages **scikit-learn** for **Logistic Regression** with **TF-IDF vectorization**, includes a command-line prediction script, and provides a **Flask API endpoint** for easy integration.

## Features

* Trains on a 5,000-sample subset of the Hugging Face IMDb dataset (2,500 positive, 2,500 negative reviews).
* Preprocesses text to remove HTML tags and special characters, enhancing model robustness.
* Offers `predict.py` for convenient command-line predictions and a `/predict` Flask endpoint for production use.
* Achieves an accuracy of **0.8840** with balanced precision and recall (~0.88-0.90).

## Dataset

* **Source:** Hugging Face IMDb Dataset.
* **Subset:** 5,000 samples from the training split.
* **Format:** CSV with `review` (text) and `sentiment` (0=negative, 1=positive) columns.
* **Preprocessing:** Removes HTML tags (e.g., `<br />`) and various special characters.

---

## Installation

To get started with the project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd sentiment-analysis
    ```

2.  **Set up a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Dataset Preparation

Generate the dataset subset required for training:

```bash
python create_subset.py

This script will download the IMDb dataset via Hugging Face and save the 5,000-sample subset to data/imdb_subset.csv.

Training the Model
Before training, ensure data/imdb_subset.csv exists.

Bash

python train.py
This script performs text cleaning, trains a Logistic Regression model using TF-IDF with 5,000 features, and saves the trained model (model.pkl) and vectorizer (vectorizer.pkl) to the models/ directory.

You'll see output similar to this:

Accuracy: 0.8840
Classification Report:
            precision    recall  f1-score   support

negative       0.90      0.86      0.88       500
positive       0.87      0.91      0.89       500

accuracy                           0.88      1000
macro avg       0.88      0.88      0.88      1000
weighted avg       0.88      0.88      0.88      1000
Command-Line Predictions
You can quickly get a sentiment prediction from the command line:

Bash

python predict.py "This movie was fantastic!<br />Really loved it!"
Example Output:

Sentiment: positive, Confidence: 0.8844
API Usage
For production deployment, you can use the Flask API:

Start the Flask server:

Bash

python app.py
Send a POST request to the /predict endpoint:

Bash

curl -X POST -H "Content-Type: application/json" -d '{"review":"This movie was fantastic!"}' http://localhost:8000/predict
Example Response:

JSON

{"sentiment":"positive","confidence":0.8844}
Project Structure
.
├── data/
│   └── imdb_subset.csv
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
├── app.py
├── create_subset.py
├── predict.py
├── requirements.txt
├── train.py
└── README.md
data/imdb_subset.csv: The prepared dataset subset.

models/model.pkl: The trained Logistic Regression model.

models/vectorizer.pkl: The fitted TF-IDF vectorizer.

create_subset.py: Script to generate the dataset subset.

train.py: Script for training the sentiment analysis model.

predict.py: Script for command-line predictions.

app.py: Flask API server for sentiment predictions.

requirements.txt: Lists all project dependencies.

README.md: This documentation file.

