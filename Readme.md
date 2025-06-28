# Sentiment Analysis Mini-Project

---

This project implements a sentiment analysis pipeline for movie reviews, built for the **Python AI Developer** role. It leverages **scikit-learn** for **Logistic Regression** with **TF-IDF vectorization**, includes a command-line prediction script, and provides a **Flask API endpoint** for easy integration.

## Features

* Trains on the **full 50,000-sample Hugging Face IMDb dataset** (25,000 positive, 25,000 negative reviews).
* Preprocesses text to remove HTML tags and special characters, enhancing model robustness.
* Includes **progress bars (tqdm)** for text cleaning, vectorization, and model training, improving user experience.
* Offers `predict.py` for convenient command-line predictions and a `/predict` Flask endpoint for production use.
* Achieves an accuracy of **0.8897** with balanced precision and recall (~0.88-0.90).

## Dataset

* **Source:** Hugging Face IMDb Dataset.
* **Size:** 50,000 samples (25,000 training, 25,000 testing; combined for training).
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

Generate the full dataset required for training:

```bash
python create_subset.py

## Training the Model
Before training, ensure data/imdb_full.csv exists.

Bash

python train.py
This script cleans text, trains a Logistic Regression model using TF-IDF with 10,000 features, and saves the trained model (model.pkl) and vectorizer (vectorizer.pkl) to the models/ directory. You'll see progress bars during text cleaning, vectorization, and model training.

You'll see output similar to this:

Accuracy: 0.8897
Classification Report:
            precision    recall  f1-score   support

negative       0.90      0.88      0.89      5055
positive       0.88      0.90      0.89      4945

accuracy                           0.89     10000
macro avg       0.89      0.89      0.89     10000
weighted avg       0.89      0.89      0.89     10000
Command-Line Predictions
You can quickly get sentiment predictions from the command line:

Bash

python predict.py "This movie was fantastic!<br />Really loved it!"
Example Output:

Sentiment: positive, Confidence: 0.9853
Bash

python predict.py "The plot was boring and predictable."
Example Output:

Sentiment: negative, Confidence: 0.9993
API Usage
For production deployment, you can use the Flask API:

Start the Flask server:

Bash

python app.py
Send a POST request to the /predict endpoint:

Bash

curl -X POST -H "Content-Type: application/json" -d '{"review":"This movie was fantastic!<br />Really loved it!"}' http://localhost:8000/predict
Example Response:

JSON

{"sentiment":"positive","confidence":0.9853}
Note: Use single quotes for JSON data in Bash to avoid issues with special characters like !.

## Project Structure
.
├── data/
│   └── imdb_full.csv
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
├── app.py
├── create_subset.py
├── predict.py
├── requirements.txt
├── train.py
├── model.py
└── README.md
├── Dockerfile
data/imdb_full.csv: The prepared full dataset.

models/model.pkl: The trained Logistic Regression model.

models/vectorizer.pkl: The fitted TF-IDF vectorizer.

create_subset.py: Script to generate the full dataset.

train.py: Script for training the sentiment analysis model with progress bars.

predict.py: Script for command-line predictions.

app.py: Flask API server for sentiment predictions.

model.py: Custom ProgressLogisticRegression class (for progress bars during training).

requirements.txt: Lists all project dependencies.

README.md: This documentation file.

Dockerfile: Docker containerization setup.

## Notes
This project utilizes scikit-learn for its simplicity and alignment with the job requirements.

Progress bars enhance the user experience significantly during long-running tasks like text cleaning, vectorization, and model training.

The text cleaning step is crucial for enhancing model performance by effectively handling HTML tags and other noisy characters.

The inclusion of a Flask API demonstrates skills beyond optional requirements, showcasing readiness for production environments.

Using the full 50,000-sample dataset helps maximize the model's accuracy and generalization capabilities.

Future Improvements
Hyperparameter Tuning: Explore and fine-tune hyperparameters for the Logistic Regression model (e.g., the C regularization parameter) to potentially improve accuracy further.

Cross-Validation: Implement k-fold cross-validation during training to ensure the model's robustness and generalization ability across different data splits.

Docker Containerization: A Dockerfile is already included in the repository, making it straightforward to containerize and deploy this application.