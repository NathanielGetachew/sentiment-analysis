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