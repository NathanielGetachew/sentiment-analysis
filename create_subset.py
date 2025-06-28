from datasets import load_dataset
import pandas as pd

def create_imdb_full(output_file):
    # Load IMDb dataset (train and test splits)
    train_dataset = load_dataset("imdb", split="train")
    test_dataset = load_dataset("imdb", split="test")
    
    # Combine train and test data
    reviews = [x['text'] for x in train_dataset] + [x['text'] for x in test_dataset]
    sentiments = [x['label'] for x in train_dataset] + [x['label'] for x in test_dataset]
    
    # Create DataFrame
    df = pd.DataFrame({'review': reviews, 'sentiment': sentiments})
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved full dataset (50,000 samples) to {output_file}")

if __name__ == "__main__":
    create_imdb_full("data/imdb_full.csv")