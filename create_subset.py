from datasets import load_dataset
import pandas as pd

def create_imdb_subset(output_file, samples_per_class=2500):
    # Load IMDb dataset from Hugging Face
    dataset = load_dataset("imdb", split="train")
    
    # Filter positive and negative reviews
    pos_reviews = [x for x in dataset if x['label'] == 1][:samples_per_class]
    neg_reviews = [x for x in dataset if x['label'] == 0][:samples_per_class]
    
    # Combine and create DataFrame
    reviews = [x['text'] for x in pos_reviews + neg_reviews]
    sentiments = [1] * samples_per_class + [0] * samples_per_class
    df = pd.DataFrame({'review': reviews, 'sentiment': sentiments})
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved subset to {output_file}")

if __name__ == "__main__":
    create_imdb_subset("data/imdb_subset.csv")