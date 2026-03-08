from sklearn.datasets import fetch_20newsgroups
import re
import pickle
from typing import List
import config

def clean_text(text: str) -> str:
    """Basic cleaning: remove extra whitespace."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_and_clean() -> List[str]:
    """Load 20 Newsgroups, remove headers/footers/quotes, and clean."""
    print("Loading 20 Newsgroups dataset...")
    newsgroups = fetch_20newsgroups(
        subset='all',
        remove=('headers', 'footers', 'quotes'),
        shuffle=True,
        random_state=42
    )
    texts = [clean_text(doc) for doc in newsgroups.data]
    print(f"Loaded {len(texts)} documents.")
    return texts

if __name__ == "__main__":
    texts = load_and_clean()
    with open(config.TEXTS_FILE, "wb") as f:
        pickle.dump(texts, f)
    print(f"Saved cleaned texts to {config.TEXTS_FILE}")