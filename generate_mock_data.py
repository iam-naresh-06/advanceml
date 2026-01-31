import pandas as pd
import random

def generate_mock_data(filename="text_classification_dataset.csv", num_rows=100):
    sentiments = ['positive', 'negative']
    texts = [
        "This product was absolutely amazing! I loved every bit of it.",
        "Terrible service, waste of time and money.",
        "It was okay, not the best but not the worst.",
        "Great quality and a wonderful experience.",
        "I was bored halfway through. Uninteresting.",
        "A masterpiece of modern engineering.",
        "The content was weak and the delivery was poor.",
        "Highly recommended for everyone.",
        "Do not buy this item, it is awful.",
        "Pretty good, I enjoyed it."
    ]

    data = {
        'text': [random.choice(texts) for _ in range(num_rows)],
        'sentiment': [random.choice(sentiments) for _ in range(num_rows)]
    }

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Mock data generated and saved to {filename}")

if __name__ == "__main__":
    generate_mock_data()
