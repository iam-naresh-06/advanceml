import pandas as pd
import random

def generate_mock_data(filename="movie_review_sentiment_analysis.csv", num_rows=100):
    sentiments = ['positive', 'negative']
    reviews = [
        "This movie was absolutely amazing! I loved every bit of it.",
        "Terrible film, waste of time and money.",
        "It was okay, not the best but not the worst.",
        "Great acting and a wonderful plot.",
        "I fell asleep halfway through. Boring.",
        "A masterpiece of modern cinema.",
        "The script was weak and the acting was wooden.",
        "Highly recommended for everyone.",
        "Do not watch this movie, it is awful.",
        "Pretty good, I enjoyed it."
    ]

    data = {
        'review': [random.choice(reviews) for _ in range(num_rows)],
        'sentiment': [random.choice(sentiments) for _ in range(num_rows)]
    }

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Mock data generated and saved to {filename}")

if __name__ == "__main__":
    generate_mock_data()
