import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

import joblib
import sys

# Removed google.colab imports as they are not needed for local execution
# from google.colab import files

def main():
    try:
        print("Loading dataset...")
        df = pd.read_csv("text_classification_dataset.csv")
    except FileNotFoundError:
        print("Error: 'text_classification_dataset.csv' not found.")
        print("Please ensure the dataset is in the same directory.")
        sys.exit(1)

    print("Dataset loaded successfully.")
    print(df.head())
    print(df.shape)
    df.info()
    print(df.isnull().sum())

    # Preprocessing
    df['sentiment'] = df['sentiment'].map({'positive': 1,'negative': 0})
    
    # Visualizations
    # Note: plt.show() will open a window. Close it to proceed with the script.
    print("Generating visualizations...")
    sns.countplot(x='sentiment', data=df)
    plt.title("Sentiment Distribution")
    # plt.show() # Commented out to prevent blocking, uncomment if you want to see plots

    df['text_length'] = df['text'].apply(len)

    plt.figure()
    plt.hist(df['text_length'])
    plt.xlabel("Text Length")
    plt.ylabel("Frequency")
    plt.title("Distribution of Text Lengths")
    # plt.show()

    plt.figure()
    sns.boxplot(x=df['text_length'])
    plt.title("Outlier Detection using Boxplot")
    # plt.show()

    # Splitting Data
    X = df['text']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorization
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1,3),
        min_df=2,
        max_df=0.9
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # LinearSVC Model
    print("Training LinearSVC...")
    svm = LinearSVC(max_iter=10000, dual=True) # dual=True is default but explicit is fine, suppressing future warnings might be needed depending on sklearn version

    # Filter out NaN values from y_train and corresponding X_train_tfidf if any
    if y_train.isna().any():
        nan_mask = y_train.isna()
        X_train_tfidf_cleaned = X_train_tfidf[~nan_mask.values]
        y_train_cleaned = y_train[~nan_mask]
        svm.fit(X_train_tfidf_cleaned, y_train_cleaned)
    else:
        svm.fit(X_train_tfidf, y_train)

    y_pred = svm.predict(X_test_tfidf)

    print("LinearSVC Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # GridSearch
    print("Starting GridSearchCV...")
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'max_iter': [1000, 5000, 10000]
    }

    grid = GridSearchCV(
        LinearSVC(dual=False),
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    grid.fit(X_train_tfidf, y_train)

    print("Best parameters:", grid.best_params_)
    print("Best F1 score:", grid.best_score_)

    best_svm = grid.best_estimator_
    best_svm.fit(X_train_tfidf, y_train)

    best_pred = best_svm.predict(X_test_tfidf)

    print("Optimized Accuracy:", accuracy_score(y_test, best_pred))
    print(classification_report(y_test, best_pred))

    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_tfidf, y_train)
    rf_pred = rf.predict(X_test_tfidf)
    print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

    # Manual Test
    manual_text = ["Not worth the price"]
    manual_tfidf = tfidf.transform(manual_text)
    prediction = svm.predict(manual_tfidf)
    label_map = {0: "Negative", 1: "Positive"}
    print(f"Text: '{manual_text[0]}', Predicted sentiment: {label_map[prediction[0]]}")

    # Saving Models
    print("Saving models...")
    joblib.dump(best_svm, "svm_text_classifier.pkl")
    joblib.dump(tfidf, "tfidf_vectorizer.pkl")
    print("Models saved: svm_text_classifier.pkl, tfidf_vectorizer.pkl")

    # Final Test
    manual_text_2 = ["Waste of money"]
    manual_tfidf_2 = tfidf.transform(manual_text_2)
    prediction_2 = svm.predict(manual_tfidf_2)
    print(f"Text: '{manual_text_2[0]}', Predicted sentiment: {label_map[prediction_2[0]]}")

if __name__ == "__main__":
    main()
