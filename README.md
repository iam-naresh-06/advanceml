# Movie Review Sentiment Analysis

This project implements a sentiment analysis model for movie reviews using Support Vector Machines (LinearSVC) and Random Forest classifiers. It processes text data using TF-IDF vectorization and predicts whether a review is positive or negative.

## Features

-   **Data Visualization**: Exploratory Data Analysis (EDA) including sentiment distribution and review length analysis.
-   **Text Preprocessing**: TF-IDF vectorization with n-grams.
-   **Model Training**:
    -   Linear Support Vector Classification (LinearSVC)
    -   Random Forest Classifier
-   **Hyperparameter Tuning**: GridSearchCV for optimizing LinearSVC parameters.
-   **Model Persistence**: Saves trained models (`svm_text_classifier.pkl`, `tfidf_vectorizer.pkl`) for future use.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/iam-naresh-06/Advanced-ML.git
    cd advanceml
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Setup
The project requires a dataset named `movie_review_sentiment_analysis.csv`.
If you don't have the real dataset, you can generate a mock dataset for testing:

```bash
python generate_mock_data.py
```

### 2. Train the Model
Run the training script to train the models and see the evaluation metrics:

```bash
python train_model.py
```

### 3. Output
The script will output:
-   Accuracy scores and Classification reports.
-   Best hyperparameters found by GridSearch.
-   Predictions for sample manual reviews.
-   Saved model files (.pkl) in the current directory.

## Dependencies

-   pandas
-   numpy
-   matplotlib
-   seaborn
-   scikit-learn
-   joblib
# ML
