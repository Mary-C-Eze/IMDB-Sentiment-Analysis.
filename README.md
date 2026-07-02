
# IMDB Sentiment Analysis (NLP Machine Learning Project)

## Project Overview
This project is part of a machine learning exercise for The Film Junky Union community.  
The goal is to build a model that classifies IMDB movie reviews as either **positive (1)** or **negative (0)** using natural language processing techniques.

The system is designed to support automated filtering and categorization of movie reviews.



## Objective
- Build a sentiment classification model for movie reviews
- Achieve an **F1 score ≥ 0.85**
- Compare multiple machine learning approaches
- Evaluate generalization on unseen custom reviews



## Dataset
- Source: IMDB movie reviews dataset (polarity labeled)
- Features:
  - `review`: raw text of movie review
  - `pos`: sentiment label (0 = negative, 1 = positive)
  - `ds_part`: train/test split indicator



## Workflow

### 1. Data Preprocessing
- Text cleaning and normalization
- Handling missing values
- Train/test split validation

### 2. Feature Engineering
- TF-IDF vectorization for text representation

### 3. Models Trained
- Logistic Regression (baseline + best performer)
- LightGBM
- spaCy + TF-IDF + Logistic Regression variant

### 4. Evaluation Metrics
- F1 Score
- ROC AUC
- Average Precision Score (APS)


## Results

| Model | Performance Summary |
|------|--------------------|
| TF-IDF + Logistic Regression | Best overall performance |
| spaCy + TF-IDF + Logistic Regression | Similar performance, no major gain |
| spaCy + TF-IDF + LightGBM | Slightly lower but meets requirement |



## Key Insight
- Simpler models (TF-IDF + Logistic Regression) can outperform more complex approaches.
- Sparse matrix representation is critical for memory efficiency in NLP pipelines.
- Converting TF-IDF to dense format caused memory crashes during training.


## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- LightGBM
- spaCy
- TF-IDF Vectorization
- NLP / Machine Learning


## Key Learnings
- Importance of sparse matrix handling in large NLP datasets
- Tradeoff between model complexity and performance
- Real-world debugging of memory issues in ML pipelines

---

## Author
AI/ML Project Portfolio
