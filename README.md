# Basic-AI-projects

# Movie Review Sentiment Analysis

## Overview

This project predicts the sentiment of movie reviews using machine learning techniques. The dataset consists of 50,000 movie reviews labeled as "Positive" or "Negative".

## Project Details
### Data Exploration

- Loaded the dataset using Pandas.
- Inspected the data structure and identified two columns: 'review' and 'sentiment'.
- Removed duplicate reviews and ensured an even distribution of positive and negative sentiments.

### Data Preprocessing

- Cleaned the review text by removing punctuation and converting to lowercase.
- Tokenized the reviews and removed stop words using NLTK.
- Applied stemming to reduce words to their base form.
- Transformed the text data into numerical features using Word2Vec embeddings.

### Modeling

- Explored various machine learning models:
  - Logistic Regression
  - Linear Support Vector Classifier
  - K-Nearest Neighbors
  - Neural Networks
  - Convolutional Neural Networks
- Evaluated model performances and selected the best-performing models based on accuracy.
- Explored visualization techniques like confusion matrices and ROC curves for model evaluation.

### Conclusion

- Achieved reasonable accuracy with all models.
- Logistic Regression and Linear Support Vector Classifier performed the best.

### Copy
- Dataset: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
-  Code: https://github.com/adacersos/Basic-AI-Projects/blob/main/Sentiment%20analysis%20challenge%20..ipynb
