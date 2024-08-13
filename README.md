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
- **Dataset:** https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
-  **Code:** https://github.com/adacersos/Basic-AI-Projects/blob/main/Sentiment%20analysis%20challenge%20..ipynb

# Basic-AI-Projects  
## MNIST Handwritten Digit Classification

### Overview
This project focuses on classifying handwritten digits using various machine learning techniques. The dataset used is the famous MNIST dataset, which contains 70,000 grayscale images of handwritten digits (0-9), each sized 28x28 pixels.

### Project Details

#### Data Exploration
- Loaded the MNIST dataset using TensorFlow/Keras.
- Visualized sample images from the dataset to understand the data structure.
- Checked the distribution of digits in the dataset to ensure an even representation.

#### Data Preprocessing
- Normalized the pixel values of the images by scaling them to the range [0, 1].
- Reshaped the image data to a 2D format suitable for input into machine learning models.
- Converted the labels to one-hot encoded vectors for classification tasks.

#### Modeling
- Explored various machine learning models:
  - **Logistic Regression**: Applied logistic regression to the flattened image data.
  - **Support Vector Machine (SVM)**: Used a linear SVM for classification.
  - **K-Nearest Neighbors (KNN)**: Implemented KNN with different values of K.
  - **Neural Networks**: Built a simple feedforward neural network.
  - **Convolutional Neural Networks (CNN)**: Developed a deep CNN model leveraging convolutional layers, pooling layers, and dropout for regularization.
- Evaluated model performance using accuracy, confusion matrices, and loss curves.
- Applied data augmentation techniques to increase the model's robustness and accuracy.

#### Conclusion
- Achieved high accuracy with most models, with the CNN model outperforming others.
- The CNN model showed excellent performance in recognizing handwritten digits with high precision.
- Further improvements could include experimenting with more complex architectures or fine-tuning hyperparameters.

### Copy
- **Dataset**: import directly (look at code)
- **Code**: (https://github.com/adacersos/Basic-AI-Projects/blob/main/MNIST%20Digit%20Classification.ipynb)](https://github.com/adacersos/Basic-AI-Projects/blob/main/Computer%20Vision..ipynb)
