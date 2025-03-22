# Summary Report - Fake News Detection Project

## Project Overview

The goal of this project was to build an effective Fake News Detection model using Natural Language Processing (NLP) techniques and machine learning algorithms. The dataset utilized contained news articles labeled as either fake or real.

## Exploratory Data Analysis (EDA)

### Key Findings:
- **Distribution of Labels:**
  - Real news and fake news labels were balanced.

- **Missing Values:**
  - Some missing values were detected primarily in the `author` column, which required handling during preprocessing.

- **News Text Length:**
  - Most articles had a length between 1,000 and 5,000 characters, with some outliers exceeding 10,000 characters.

## Preprocessing

### Methods Applied:
- Lowercasing
- Removal of punctuation and numerical characters
- Stopword removal (English)
- Lemmatization using spaCy for dimensionality reduction

These steps significantly reduced noise in the text data and enhanced feature extraction.

## Model Training and Selection

Three machine learning models were trained and evaluated using cross-validation:

- **Logistic Regression:**
  - Best Parameters: `C=10`
  - Best Cross-Validation Accuracy: ~94%

- **Random Forest:**
  - Best Parameters: `n_estimators=200, max_depth=None`
  - Best Cross-Validation Accuracy: ~94%

- **Multinomial Naive Bayes:**
  - Best Parameters: `alpha=0.1`
  - Best Cross-Validation Accuracy: ~88%

### Best Model:
**Logistic Regression** was selected as the best-performing model.

## Evaluation

The Logistic Regression model showed strong predictive performance on validation data:

- **Accuracy:** ~94%
- **Precision and Recall:** Balanced with good precision and recall across both labels.
- **Confusion Matrix:**
  - Low false-positive and false-negative rates.

## Conclusion and Next Steps

### Conclusions:
- Logistic Regression is highly effective for this text classification task.
- Proper text preprocessing significantly impacts model performance.

### Future Recommendations:
- Explore deep learning models (e.g., LSTM, BERT) for potentially higher accuracy.
- Expand dataset size to increase model robustness.
- Deploy the trained model into a real-time web-based application for practical usage.

