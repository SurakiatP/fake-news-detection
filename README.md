# Fake News Detection Project

## Project Overview

This project involves building a machine learning pipeline for detecting fake news using Natural Language Processing (NLP). The goal is to develop an efficient and accurate model capable of distinguishing between real and fake news.

---

## Project Scoping

### Goals
- Accurately classify news articles as fake or real.
- Build an end-to-end machine learning pipeline including preprocessing, feature engineering, model training, and evaluation.
- Clearly communicate findings through visualizations and reports.

### Data
- Dataset source: [Kaggle's Fake News Detection Competition](https://www.kaggle.com/c/fake-news/data).
- Contains labeled news articles including news titles, authors, article texts, and labels (real/fake).

### Analysis
- Initial Exploratory Data Analysis (EDA) to understand data distribution, missing values, and text characteristics.
- Text preprocessing to clean and prepare the data for modeling.

---

## Project Structure

```
📂 fake-news-detection
│── 📜 README.md
│── 📂 data
│── 📂 notebooks
│── 📂 src
│── 📂 models
│── 📂 reports
│── 📜 requirements.txt
```

### Components:
- **Jupyter Notebooks:** Exploratory Data Analysis (EDA) and initial modeling.
- **CSV Data Files:** train.csv, test.csv (obtained from Kaggle).
- **Python Scripts (`src`):** ETL, preprocessing, model training, evaluation, and pipeline creation.
- **Trained Models:** Saved in the `models` folder.
- **Reports and Visualizations:** Saved in the `reports` folder.

---

## Extract, Transform, and Load (ETL)
- Loaded data from multiple CSV files (`train.csv`, `test.csv`).
- Transformed data using pandas for handling missing values and cleaning text data.

---

## Feature Engineering
- Text preprocessing: stopword removal, punctuation removal, lemmatization.
- Feature extraction using TF-IDF vectorization.
- Split data into training, testing, and validation sets.

---

## Machine Learning Workflow

### Evaluated Models:
- Logistic Regression
- Random Forest
- Multinomial Naive Bayes

### Model Selection:
- Hyperparameter tuning with GridSearchCV.
- Best performing model: **Logistic Regression** (Accuracy ~94%).

---

## ML Pipeline
- Modularized and organized code using scikit-learn's Pipeline module.
- Streamlined preprocessing, vectorization, and prediction steps for easy deployment.

---

## Communicate Findings
- **Key findings:**
  - Logistic Regression showed the highest predictive accuracy.
  - Proper text preprocessing significantly improved model performance.
- Results visualized through confusion matrices and accuracy comparison charts.

---

## Conclusion and Next Steps

### Conclusions
- Successfully built an effective fake news detection model using NLP and classical ML models.
- Highlighted the importance of preprocessing and model tuning.

### Future Recommendations
- Explore advanced NLP models such as BERT or LSTM for accuracy improvements.
- Expand dataset for greater generalizability.
- Deploy the trained model in real-world applications such as web applications or news verification systems.

---

## Setup

### How to Run this Project

1. Clone this repository.

2. Set up the virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

4. Run training and evaluation scripts from the `src` folder:

```bash
python src/train_model.py
python src/evaluate.py
```

The results, including models, reports, and visualizations, will be saved in their respective folders.

---

## Project Author

| Name           | Contact Information                                                  |
|----------------|----------------------------------------------------------------------|
| **Surakiat P.** |                                                                      |
| 📧 Email       | [surakiat.0723@gmail.com](mailto:surakiat.0723@gmail.com)   |
| 🔗 LinkedIn    | [linkedin.com/in/surakiat](https://www.linkedin.com/in/surakiat-kansa-ard-171942351/)     |
| 🌐 GitHub      | [github.com/SurakiatP](https://github.com/SurakiatP)                 |

