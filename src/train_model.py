import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing import preprocess_dataframe
from vectorizer import TextVectorizer

def load_data(filepath, text_column='text', label_column='label'):
    df = pd.read_csv(filepath)
    df = preprocess_dataframe(df, text_column)
    return df['clean_text'], df[label_column]

def train_and_select_model(X, y):
    """Trains and selects the best model using GridSearchCV."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TextVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(),
        "MultinomialNB": MultinomialNB()
    }

    param_grid = {
        "LogisticRegression": {"C": [0.01, 0.1, 1, 10]},
        "RandomForest": {"n_estimators": [50, 100], "max_depth": [None, 10]},
        "MultinomialNB": {"alpha": [0.1, 0.5, 1.0]}
    }

    best_score = 0
    best_model = None
    best_model_name = ""
    model_accuracies = {}

    for model_name, model in models.items():
        grid = GridSearchCV(model, param_grid[model_name], cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train_vec, y_train)

        print(f"{model_name} best params: {grid.best_params_}")
        print(f"{model_name} best accuracy: {grid.best_score_}")

        model_accuracies[model_name] = grid.best_score_

        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
            best_model_name = model_name

    print(f"\nBest model selected: {best_model_name}")

    # Evaluate best model
    y_pred = best_model.predict(X_test_vec)
    class_report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Classification Report:")
    print(class_report)

    # Save reports directory
    os.makedirs('../reports', exist_ok=True)

    # Save classification report
    with open('../reports/classification_report.txt', 'w') as f:
        f.write(f"Best Model: {best_model_name}\n\n")
        f.write(class_report)

    # Save confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.savefig('../reports/confusion_matrix.png')
    plt.close()

    # Plot model comparison
    model_names = list(model_accuracies.keys())
    scores = list(model_accuracies.values())
    plt.figure(figsize=(8, 5))
    sns.barplot(x=model_names, y=scores)
    plt.title('Accuracy Comparison of Models')
    plt.ylabel('Cross-Validated Accuracy')
    plt.xlabel('Model')
    plt.ylim(0, 1)
    plt.savefig('../reports/accuracy_comparison.png')
    plt.close()

    # Save best model and vectorizer
    joblib.dump(best_model, f'../models/{best_model_name}_best_model.pkl')
    vectorizer.save('../models/tfidf_vectorizer.pkl')

    return best_model, vectorizer

if __name__ == '__main__':
    X, y = load_data('../data/train.csv')
    train_and_select_model(X, y)
