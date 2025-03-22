import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from preprocessing import preprocess_dataframe
from vectorizer import TextVectorizer
import os


def load_test_data(filepath, text_column='text', label_column=None):
    df_test = pd.read_csv(filepath)
    df_test = preprocess_dataframe(df_test, text_column)
    X_test = df_test['clean_text']
    y_test = df_test[label_column] if label_column else None
    return X_test, y_test


def evaluate_model(model_path, vectorizer_path, test_filepath, text_column='text', label_column='label'):
    # Load the trained model and vectorizer
    model = joblib.load(model_path)
    vectorizer = TextVectorizer()
    vectorizer.load(vectorizer_path)

    # Load and preprocess test data
    X_test, y_test = load_test_data(test_filepath, text_column, label_column)
    X_test_vec = vectorizer.transform(X_test)

    # Predict test data
    predictions = model.predict(X_test_vec)

    # If true labels are provided, evaluate model performance
    if y_test is not None:
        print("No label column provided — skipping evaluation report and confusion matrix.")
        accuracy = accuracy_score(y_test, predictions)
        class_report = classification_report(y_test, predictions)

        print(f"Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(class_report)

        # Save classification report
        os.makedirs('../reports', exist_ok=True)
        with open('../reports/classification_report.txt', 'w') as f:
            f.write(f"Accuracy: {accuracy:.2f}\n\n")
            f.write(class_report)

        # Confusion Matrix visualization
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.savefig('../reports/confusion_matrix.png')
        plt.show()
    elif y_test is None:
        print("No label column provided — skipping evaluation report and confusion matrix.")

    # Return predictions as DataFrame
    results_df = pd.DataFrame({'text': X_test, 'predicted_label': predictions})
    return results_df


if __name__ == '__main__':
    results = evaluate_model(
        model_path='../models/LogisticRegression_best_model.pkl',
        vectorizer_path='../models/tfidf_vectorizer.pkl',
        test_filepath='../data/test.csv',
        text_column='text',
        label_column=None  # set to actual label column if available
    )

    # Save predictions
    results.to_csv('../data/test_predictions.csv', index=False)
    print("Predictions saved to '../data/test_predictions.csv'")
