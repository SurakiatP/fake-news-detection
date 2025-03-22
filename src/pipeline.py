import pandas as pd
from preprocessing import preprocess_dataframe
from vectorizer import TextVectorizer
import joblib

class FakeNewsPipeline:
    def __init__(self, model_path, vectorizer_path):
        """
        Initializes the pipeline by loading the trained model and vectorizer.
        """
        self.model = joblib.load(model_path)
        self.vectorizer = TextVectorizer()
        self.vectorizer.load(vectorizer_path)

    def predict(self, texts):
        """
        Makes predictions on a list or Series of raw text data.

        Args:
            texts (list or Series): Raw text data for predictions.

        Returns:
            list: Predicted labels.
        """
        processed_texts = [preprocess_dataframe(pd.DataFrame({'text':[text]}))['clean_text'].iloc[0] for text in texts]
        vectorized_texts = self.vectorizer.transform(processed_texts)
        predictions = self.model.predict(vectorized_texts)
        return predictions

    def predict_from_csv(self, filepath, text_column='text'):
        """
        Loads texts from a CSV file and predicts labels.

        Args:
            filepath (str): Path to CSV file containing texts.
            text_column (str): Column name containing texts.

        Returns:
            DataFrame: Original texts along with their predicted labels.
        """
        df = pd.read_csv(filepath)
        df = preprocess_dataframe(df, text_column)
        df['predicted_label'] = self.predict(df['clean_text'])
        return df


# Example usage
if __name__ == '__main__':
    pipeline = FakeNewsPipeline(
        model_path='../models/LogisticRegression_best_model.pkl',
        vectorizer_path='../models/tfidf_vectorizer.pkl'
    )

    predictions_df = pipeline.predict_from_csv('../data/test.csv')
    predictions_df.to_csv('../data/pipeline_predictions.csv', index=False)
    print("Pipeline predictions saved successfully.")
