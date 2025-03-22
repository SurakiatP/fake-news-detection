from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

class TextVectorizer:
    def __init__(self, max_features=5000, stop_words='english'):
        """
        Initializes the TextVectorizer class with a TF-IDF Vectorizer.

        Args:
            max_features (int): The maximum number of features to keep.
            stop_words (str or list): Stop words to be removed from text.
        """
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)

    def fit_transform(self, texts):
        """
        Fits the vectorizer to the texts and transforms the texts into vectors.

        Args:
            texts (list or Series): Text data to fit and transform.

        Returns:
            sparse matrix: TF-IDF vectors.
        """
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        """
        Transforms new texts into vectors using the previously fitted vectorizer.

        Args:
            texts (list or Series): Text data to transform.

        Returns:
            sparse matrix: TF-IDF vectors.
        """
        return self.vectorizer.transform(texts)

    def save(self, filepath):
        """
        Saves the fitted vectorizer to a file for later use.

        Args:
            filepath (str): Path where the vectorizer will be saved.
        """
        joblib.dump(self.vectorizer, filepath)

    def load(self, filepath):
        """
        Loads a previously saved vectorizer from a file.

        Args:
            filepath (str): Path from where the vectorizer will be loaded.
        """
        self.vectorizer = joblib.load(filepath)