import joblib
import logging
import os

# Setup logging for consistent and clear tracking of pipeline activities
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def save_model(model, filepath):
    """
    Saves a trained model to disk.

    Args:
        model: Trained machine learning model.
        filepath (str): Filepath to save the model.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    logging.info(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Loads a trained model from disk.

    Args:
        filepath (str): Filepath from where the model will be loaded.

    Returns:
        Loaded model.
    """
    if os.path.exists(filepath):
        model = joblib.load(filepath)
        logging.info(f"Model loaded from {filepath}")
        return model
    else:
        logging.error(f"Model file {filepath} does not exist.")
        raise FileNotFoundError(f"Model file {filepath} does not exist.")


def save_vectorizer(vectorizer, filepath):
    """
    Saves a text vectorizer to disk.

    Args:
        vectorizer: Fitted text vectorizer.
        filepath (str): Filepath to save the vectorizer.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(vectorizer, filepath)
    logging.info(f"Vectorizer saved to {filepath}")


def load_vectorizer(filepath):
    """
    Loads a text vectorizer from disk.

    Args:
        filepath (str): Filepath from where the vectorizer will be loaded.

    Returns:
        Loaded vectorizer.
    """
    if os.path.exists(filepath):
        vectorizer = joblib.load(filepath)
        logging.info(f"Vectorizer loaded from {filepath}")
        return vectorizer
    else:
        logging.error(f"Vectorizer file {filepath} does not exist.")
        raise FileNotFoundError(f"Vectorizer file {filepath} does not exist.")


def create_directory(dir_path):
    """
    Creates a directory if it doesn't exist.

    Args:
        dir_path (str): Directory path to be created.
    """
    os.makedirs(dir_path, exist_ok=True)
    logging.info(f"Directory {dir_path} created or already exists.")