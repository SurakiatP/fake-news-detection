import re
import spacy
from nltk.corpus import stopwords
from joblib import Memory

# Initialize caching mechanism for performance
memory = Memory(location='./cachedir', verbose=0)

# Load spaCy model once for efficiency
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
stop_words = set(stopwords.words('english'))

@memory.cache
def clean_text(text):
    """
    Cleans and preprocesses text by applying lowercasing, removing punctuation,
    numbers, stopwords, and lemmatization. Results are cached for performance.

    Args:
        text (str): Raw text data.

    Returns:
        str: Cleaned text.
    """
    # Lowercase the text
    text = text.lower()

    # Remove punctuation, special characters, and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # Lemmatize tokens using spaCy
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.lemma_.strip() != '']

    return " ".join(tokens)


def preprocess_dataframe(df, text_column='text'):
    """
    Applies efficient text cleaning function to an entire DataFrame column, utilizes caching.

    Args:
        df (DataFrame): Pandas DataFrame with raw text.
        text_column (str): Column name containing raw text.

    Returns:
        DataFrame: DataFrame with additional column 'clean_text'.
    """
    df = df.dropna(subset=[text_column]).copy()
    df['clean_text'] = df[text_column].apply(clean_text)
    return df