# first line: 13
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
