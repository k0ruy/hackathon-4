# source: https://towardsdatascience.com/how-to-build-and-deploy-an-nlp-model-with-fastapi-part-1-9c1c7030d40

# import important modules
import numpy as np
import pandas as pd
# sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB # classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# text preprocessing modules
from string import punctuation
# text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression
from act_learning import check_preds

# Download dependency
for dependency in (
        "brown",
        "names",
        "wordnet",
        "averaged_perceptron_tagger",
        "universal_tagset",
):
    nltk.download(dependency)

import warnings

warnings.filterwarnings("ignore")

stop_words = stopwords.words('english')


def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r'http\S+', ' link ', text)
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text)  # remove numbers

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])

    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)

    # Return a list of words
    return text


def classify(df):
    PE_Conditions = [
        (df['sentiment_score'] < -0.4),
        (df['sentiment_score'] >= 0.4) & (df['sentiment_score'] < -0.4),
        (df['sentiment_score'] >= 0.4)
    ]
    PE_Categories = [-1, 0, 1]
    df['sentiment'] = np.select(PE_Conditions, PE_Categories)

    return df


if __name__ == '__main__':
    # load data

    folder = "complete"
    data = pd.read_csv(f"../../data/{folder}/CL_BBC_Bahrainnyt_complete.csv")  # be sure to change also file name depending on folder

    data = classify(data)

    # clean the review
    if folder == "reddit_data":
        data["clean_title"] = data["Title"].apply(text_cleaning)
    elif folder == "bbc_data":
        data["clean_title"] = data["content"].apply(text_cleaning)
    elif folder == "nyt_data":
        data["clean_title"] = data["CleanedText"].apply(text_cleaning)
    elif folder == "complete":
        data["clean_title"] = data["CleanedText"].apply(text_cleaning)

    # split features and target from  data
    X = data["clean_title"]
    y = data.sentiment.values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42,
        shuffle=True,
        stratify=y,
    )

    sentiment_classifier = Pipeline(steps=[
        ('pre_processing', TfidfVectorizer(lowercase=False)),
        ('rf', RandomForestClassifier())
    ])

    sentiment_classifier.fit(X_train, y_train)

    pred_df = pd.DataFrame()
    pred_df["y_preds"] = sentiment_classifier.predict(X_test)
    pred_df['pred_proba'] = list(sentiment_classifier.predict_proba(X_test))

    pred_df['annotate_decision'] = pred_df['pred_proba'].apply(check_preds)

    print(pred_df.annotate_decision.value_counts())

    accuracy_score = accuracy_score(y_test, pred_df["y_preds"])
    print(f"{accuracy_score=}")

