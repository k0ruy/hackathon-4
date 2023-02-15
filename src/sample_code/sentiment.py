import nltk

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def sentiment_analysis(sentence):
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(str(sentence))
