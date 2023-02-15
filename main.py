import os
from src.nlp.vader_lexicon import sentiment_analysis
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def sentiment_nyt():
    for file in sorted(os.listdir("data/nyt_data")):
        f = os.path.join("data/nyt_data", file)
        # checking if it is a file
        if os.path.isfile(f):
            temp = pd.read_csv(f)
            if 'index' in temp.columns:
                temp['row_id'] = temp['index'].copy()
                temp = temp.iloc[:, 3:]
            temp.to_csv(f, index=False)
            sentiment_analysis(f, column="CleanedText")


def sentiment_reddit():
    for file in sorted(os.listdir("src/data_collection/data")):
        f = os.path.join("src/data_collection/data", file)
        # checking if it is a file
        if os.path.isfile(f):
            if os.path.isfile(f):
                sentiment_analysis(f, column="Title")


if __name__ == '__main__':
    # sentiment_nyt()
    sentiment_reddit()
