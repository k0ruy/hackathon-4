import os

import pandas as pd
import numpy as np

from src.nlp.vader_lexicon import *


def sentiment_nyt():
    for file in sorted(os.listdir("data/nyt_data")):
        f = os.path.join("data/nyt_data", file)
        # checking if it is a file
        if os.path.isfile(f):
            df = pd.read_csv(f)
            for idx, srs in df.iterrows():
                txt = srs.CleanedText
                # print(txt)
                df["SentimentScore"] = sentiment_analysis(txt)
            print(f"saving to csv file {f}")
            df.to_csv(f, index=False)


def sentiment_reddit():
    for file in sorted(os.listdir("src/data_collection/data")):
        f = os.path.join("src/data_collection/data", file)
        # checking if it is a file
        if os.path.isfile(f):
            df = pd.read_csv(f)
            for idx, srs in df.iterrows():
                txt = srs.Title
                # print(txt)
                df = df.append(sentiment_analysis(txt), ignore_index=True)
            print(f"saving to csv file {f}")
            df.to_csv(f, index=False)


if __name__ == '__main__':
    sentiment_reddit()
