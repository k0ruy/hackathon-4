import os
import os.path
from src.nlp.vader_lexicon import sentiment_analysis
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


def sentiment_bbc():
    for file in sorted(os.listdir("data/bbc_data")):
        f = os.path.join("data/bbc_data", file)
        # checking if it is a file
        if os.path.isfile(f):
            temp = pd.read_csv(f, encoding_errors="ignore")
            temp["row_id"] = temp.index
            if 'row_id' in temp.columns:
                temp.reset_index(inplace=True)
                sentiment_analysis(file=temp, column="content", name=f)


def sentiment_nyt():
    for file in sorted(os.listdir("data/nyt_data")):
        f = os.path.join("data/nyt_data", file)
        # checking if it is a file
        if os.path.isfile(f):
            temp = pd.read_csv(f, encoding_errors="ignore")
            temp["row_id"] = temp.index
            if 'row_id' in temp.columns:
                temp.reset_index(inplace=True)
                sentiment_analysis(file=temp, column="CleanedText", name=f)


def sentiment_reddit():
    for file in sorted(os.listdir("data")):
        f = os.path.join("data", file)
        # checking if it is a file
        if os.path.isfile(f):
            if os.path.isfile(f):
                if not f.endswith("DS_Store"):
                    sentiment_analysis(file=f, column="Title")


if __name__ == '__main__':
    # sentiment_nyt()
    sentiment_bbc()
    # sentiment_reddit()
