import os
import os.path
from src.nlp.vader_lexicon import sentiment_analysis
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def sentiment_nyt():
    for file in sorted(os.listdir("data/nyt_data/cl")):
        f = os.path.join("data/nyt_data/cl", file)
        # checking if it is a file
        if os.path.isfile(f):
            temp = pd.read_csv(f, encoding_errors="ignore")
            temp.rename(columns={temp.columns[0]: "row_id"}, inplace=True)
            if 'row_id' in temp.columns:
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
    sentiment_nyt()
    # sentiment_reddit()
