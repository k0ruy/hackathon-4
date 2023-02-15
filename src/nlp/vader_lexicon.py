import nltk

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


def sentiment_analysis(f, column: str):
    """
    adds sentiment analysis to the file passed
    :param column: the column to process
    :param f: the file to process
    :return: the file with added sentiment columns
    """

    df = pd.read_csv(f)
    # create a new data frame with "id" and "comment" fields
    df_subset = df[['row_id', column]].copy()
    # data clean-up
    # remove all non-aphabet characters
    df_subset[column] = df_subset[column].str.replace("[^a-zA-Z#]", " ")
    # covert to lower-case
    df_subset[column] = df_subset[column].str.casefold()

    df1 = pd.DataFrame()
    df1['row_id'] = ['99999999999']
    df1['sentiment_type'] = 'NA999NA'
    df1['sentiment_score'] = 0

    print(f'Processing sentiment analysis for file {f.split("/")[-1]}...')
    sid = SentimentIntensityAnalyzer()
    t_df = df1
    for index, row in df_subset.iterrows():
        scores = sid.polarity_scores(row[1])
        for key, value in scores.items():
            df1['row_id'] = row[0]
            df1['sentiment_type'] = key
            df1['sentiment_score'] = value
            if 'sentiment_type' in df or 'sentiment_score'  in df:
                continue
            t_df = t_df.append(df1)
    # remove dummy row with row_id = 99999999999
    t_df_cleaned = t_df[t_df.row_id != '99999999999']
    # remove duplicates if any exist
    t_df_cleaned.drop_duplicates(inplace=True)
    # only keep rows where sentiment_type = compound
    t_df_cleaned = t_df[t_df.sentiment_type == 'compound']

    # merge dataframes
    df_output = pd.merge(df, t_df_cleaned, on='row_id', how='inner')

    df_output.to_csv(f, index=False)
