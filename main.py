from scraper import *
from sentiment import sentiment_analysis


if __name__ == '__main__':

    res = requests.get(
        'https://api.pushshift.io/reddit/search/submission?subreddit=worldnews&size=0&metadata=true&size=300')
    new_df = df_from_response(res)

    # print(new_df.title)
    for title in new_df.title:
        print(sentiment_analysis(title))
