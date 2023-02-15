import requests
import pandas as pd
from datetime import datetime
from sentiment import sentiment_analysis
import warnings
warnings.filterwarnings('ignore')


# we use this function to convert responses to dataframes
def df_from_response(res):
    # initialize temp dataframe for batch of data in response
    df = pd.DataFrame()

    bho = res.json()["data"]
    # loop through each post pulled from res and append to df
    for post in res.json()["data"]:
        df = df.append({
            # "subreddit": post["subreddit"],
            "title": post["title"],
            "selftext": post["selftext"],
            "upvote_ratio": post["upvote_ratio"],
            # "ups": post["ups"],
            # "downs": post["downs"],
            "score": post["score"],
            "link_flair_css_class": post["link_flair_css_class"],
            "created_utc": datetime.fromtimestamp(post["created_utc"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "id": post["id"],
            "fullname": post["author_fullname"] if "author_fullname" in post.keys() else "anon"
        }, ignore_index=True)

    return df


keyword = ["Ukraine", "Russia", "Syria", "Iran"]
# initialize dataframe and parameters for pulling data in loop
data = pd.DataFrame()
params = {"size": 300}

for word in keyword:
    for _ in range(2):
        res = requests.get(f"https://api.pushshift.io/reddit/search/submission?subreddit=worldnews&metadata=true&q={word}", params=params)

        new_df = df_from_response(res)

        data = data.append(new_df, ignore_index=True)

        # add/update fullname in params
        params["before"] = new_df.iloc[-1]
        print(params["before"])

    del params["before"]

print(data.info())
