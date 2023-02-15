import praw
import pandas as pd
import json
import os

# source: https://blog.devgenius.io/scraping-reddit-with-praw-python-reddit-api-wrapper-eaa7d788d7b9


if __name__ == '__main__':

    # checking if the directory demo_folder2
    # exist or not.
    if not os.path.isdir("data"):
        # if the demo_folder2 directory is
        # not present then create it.
        os.makedirs("data")

    r = praw.Reddit(client_id="hpLseXfJUivTmevTSPSrQQ",
                    client_secret="GM9ROBDjEpKrhET5zajLLaO4ozl5cw",
                    user_agent="Maleficent-Cow-1318",
                    )

    topics = ["Bahrain", "Cyprus", "Egypt", "Iran", "Iraq", "Israel", "Jordan", "Kuwait", "Lebanon", "Oman",
              "Palestine", "Qatar", "Saudi Arabia", "Syria", "Turkey", "United Arab Emirates", "Yemen"]
    sub = 'worldnews'  # subreddit
    sort = "new"
    limit = 50
    for q in topics:
        new_posts = r.subreddit(sub).search(q, sort=sort, limit=limit)
        all_posts = list()
        for post in new_posts:
            # print(vars(post)) # print all properties
            title = post.title,
            score = post.score,
            n_comments = post.num_comments,
            pub_date = post.created,
            link = post.permalink,
            res = {"Title": title[0], "Score": score[0], "Number_Of_Comments": n_comments[0],
                   "Publish_Date": pub_date[0], "Link": 'https://www.reddit.com' + link[0]}
            all_posts.append(res)

            # create csv file
            df = pd.DataFrame(all_posts)
            df.to_csv(f'data/data_{q}.csv', sep=',', index=False)

            # # create json file
            # json_string = json.dumps(all_posts)
            # jsonFile = open(f'data/data_key_{q}.json', "w")
            # jsonFile.write(json_string)
            # jsonFile.close()
