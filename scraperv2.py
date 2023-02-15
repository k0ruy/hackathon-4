import praw
import pandas as pd
import json
import os


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

    topics = ["Bahrain", "Cyprus", "Egypt", "Iran", "Iraq", "Israel", "Jordan", "Kuwait", "Lebanon", "Oman", "Palestine", "Qatar", "Saudi Arabia", "Syria", "Turkey", "United Arab Emirates", "Yemen"]
    sub = 'worldnews'  # subreddit
    sort = "new"
    limit = 50
    for q in topics:
        top_posts = r.subreddit(sub).search(q, sort=sort, limit=limit)
        total_posts = list()
        for post in top_posts:
            # print(vars(post)) # print all properties
            Title = post.title,
            Score = post.score,
            Number_Of_Comments = post.num_comments,
            Publish_Date = post.created,
            Link = post.permalink,
            data_set = {"Title": Title[0], "Score": Score[0], "Number_Of_Comments": Number_Of_Comments[0],
                        "Publish_Date": Publish_Date[0], "Link": 'https://www.reddit.com' + Link[0]}
            total_posts.append(data_set)

            # create csv file
            df = pd.DataFrame(total_posts)
            df.to_csv(f'data/data_key_{q}.csv', sep=',', index=False)

            # create json file
            json_string = json.dumps(total_posts)
            jsonFile = open(f'data/data_key_{q}.json', "w")
            jsonFile.write(json_string)
            jsonFile.close()
