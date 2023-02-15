# Data manipulation:
from pathlib import Path
import os
from datetime import datetime
# Plotting:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# Type hinting:
from typing import Literal

plot_path: Path = Path(Path(__file__).parent, "plots")


# Functions:
# def plot_count_linechart(df: pd.DataFrame, news_type: Literal["positive", "negative", "neutral"]) -> None:
def plot_count_linechart(df: pd.DataFrame, file_name) -> None:
    """
    Plot the correlation matrix.
    :param df: pd.DataFrame: the dataframe to compute the correlation matrix from.
    :return: None
    """
    # print(df.info())
    plt.figure(figsize=(12, 10))
    # sns.lineplot(x=df['month'].astype(str), y=df['sentiment_score'], hue=df['bins'])
    sns.lineplot(x=df['month'].dt.year, y=df['sentiment_score'])
    # Setting Ticks

    plt.tick_params(axis='x', labelsize=5, rotation=90)
    plt.tight_layout()

    # Display

    # plt.show()

    plt.savefig(Path(plot_path, f'{file_name}_lineplot.png'))
    plt.close()

    # df["month"] = df["month"].dt.strftime('%Y-%m')

    # print(df['Negative'])

    # transform the dataframe to long format using melt()
    data_melt = pd.melt(df, id_vars=['month'], value_vars=['Negative', 'Neutral', 'Positive'])

    print(data_melt)

    # create a plot using seaborn
    # sns.lineplot(data=data_melt, x='month', y='value', hue='variable')

    plt.plot(x=df['month'].dt.year, y=df['Positive'], ci= None)
    plt.plot(x=df['month'].dt.year, y=df['Neutral'], ci= None)
    plt.plot(x=df['month'].dt.year, y=df['Negative'], ci= None)
    plt.savefig(Path(plot_path, f'{file_name}_countplot.png'))
    plt.close()


def plot_frequency_linechart(df: pd.DataFrame, news_type: Literal["positive", "negative", "neutral"]) -> None:
    # create area chart
    plt.stackplot(df.period, df.team_B, df.team_C,
                  labels=['Team B', 'Team C'],
                  colors=color_map)


def plot_word_cloud(df: pd.DataFrame, file_name) -> None:
    # # Start with one review:
    # text = df.CleanedText[1]
    #
    # # Create and generate a word cloud image:
    # wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
    #
    # # Display the generated image:
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")
    # plt.show()

    for index, row in df.iterrows():
        month = row['month']
        text = row['CleanedText']

        # generate the wordcloud for this month's text
        wordcloud = WordCloud(background_color='white').generate(text)

        # plot the wordcloud for this month
        plt.figure()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(month)
        plt.savefig(Path(plot_path, f'{month}_{file_name}_wordcloud.png'))
        plt.close()


def main() -> None:
    '''
    # Only egypt for the moment
    data_path: Path = Path(Path(__file__).parent.parent, "clean_data", "C_Egyptnyt.csv")
    df = pd.read_csv(data_path)
    print(df.pub_date)

    # assuming your dataframe is called df

    
    df['pub_date'] = pd.to_datetime(df['pub_date'])
    df['month'] = df['pub_date'].dt.to_period('M')

    # group by month and aggregate the data as needed
    df_grouped = df.groupby('month').agg({'CleanedText': 'sum'}).reset_index()

    print(df_grouped.info())

    # Plot the word cloud:
    # plot_word_cloud(df_grouped)
    '''

    # get a list of files in the folder
    file_list = os.listdir("../../data/nyt_data")

    # filter the list to include only CSV files
    csv_files = [f for f in file_list if f.endswith('.csv')]

    # loop through the list and read the CSV files, storing their names and data
    data = []
    for csv_file in csv_files:
        file_path = os.path.join("../../data/nyt_data", csv_file)
        df = pd.read_csv(file_path)
        # print(df.info())

        # assuming your dataframe is called df
        df['pub_date'] = pd.to_datetime(df['pub_date'])
        df['month'] = df['pub_date'].dt.to_period('M')

        # assuming your DataFrame is called df and the numerical feature is called 'num_feature'
        bin_labels = ['Negative', 'Neutral', 'Positive']
        num_quartiles = 3
        df['bins'] = pd.qcut(df['sentiment_score'], q=num_quartiles, labels=bin_labels)

        dummy = pd.get_dummies(df['bins'])

        # concatenate the original dataframe with the dummy variables
        df = pd.concat([df, dummy], axis=1)

        print(df.info())

        # group by month and aggregate the data as needed
        df_grouped = df.groupby('month').agg({'CleanedText': 'sum', 'sentiment_score': 'mean', 'Negative': 'count',
                                              'Neutral': 'count', 'Positive': 'count'}).reset_index()

        # df_grouped["month"] = df_grouped["month"].dt.strftime('%Y-%m')



        # plot_word_cloud(df_grouped, csv_file)

        plot_count_linechart(df_grouped, csv_file)

    # df['Publish_Date'] = pd.to_datetime(df['Publish_Date'], unit='s')


# Driver:
if __name__ == '__main__':
    main()
