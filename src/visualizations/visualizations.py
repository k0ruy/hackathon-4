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

    # df["month"] = df["month"].dt.strftime('%Y-%m')
    df["year"] = df["month"].dt.year

    # Mean line plot
    plt.figure(figsize=(12, 10))
    sns.lineplot(x='year', y='sentiment_score', data=df)
    # Setting Ticks
    plt.tick_params(axis='x', labelsize=5, rotation=90)
    plt.tight_layout()

    plt.savefig(Path(plot_path, f'{file_name}_lineplot.png'))
    plt.close()

    # Create a figure with multiple subplots
    fig, ax = plt.subplots()

    # Plot first sns.linplot on the first subplot
    sns.lineplot(x="year", y="Negative", data=df, ax=ax)

    # Plot second sns.linplot on the second subplot
    sns.lineplot(x="year", y="Positive", data=df, ax=ax)

    # Show the plot
    plt.show()

    # fig.savefig(Path(plot_path, f'{file_name}_countplot.png'))
    # plt.close()


def plot_frequency_linechart(df: pd.DataFrame, news_type: Literal["positive", "negative", "neutral"]) -> None:
    # create area chart
    plt.stackplot(df.period, df.team_B, df.team_C,
                  labels=['Team B', 'Team C'],
                  colors=color_map)


def plot_word_cloud(df: pd.DataFrame, file_name) -> None:
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
        # df['bins'] = pd.qcut(df['sentiment_score'], q=num_quartiles, labels=bin_labels)
        df['bins'] = pd.cut(df['sentiment_score'], bins=[-1, -0.5, 0.5, 1], labels=bin_labels)

        dummy = pd.get_dummies(df['bins'])

        # concatenate the original dataframe with the dummy variables
        df = pd.concat([df, dummy], axis=1)

        print(df.info())

        # group by month and aggregate the data as needed
        df_grouped = df.groupby('month').agg({'CleanedText': 'sum', 'sentiment_score': 'mean', 'Negative': 'count',
                                              'Neutral': 'count', 'Positive': 'count'}).reset_index()

        # plot_word_cloud(df_grouped, csv_file)

        plot_count_linechart(df_grouped, csv_file)

    # df['Publish_Date'] = pd.to_datetime(df['Publish_Date'], unit='s')


# Driver:
if __name__ == '__main__':
    main()
