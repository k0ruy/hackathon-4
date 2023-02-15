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
def plot_linechart(df: pd.DataFrame, file_name) -> None:
    # df["month"] = df["month"].dt.strftime('%Y-%m')
    df["year"] = df["month"].dt.year

    # Mean line plot
    plt.figure(figsize=(12, 10))
    sns.lineplot(x='year', y='sentiment_score', data=df, errorbar=None)
    # Setting Ticks
    plt.tick_params(axis='x', labelsize=15, rotation=90)
    plt.tight_layout()

    plt.savefig(Path(plot_path, f'{file_name}_lineplot.png'))
    plt.close()


def plot_count_linechart(df: pd.DataFrame, file_name) -> None:
    df["year"] = df["month"].dt.year

    sns.lineplot(x="year", y="sentiment_score", hue='bins', data=df, estimator="count", errorbar=None)

    # Set axis labels
    plt.xlabel("Month")
    plt.ylabel("Count")

    plt.savefig(Path(plot_path, f'{file_name}_countplot.png'))
    plt.close()


def plot_frequency_linechart(df: pd.DataFrame, file_name) -> None:
    # Calculate the total number of tips for each month
    total_tips = df.groupby("month")["sentiment_score"].count()

    # Calculate the frequency of each tip category for each month
    category_freq = df.groupby(["month", "bins"])["sentiment_score"].count() / total_tips

    # Reset the index of the category_freq DataFrame
    category_freq = category_freq.reset_index()

    category_freq["year"] = category_freq["month"].dt.year

    # Create a lineplot with the frequency of the categories over time
    sns.lineplot(data=category_freq, x="year", y="sentiment_score", hue="bins", errorbar=None)

    # Set axis labels
    plt.xlabel("Month")
    plt.ylabel("Frequency")

    plt.savefig(Path(plot_path, f'{file_name}_freqplot.png'))
    plt.close()


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
    # get a list of files in the folder
    file_list = os.listdir("../../data/nyt_data")

    # filter the list to include only CSV files
    csv_files = [f for f in file_list if f.endswith('.csv')]

    # loop through the list and read the CSV files, storing their names and data
    for csv_file in csv_files:
        file_path = os.path.join("../../data/nyt_data", csv_file)
        df = pd.read_csv(file_path)

        # assuming your dataframe is called df
        df['pub_date'] = pd.to_datetime(df['pub_date'])
        df['month'] = df['pub_date'].dt.to_period('M')

        # assuming your DataFrame is called df and the numerical feature is called 'num_feature'
        bin_labels = ['Negative', 'Neutral', 'Positive']
        df['bins'] = pd.cut(df['sentiment_score'], bins=[-1, -0.5, 0.5, 1], labels=bin_labels)

        plot_linechart(df, csv_file)

        plot_count_linechart(df, csv_file)

        plot_frequency_linechart(df, csv_file)

        # group by month and aggregate the data as needed
        df_grouped = df.groupby('month').agg({'CleanedText': 'sum', 'sentiment_score': 'mean'}).reset_index()

        plot_word_cloud(df_grouped, csv_file)

    # df['Publish_Date'] = pd.to_datetime(df['Publish_Date'], unit='s')


# Driver:
if __name__ == '__main__':
    main()
