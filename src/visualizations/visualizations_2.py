# Data manipulation:
from pathlib import Path
import os
# Plotting:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from plotly.tools import mpl_to_plotly

plot_path: Path = Path(Path(__file__).parent, "plots")
if not plot_path:
    plot_path.mkdir(parents=True, exist_ok=True)


# Functions:
def plot_linechart(df: pd.DataFrame, file_name: str) -> None:
    df["year"] = df["month"].dt.year

    # Mean line plot
    figure = plt.figure()
    sns.lineplot(x='year', y='sentiment_score', data=df, errorbar=None)

    plt.title(file_name)

    # Set axis labels
    plt.xlabel("Year")
    plt.ylabel("Mean score")

    # Setting Ticks
    # plt.tick_params(axis='x', labelsize=15, rotation=90)
    plt.tight_layout()
    return mpl_to_plotly(figure)



def plot_count_linechart(df: pd.DataFrame, file_name: str) -> None:
    df['month'] = pd.to_datetime(df['month'])
    df["year"] = df["month"].dt.year
    print(df.head())

    figure = plt.figure()
    sns.lineplot(x="year", y="sentiment_score", hue='bins', data=df, estimator="count", errorbar=None)

    plt.title(file_name)

    # Set axis labels
    plt.xlabel("Year")
    plt.ylabel("Count")

    plt.tight_layout()

    #plt.savefig(Path(plot_path, f'{file_name}_countplot.png'))
    plt.show()
    return mpl_to_plotly(figure)


def plot_frequency_linechart(df: pd.DataFrame, file_name: str) -> None:
    # Calculate the total number of tips for each month
    total_tips = df.groupby("month")["sentiment_score"].count()

    # Calculate the frequency of each tip category for each month
    category_freq = df.groupby(["month", "bins"])["sentiment_score"].count() / total_tips

    # Reset the index of the category_freq DataFrame
    category_freq = category_freq.reset_index()

    category_freq["year"] = category_freq["month"].dt.year
    figure = plt.figure()
    # Create a lineplot with the frequency of the categories over time
    sns.lineplot(data=category_freq, x="year", y="sentiment_score", hue="bins", errorbar=None)

    plt.title(file_name)

    # Set axis labels
    plt.xlabel("Year")
    plt.ylabel("Frequency")

    plt.tight_layout()

    return mpl_to_plotly(figure)


def plot_word_cloud(df: pd.DataFrame, file_name) -> None:
    for index, row in df.iterrows():
        month = row['month']
        text = row['CleanedText']

        # generate the wordcloud for this month's text
        wordcloud = WordCloud(background_color='white').generate(text)

        # plot the wordcloud for this month
        figure = plt.figure()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(month)
        return figure

import glob
def main() -> None:
    # get a list of files in the folder
    csv_files = glob.glob("../../data/complete/*.csv")
    # loop through the list and read the CSV files, storing their names and data
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        df['pub_date'] = pd.to_datetime(df['pub_date'])
        df['month'] = df['pub_date'].dt.to_period('M')

        bin_labels = ['Negative', 'Neutral', 'Positive']
        df['bins'] = pd.cut(df['sentiment_score'], bins=[-1, -0.5, 0.5, 1], labels=bin_labels)

        plot_linechart(df, "")

        plot_count_linechart(df, "")

        plot_frequency_linechart(df, "")

        # group by month and aggregate the data as needed
        df_grouped = df.groupby('month').agg({'CleanedText': 'sum', 'sentiment_score': 'mean'}).reset_index()

        plot_word_cloud(df_grouped, file_name)

    # df['Publish_Date'] = pd.to_datetime(df['Publish_Date'], unit='s')


# Driver:
if __name__ == '__main__':
    main()
