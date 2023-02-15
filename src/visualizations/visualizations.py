# Data manipulation:
from pathlib import Path
import os
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
def plot_count_linechart(df: pd.DataFrame, news_type: Literal["positive", "negative", "neutral"]) -> None:
    """
    Plot the correlation matrix.
    :param df: pd.DataFrame: the dataframe to compute the correlation matrix from.
    :return: None
    """
    sns.lineplot(data=df, x="year", y="passengers", hue="month")

    # # select only the numerical features:
    # numerical_features = df.select_dtypes(exclude='category').columns
    # # compute the correlation matrix:
    # corr_matrix = df[numerical_features].corr('spearman')
    # # mask the upper triangle:
    # mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    # # get the y_tick:
    # x_tick = corr_matrix.index
    # # replace the last y_tick label with an empty string:
    # y_tick = [col.replace('limit_bal', '') for col in x_tick]
    #
    # # plot the correlation matrix:
    # plt.figure(figsize=(20, 20))
    # sns.heatmap(corr_matrix, annot=True, cbar=True, cmap='coolwarm', mask=mask,
    #             fmt='.2f', annot_kws={'size': 10}, linewidths=0.5,
    #             xticklabels=x_tick[:-1], yticklabels=y_tick)
    #
    # # Annotate the correlation matrix:
    # plt.title('Correlation Matrix with Spearman Correlation')
    # # Save the plot:
    # plot_path.mkdir(parents=True, exist_ok=True)
    # plt.savefig(Path(plot_path, 'correlation_matrix.png'))
    # plt.close()


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

    folder_path = "../clean_data"

    # get a list of files in the folder
    file_list = os.listdir(folder_path)

    # filter the list to include only CSV files
    csv_files = [f for f in file_list if f.endswith('.csv')]

    # loop through the list and read the CSV files, storing their names and data
    data = []
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)

        # assuming your dataframe is called df
        df['pub_date'] = pd.to_datetime(df['pub_date'])
        df['month'] = df['pub_date'].dt.to_period('M')

        # group by month and aggregate the data as needed
        df_grouped = df.groupby('month').agg({'CleanedText': 'sum'}).reset_index()

        # data.append({'file_name': csv_file, 'data': df})
        plot_word_cloud(df_grouped, csv_file)


# Driver:
if __name__ == '__main__':
    main()
