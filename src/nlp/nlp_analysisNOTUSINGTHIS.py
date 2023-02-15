import pandas as pd
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# global variables:
similarity_threshold = 0.8


def train_val_test_split(x, y, validation=True):
    if validation:
        X_tr, X_te, y_tr, y_te = train_test_split(x, y, test_size=0.2)
        X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2)
        return X_tr, X_val, X_te, y_tr, y_val, y_te
    X_tr, X_te, y_tr, y_te = train_test_split(x, y, test_size=0.2)
    return X_tr, X_te, y_tr, y_te


# Functions:
def jaccard_similarity(set1: set, set2: set) -> float:
    """
    Calculates the Jaccard index between two sets.
    @param set1: first set.
    @param set2: second set.
    :return: Jaccard index.
    """
    return len(set1.intersection(set2)) / len(set1.union(set2))


def print_results(training_set: pd.DataFrame, testing_set: pd.DataFrame) -> None:
    """
    Prints the results of the analysis.
    @param training_set: training set.
    @param testing_set: testing set.
    :return: None.
    """

    # test the minimum and maximum values of the Cluster_i features for the train and test sets:
    print(training_set.filter(regex='Cd_').min().min())
    print(training_set.filter(regex='Cd_').max().max())
    print(testing_set.filter(regex='Cd_').min().min())
    print(testing_set.filter(regex='Cd_').max().max())

    # count how many customers are in each cluster for both train and test sets:
    print(training_set.filter(regex='Cd_').sum().sum())
    print(testing_set.filter(regex='Cd_').sum().sum())


if __name__ == '__main__':

    # import the aggregated dataset:
    df_agg = pd.read_csv("data" / "data_key_Bahrain.csv")

    # tokenize the title column:
    df_agg['title'] = df_agg['title'].apply(word_tokenize)

    # remove stopwords, punctuation and numbers:
    punctuation = ['.', ',', '!', '?', '(', ')', '[', ']', '{', '}', ':', ';', '"', "'"]
    stop_words = stopwords.words('english') + punctuation
    df_agg['title'] = df_agg['title'].apply(lambda x: [word for word in x if word not in stop_words
                                                       and not word.isdigit()])

    # remove duplicate words from the tokenized title:
    df_agg['title'] = df_agg['title'].apply(lambda x: set(x))

    # split the data set into train and test sets:
    X_train, X_test, y_train, y_test = train_val_test_split(df_agg.drop('CustomerChurned', axis=1),
                                                            df_agg['CustomerChurned'])
    # cluster on the training set:
    df_clusters = pd.DataFrame(columns=['CustomerId', 'ClusterId', 'title'])

    for row in tqdm(X_train.itertuples(), total=X_train.shape[0]):
        similarity = []
        # save all the similarity scores in a list
        for row2 in X_train.itertuples():
            similarity.append(jaccard_similarity(row.title, row2.title))
        # find the indexes of the rows that have similarity score bigger than the similarity threshold
        indexes = [i for i, x in enumerate(similarity) if x >= similarity_threshold]
        # if there is more than one row with similarity score bigger than the threshold:
        if len(indexes) > 1:
            # add the row to the new data frame and the shared title to the cluster:
            df_clusters = pd.concat([df_clusters, pd.DataFrame(
                [[row.CustomerId, ','.join([str(X_train.iloc[i]['CustomerId']) for i in indexes])]],
                columns=['CustomerId', 'ClusterId'])], ignore_index=True)

    # for each cluster, compute the shared title:
    df_clusters['title'] = df_clusters['ClusterId'] \
        .apply(lambda x: set.union(*[X_train.loc[X_train['CustomerId'] == int(float(i))]['title'].iloc[0]
                                     for i in x.split(',')]))

    # cast the ClusterId as a set of floats:
    df_clusters['ClusterId'] = df_clusters['ClusterId'].apply(
        lambda x: tuple(set([float(i) for i in x[0:-1].split(',')])))
    df_clusters['ClusterSize'] = df_clusters['ClusterId'].apply(lambda x: len(x))
    df_clusters.drop('CustomerId', axis=1, inplace=True)

    # Keep only the first row of each cluster id:
    df_clusters.drop_duplicates(subset='ClusterId', keep='first', inplace=True)

    # For each cluster, check the purity with the CustomerChurned column:
    df_clusters['Churned'] = df_clusters['ClusterId'].apply(lambda x: df_agg[df_agg['CustomerId'].isin(x)]
    ['CustomerChurned'].sum())

    df_clusters['Churned'] = df_clusters['Churned'] / df_clusters['ClusterSize']
    # create a new column with the churned purity, 1 if all churned or did not churn, 0 otherwise:
    df_clusters['ClusterPurity'] = df_clusters['Churned'].apply(lambda x: 1 if x == 0 or x == 1 else 0)

    # save the clusters to a csv file:
    df_clusters.to_csv(Path('..', '..', 'data', f'online_sales_dataset_clusters_{similarity_threshold}.csv'),
                       index=False)

    # for each cluster create a feature in the train dataset, that indicates if the customer is in the cluster:
    # for each cluster create a feature in the test dataset, that indicates if the customer is in the cluster,
    # the assignment is based on the jaccard similarity of the title of the customer and cluster:
    for row in tqdm(df_clusters.itertuples(), total=df_clusters.shape[0]):
        X_train['Cd_' + str(row.Index)] = X_train['CustomerId'].apply(lambda x: 1 if x in row.ClusterId else 0)
        X_test['Cd_' + str(row.Index)] = X_test['title'] \
            .apply(lambda x: 1 if jaccard_similarity(x, row.title) > similarity_threshold else 0)

    # rename the 'Cd_index' columns to 'Cd_0', 'Cd_1', etc.
    X_train.rename(columns={col: 'Cd_' + str(i) for i, col in enumerate(X_train.filter(regex='Cd_').columns)},
                   inplace=True)
    X_test.rename(columns={col: 'Cd_' + str(i) for i, col in enumerate(X_test.filter(regex='Cd_').columns)},
                  inplace=True)

    # check the results:
    print_results(X_train, X_test)

    # Add a feature measuring the length of the title for each customer, with the idea that possibly the longer
    # the title, the more likely the customer is not to churn:
    X_train['titleLength'] = X_train['title'].apply(lambda x: len(x))
    X_test['titleLength'] = X_test['title'].apply(lambda x: len(x))

    # save the new datasets:
    X_train.to_csv(Path('..', '..', 'data', 'online_sales_dataset_agg_nlp_train.csv'), index=False)
    X_test.to_csv(Path('..', '..', 'data', 'online_sales_dataset_agg_nlp_test.csv'), index=False)
    y_train.to_csv(Path('..', '..', 'data', 'online_sales_dataset_agg_nlp_train_labels.csv'), index=False)
    y_test.to_csv(Path('..', '..', 'data', 'online_sales_dataset_agg_nlp_test_labels.csv'), index=False)

    # Unfortunately the clustering does not affect a large number of customers.
