from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import itertools


def suffle_data(id_list, label_list):
    idx = np.arange(len(id_list))
    np.random.shuffle(idx)
    shuffled_list = [id_list[k] for k in idx]
    shuffled_label_list = [label_list[k] for k in idx]
    return shuffled_list, shuffled_label_list


def split_by_KFold(data, nb_splits=3):
    kfold = KFold(n_splits=nb_splits, shuffle=True)

    index_folds = []
    for train_index, test_index in kfold.split(data):
        index_folds.append([train_index, test_index])

    train_data = [data[k] for k in index_folds[0][0]]
    val_data = [data[k] for k in index_folds[0][1]]
    return train_data, val_data


def split_by_StratifiedKFold(data, labels, nb_splits=3):
    """
    split data into training and validation set,
    and make sure the distribution of training is same as validation

    :param data: np.array, list of file_name
    :param labels: np.array, label
    :param nb_splits:  Number of folds. Must be at least 2.
    :return: train_data, train_labels, val_data, val_labels of one k-fold,type=list
    """
    skf = StratifiedKFold(n_splits=nb_splits, shuffle=True)

    index_folds = []
    for train_index, test_index in skf.split(data, labels):
        index_folds.append([train_index, test_index])

    # here we only return the first k-fold
    train_data = [data[k] for k in index_folds[0][0]]
    train_labels = [[labels[k] for k in index_folds[0][0]]]
    val_data = [data[k] for k in index_folds[0][1]]
    val_labels = [[labels[k] for k in index_folds[0][1]]]
    return train_data, train_labels, val_data, val_labels


def splitter_df(df, rate):
    """

    Args:
        df = dataframe
        rate = % to be training data

    Returns:
        train dataframe & test dataframe
    """
    # TODO: using StratifiedKFold
    nb_data = len(df)
    # the data with seed larger than test_nb is test data
    test_nb = len(df) * (1.0 - rate)
    seeds = np.arange(nb_data)
    np.random.shuffle(seeds)

    df['seed'] = seeds
    train_df = df[df['seed'] > test_nb]
    test_df = df[df['seed'] <= test_nb]

    train_df = train_df.drop('seed', axis=1)
    train_df.reset_index(drop=False, inplace=True)
    test_df = test_df.drop('seed', axis=1)
    test_df.reset_index(drop=False, inplace=True)

    return train_df, test_df


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function plots and save the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print(title)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # print(cm)

    plt.figure(figsize=(12,9))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title)


"""
# for test 
X=np.array([
    [1,2,3,4],
    [11,12,13,14],
    [21,22,23,24],
    [31,32,33,34],
    [41,42,43,44],
    [51,52,53,54],
    [61,62,63,64],
    [71,72,73,74]
])

y=np.array([1,1,0,0,1,1,0,0])

print(split_by_KFold(X, 4))
"""