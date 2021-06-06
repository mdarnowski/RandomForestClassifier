"""
Random Forest Classifier project.
"""

import argparse
import pandas
import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def main(args):
    """
    This is a main function of this module. It will:
        *read csv file with data
        *split arrays into random train and test subsets
        *create random forest classifier
        *train the model on the training dataset
        *perform predictions on the test dataset
        *calculate accuracy of the predictions,
         numerical feature importances, save results to a txt file
        *save a visualisation of a single tree to a png file

    :param args: a pythonic namespace containing parameters from a command line
    """

    # read csv
    data = pandas.read_csv(args.train_file)

    # preprocessing
    data[data == np.inf] = np.nan
    data.fillna(data.mean(), inplace=True)

    # define dataset
    column_loc = data.columns.get_loc(args.classification_column)
    _x = data.iloc[:, 0:column_loc]
    _y = data.iloc[:, column_loc]

    # splitting arrays into train and test subsets
    x_train, x_test, y_train, y_test = train_test_split(_x, _y, test_size=args.train_test_split)

    # creating a Random Forest classifier
    rfc = RandomForestClassifier(max_depth=args.max_depth,
                                 min_impurity_decrease=args.acceptable_impurity)

    # training the model
    rfc.fit(x_train, y_train)

    # performing predictions
    prediction = rfc.predict(x_test)

    # create txt
    out = open("out.txt", "w+")

    # save accuracy calculation to txt file
    out.write('accuracy of the model:    ' +
              metrics.accuracy_score(y_test, prediction).__str__() + '\n')

    # save the list of numerical features importances to a txt file
    f_importances = [(feature, round(importance, 2)) for feature, importance in
                     zip(_x.columns, list(rfc.feature_importances_))]
    for pair in f_importances:
        out.write('{:25} importance: {}'.format(*pair) + '\n')

    # visualizing a single tree
    fig = plt.subplots(figsize=(2, 2), dpi=800)[0]
    tree.plot_tree(rfc.estimators_[0],
                   feature_names=_x.columns,
                   class_names=True,
                   filled=True)

    # save visualization to a png file
    fig.savefig('singleTree.png')


def parse_arguments():
    """
    This is a function that parses arguments from command line.

    :return a pythonic namespace with parameters from a command line
    """
    parser = argparse.ArgumentParser(description='This is a random forest classifier program.')
    parser.add_argument('-t',
                        '--train_file',
                        type=argparse.FileType('r'),
                        help='csv file with data (required)',
                        required=True)

    parser.add_argument('-s',
                        '--train_test_split',
                        type=float_in_range(0.2, 0.8),
                        default=0.2,
                        help='how many of datapoint will be used for tests'
                             ' (default and min 0.2; while max 0.8)')

    parser.add_argument('-c',
                        '--classification_column',
                        type=str,
                        help='name of column in dataset with classification data (required)',
                        required=True)

    parser.add_argument('-m',
                        '--max_depth',
                        type=int,
                        default=5,
                        help='maximum depth of tree (default value is 5)')

    parser.add_argument('-a',
                        '--acceptable_impurity',
                        '--demo_argument',
                        type=float,
                        default=0.0,
                        help='level of impurity at which we no longer split nodes'
                             ' (default value is 0.0)')

    return parser.parse_args()


def float_in_range(_min, _max):
    """
    This is a function for checking a float range.

    :param _min: minimum acceptable argument
    :param _max: maximum acceptable argument

    :return a function handle of an argument type function for ArgumentParser
    """

    def range_check(_fl):
        """
        This is a new type function for argparse.

        :param _fl: a float within predefined range

        :return a function handle of an argument type function for ArgumentParser
        """

        try:
            handle = float(_fl)
        except ValueError as err:
            raise argparse.ArgumentTypeError("expects float") from err
        if handle < _min or handle > _max:
            raise argparse.ArgumentTypeError(
                "must be in range [" + str(_min) + ", " + str(_max) + "]")
        return handle

    return range_check


if __name__ == '__main__':
    main(parse_arguments())
