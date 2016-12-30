from recommender import *

import numpy as np
import pandas as pd
import sklearn.cross_validation as cv


def build_user_item_matrix(data):
    mat = np.zeros([max(data.user), max(data.item)])
    for i, row in data.iterrows():
        mat[row['user']-1][row['item']-1] = row['rating']
    return mat


if __name__=='__main__':
    data = pd.read_csv('ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'time'])
    num_user = len(data.user.unique())
    num_item = len(data.item.unique())
    print num_user, num_item

    train, test = cv.train_test_split(data, test_size=0.2)
    print train.shape, test.shape

    train_user_item = build_user_item_matrix(train)
    test_user_item = build_user_item_matrix(test)

    with open('output.txt', 'w+') as outfile:
        outfile.write('metric  | item    user\n')
        for metric in ['jaccard', 'cosine', 'correlation']:
            outfile.write(metric + '\t')
            for filter_mode in ['item', 'user']:
                r = Recommender(train_user_item, metric, filter_mode)
                outfile.write(str(r.evaluate(test_user_item)) + '    ')
            outfile.write('\n')
