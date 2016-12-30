import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Recommender:

    def __init__(self, user_item, metric, filtering_mode='item'):
        self.user_item = user_item
        self.metric = metric
        self.filtering_mode=filtering_mode
        self._train()
        self.predict()


    def _train(self):
        if self.filtering_mode == 'item':
            self.sim = pairwise_distances(np.copy(self.user_item.T), metric=self.metric)
        else:
            self.sim = pairwise_distances(np.copy(self.user_item), metric=self.metric)


    def predict(self):
        # mask: bit array indicating whether the item is rated
        mask = np.copy(self.user_item)
        mask[self.user_item> 0] = 1

        if self.filtering_mode == 'item':
            self.pred = self.user_item.dot(self.sim) / mask.dot(np.abs(self.sim))
        else:
            # User filtering
            mean = self.user_item.sum(1) / (self.user_item!=0).sum(1)
            diff = self.user_item - np.multiply(mean[:, np.newaxis], mask)
            # Old prediction rule:
            # self.pred = self.sim.dot(self.user_item) / np.array([np.abs(self.sim).sum(axis=1)]).T
            # dotting the mask excluds un-rated entried
            self.pred = mean[:, np.newaxis] + self.sim.dot(diff) / (np.abs(self.sim)).T.dot(mask)

        return self.pred


    def evaluate(self, test_user_item):
        prediction = self.pred[test_user_item.nonzero()].flatten()

        idx = np.where(np.isnan(prediction))
        prediction[idx] = 5
        idx = np.where(np.isinf(prediction))
        prediction[idx] = 5

        test_user_item = test_user_item[test_user_item.nonzero()].flatten()
        np.round(prediction)

        count = [0] * 6
        distribution = np.zeros([6, 6])
        for i in range(len(prediction)):
            if prediction[i] > 5:
                prediction[i]  = 5
            if prediction[i] < 1:
                prediction[i] = 1

            count[int(prediction[i])] += 1
            distribution[int(prediction[i])][int(test_user_item[i])] += 1

        print count

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        #p lt.plot(prediction, test_user_item, 'r.')
        y = range(6) * 6
        x = [i / 6 for i in range(36)]
        z = np.zeros(36)
        dx = np.ones(36) * 0.7
        dy = np.ones(36) * 0.7
        dz = [distribution[i][j] for i,j in zip(x,y)]
        ax.bar3d(x, y, z, dx, dy, dz, color='r', zsort='average')
        plt.xlabel = 'Predicted Rating'
        plt.ylabel = 'Actual Rating'
        plt.zlabel = 'Count'
        plt.savefig(self.metric + '_' + self.filtering_mode + '.png')

        return sqrt(mean_squared_error(test_user_item, np.abs(prediction)))
