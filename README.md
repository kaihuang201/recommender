# Recommender System using Memory-based Collaborative Filtering

This is a toy recommender system project using memory-based collaborative filtering written in python.

Data set used can be found here: [MovieLens 100K Dataset](http://grouplens.org/datasets/movielens/100k/)

To run, execute `python run.py`. The logic for reading data file and spliting data set into train and test sets can also be found in run.py.

The training and prediction logic can be found in `recommender.py`

# Acknowledgement

This algorithm used in this project is written mostly following the tutorial [Implementing your own recommender systems in Python](http://online.cambridgecoding.com/notebooks/eWReNYcAfB/implementing-your-own-recommender-systems-in-python-2).
However it is worth pointing out that the algorithm used in this project differs from the tutorial in that un-rated items are not considered in training and prediction in this project, this seems to have resulted in a drastic improvement in prediction accuracy.
For more details take a look at the `_train` and `predict` functions in `recommender.py`
