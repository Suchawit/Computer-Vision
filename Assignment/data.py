# data.py
"""
This file contains helper functions
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import itertools
import time


"""
Code based on: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
"""
def plot_confusion_matrix(cm, title="Confusion Matrix", save=True):
    plt.figure()
    classes = np.arange(cm.shape[0])
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    if save:
        plt.savefig('confusion_matrix.png'.format(title), bbox_inches='tight')
        return

    plt.show()

class Timer(object):
    """
    Class to evaluate the execution time of code

    with Timer(verbose=True) as t:
        [your code you want to check]
    print("this took {} mins".format(t.mins))
    """
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.mins = self.secs // 60
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print('elapsed time: {:.2f} s ({:.4f} ms)'.format(self.secs, self.msecs))

#
# data.py ends here
