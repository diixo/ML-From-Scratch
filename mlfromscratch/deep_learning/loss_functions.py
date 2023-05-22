from __future__ import division
import numpy as np
from mlfromscratch.utils import accuracy_score
from mlfromscratch.deep_learning.activation_functions import Sigmoid

class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0

class SquareLoss(Loss):
    def __init__(self): pass

    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)

class CrossEntropy(Loss):
    def __init__(self): pass

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def acc(self, y, p):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)


class NLLLoss(Loss):
    def __init__(self): pass

    def loss(self, targets, pred):
        logs = pred[-1]
        out = logs[range(len(targets)), targets]
        tmp = -out.sum() / len(out)
        return -out

    # return accuracy
    def acc(self, y, y_pred):
        # stub temporary
        return 0

    # log_softmax_crossentropy_with_logits
    def gradient(self, target, preds):
        logits, o = preds

        out = np.zeros_like(logits)
        out[np.arange(len(logits)), target] = 1

        softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

        return (-out + softmax) / logits.shape[0]
