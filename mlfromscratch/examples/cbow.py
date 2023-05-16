
# https://www.kaggle.com/code/alincijov/nlp-starter-continuous-bag-of-words-cbow/

import re
import numpy as np
import string
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from operator import itemgetter

from subprocess import check_output

# Import helper functions
from mlfromscratch.deep_learning import NeuralNetwork
from mlfromscratch.utils import train_test_split, to_categorical, normalize, Plot
from mlfromscratch.utils import get_random_subsets, shuffle_data, accuracy_score
from mlfromscratch.deep_learning.optimizers import SGD, StochasticGradientDescent, Adam, RMSprop, Adagrad, Adadelta
from mlfromscratch.deep_learning.loss_functions import CrossEntropy
from mlfromscratch.utils.misc import bar_widgets
from mlfromscratch.deep_learning.layers import Dense, Dropout, Activation, Embedding

from mlfromscratch.nlp.sentencizer import Sentencizer

tokenizer = Sentencizer()

try:
    tokenizer.readFile("../data/train-nn.txt", "../data/stopwords.txt")
except IOError:  # FileNotFoundError in Python 3
    print("File not found")

context_wnd = 3 # 2, 3 or 4: [(context_wnd), target]

word_to_ix = {word: i for i, word in enumerate(tokenizer.vocab)}
ix_to_word = {i: word for i, word in enumerate(tokenizer.vocab)}

# data - [(context), target]
data = []
X = [] #context
y = [] #target
############
for sentence in tokenizer.sentences:
    if (context_wnd == 4):
        for i in range(2, len(sentence) - 2):
            context = [sentence[i - 2], sentence[i - 1], sentence[i + 1], sentence[i + 2]]
            target = sentence[i]
            data.append((context, target))

            X.append(context)
            y.append(target)

    if (context_wnd == 3):
        for i in range(0, len(sentence) - 3):
            context = [sentence[i], sentence[i + 1], sentence[i + 2]]
            target = sentence[i + 3]
            data.append((context, target))

            X.append(context)
            y.append(target)
            #print("#" + target + " : " + context[0] + ", " + context[1] + ", " + context[2])

            context = [sentence[i], sentence[i + 1], sentence[i + 3]]
            target = sentence[i + 2]
            data.append((context, target))

            X.append(context)
            y.append(target)
            #print("#" + target + " : " + context[0] + ", " + context[1] + ", " + context[2])

            context = [sentence[i], sentence[i + 2], sentence[i + 3]]
            target = sentence[i + 1]
            data.append((context, target))

            X.append(context)
            y.append(target)
            #print("#" + target + " : " + context[0] + ", " + context[1] + ", " + context[2])

    if (context_wnd == 2):
        for i in range(0, len(sentence) - 2):
            context = [sentence[i], sentence[i + 1]]
            target = sentence[i + 2]
            data.append((context, target))

            X.append(context)
            y.append(target)
            #print("#" + target + " : " + sentence[i + 1] + ", " + sentence[i + 2])

############
#print(X)

epochs = 100
embed_dim = 100  #sqrt(tokenizer.sentences.sz)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=1)

cbow = NeuralNetwork(optimizer=SGD, loss=CrossEntropy, validation_data=(X_test, y_test))

cbow.add(Embedding(tokenizer.vocab, embed_dim))

#train_err, val_err = cbow.fit(X_train, y_train, n_epochs=epochs, batch_size=1)
