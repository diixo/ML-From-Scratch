
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
from mlfromscratch.deep_learning.loss_functions import CrossEntropy, NLLLoss
from mlfromscratch.utils.misc import bar_widgets
from mlfromscratch.deep_learning.layers import Dense, Dropout, Activation, Embedding

from mlfromscratch.nlp.sentencizer import Sentencizer

tokenizer = Sentencizer()

try:
    tokenizer.readFile("../data/train-1.txt", "../data/stopwords.txt")
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

    if (context_wnd == 3):
        for i in range(0, len(sentence) - 3):
            context = [sentence[i], sentence[i + 1], sentence[i + 2]]
            target = sentence[i + 3]
            data.append((context, target))
            #print("#" + target + " : " + context[0] + ", " + context[1] + ", " + context[2])

            context = [sentence[i], sentence[i + 1], sentence[i + 3]]
            target = sentence[i + 2]
            data.append((context, target))
            #print("#" + target + " : " + context[0] + ", " + context[1] + ", " + context[2])

            context = [sentence[i], sentence[i + 2], sentence[i + 3]]
            target = sentence[i + 1]
            data.append((context, target))
            #print("#" + target + " : " + context[0] + ", " + context[1] + ", " + context[2])

    if (context_wnd == 2):
        for i in range(0, len(sentence) - 2):
            context = [sentence[i], sentence[i + 1]]
            target = sentence[i + 2]
            data.append((context, target))
            #print("#" + target + " : " + sentence[i + 1] + ", " + sentence[i + 2])

############
#print(X)

##########################
for context, target in data:
    context_idxs = np.array([word_to_ix[w] for w in context])
    target_idxs = np.array([word_to_ix[target]])

    X.append(context_idxs)
    y.append(target_idxs)

# reshape to 2D-array
X = np.array(X) # reshape to array([sz, context_wnd])
y = np.array(y) # reshape to array([sz, 1]) = shape()
##########################

epochs = 100
embed_dim = 100  # 4*sqrt(tokenizer.sentences.sz)

#cbow = NeuralNetwork(optimizer=SGD, loss=NLLLoss, validation_data=(X, y))
cbow = NeuralNetwork(optimizer=SGD, loss=NLLLoss)

# TODO: need to verify with https://github.com/viix-co/ann-pure-numpy/tree/main
cbow.add(Embedding(tokenizer.vocab, embed_dim))

train_err, val_err = cbow.fit(X, y, n_epochs=epochs, batch_size=1)

accuracy = 100.0 * cbow.test_by_one(X, y)

print("Accuracy:", accuracy)

# Training and validation error plot
#n = len(train_err)
#training, = plt.plot(range(n), train_err, label="Training Error")
#validation, = plt.plot(range(n), val_err, label="Validation Error")
#plt.legend(handles=[training, validation])
#plt.title("Error Plot")
#plt.ylabel('Error')
#plt.xlabel('Iterations')
#plt.show()

#_, accuracy = clf.test_on_batch(X_test, y_test)
#print("Accuracy:", 0)
