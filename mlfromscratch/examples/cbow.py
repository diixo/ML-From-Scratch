
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
from mlfromscratch.deep_learning.layers import Dense, Dropout, Activation

from mlfromscratch.nlp.sentencizer import Sentencizer

tokenizer = Sentencizer()

try:
    tokenizer.readFile("train-nn.txt")
except IOError:  # FileNotFoundError in Python 3
    print("File not found")


vocab_size = len(tokenizer.vocab)

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

###################################################################################

CONTEXT_SIZE = 2

# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.
# Each tuple is ([ word_i-CONTEXT_SIZE, ..., word_i-1 ], target word)
ngrams = [
    (
        [test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)],
        test_sentence[i]
    )
    for i in range(CONTEXT_SIZE, len(test_sentence))
]
# Print the first 3, just so you can see what they look like.
print(ngrams[:42])

