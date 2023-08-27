
from __future__ import print_function
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# Import helper functions
from mlfromscratch.deep_learning import NeuralNetwork
from mlfromscratch.utils import train_test_split, to_categorical, normalize, Plot
from mlfromscratch.utils import get_random_subsets, shuffle_data, accuracy_score
from mlfromscratch.deep_learning.optimizers import StochasticGradientDescent, Adam, RMSprop, Adagrad, Adadelta
from mlfromscratch.deep_learning.loss_functions import CrossEntropy
from mlfromscratch.utils.misc import bar_widgets
from mlfromscratch.deep_learning.layers import Dense, Dropout, Activation

i2hex = {
    0: "__ ",
    1: "__ ",
    2: "__ ",
    3: "03 ",
    4: "04 ",
    5: "05 ",
    6: "06 ",
    7: "07 ",
    8: "08 ",
    9: "09 ",
    10: "0A ",
    11: "0B ",
    12: "0C ",
    13: "0D ",
    14: "0E ",
    15: "0F ",
    16: "10 "}


def main():

    optimizer = Adam()

    #-----
    # MLP
    #-----

    data = datasets.load_digits()
    X = data.data
    y = data.target
    imgs = data.images

    #f = open("digits-8x8.bin", "wb")
    #for i in range(imgs.shape[0]):
    #    for h in range(8):

    #        print(f"{i2hex[int(imgs[i,h,0])]}{i2hex[int(imgs[i,h,1])]}{i2hex[int(imgs[i,h,2])]}{i2hex[int(imgs[i,h,3])]}{i2hex[int(imgs[i,h,4])]}{i2hex[int(imgs[i,h,5])]}{i2hex[int(imgs[i,h,6])]}{i2hex[int(imgs[i,h,7])]}")

    #    print(f"{str(int(y[i]))}------------------------")
                #xx = np.array(x, dtype='d')
        #f.write(xx)
    #f.close()

    # Convert to one-hot encoding
    y = to_categorical(y.astype("int"))

    n_samples, n_features = X.shape
    n_hidden = 150

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=1)

    clf = NeuralNetwork(optimizer=optimizer,
                        loss=CrossEntropy,
                        validation_data=(X_test, y_test))

    clf.add(Dense(n_hidden, input_shape=(n_features,)))
    clf.add(Activation('leaky_relu'))

    clf.add(Dense(n_hidden))
    clf.add(Activation('leaky_relu'))
    clf.add(Dropout(0.25))

    clf.add(Dense(n_hidden))
    clf.add(Activation('leaky_relu'))
    clf.add(Dropout(0.25))

    #clf.add(Dense(n_hidden))
    #clf.add(Activation('leaky_relu'))
    #clf.add(Dropout(0.25))

    clf.add(Dense(10))
    clf.add(Activation('softmax'))

    print ()
    clf.summary(name="MLP")
    
    train_err, val_err = clf.fit(X_train, y_train, n_epochs=100, batch_size=80)
    
    # Training and validation error plot
    n = len(train_err)
    training, = plt.plot(range(n), train_err, label="Training Error")
    validation, = plt.plot(range(n), val_err, label="Validation Error")
    plt.legend(handles=[training, validation])
    plt.title("Error Plot")
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.show()

    _, accuracy = clf.test_on_batch(X_test, y_test)
    print ("Accuracy:", accuracy)

    # Reduce dimension to 2D using PCA and plot the results
    y_pred = np.argmax(clf.predict(X_test), axis=1)
    Plot().plot_in_2d(X_test, y_pred, title="Multilayer Perceptron", accuracy=accuracy, legend_labels=range(10))


if __name__ == "__main__":
    main()