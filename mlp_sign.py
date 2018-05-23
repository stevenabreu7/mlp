import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def der_sigmoid(y):
    return y * (1.0 - y)

class Sign_MLP(object):
    def __init__(self, input, hidden, output):
        # variables for the number of units in each layer
        self.input = input + 1
        self.hidden = hidden + 1
        self.output = output

        # initial activations all set to 1
        self.activations_input = np.ones(self.input)
        self.activations_hidden = np.ones(self.hidden)
        self.activations_output = np.ones(self.output)

        # random initial weights in range [-1,1]
        self.weights_hidden = np.random.random((self.input, self.hidden)) * 2 - 1
        self.weights_output = np.random.random((self.hidden, self.output)) * 2 - 1

    def forward(self, input_vector):
        # add bias to input vector
        self.activations_input = np.append(input_vector, 1)

        # activations for hidden layer (leave out bias unit)
        for i in range(self.hidden-1):
            potential = self.activations_input @ self.weights_hidden[:,i]
            self.activations_hidden[i] = sigmoid(potential)
        
        # activations for output layer (only one output unit)
        self.activations_output = self.activations_hidden @ self.weights_output

        # return the output vector
        return self.activations_output
    
    def backward(self, target, rate):
        # deltas for output
        delta_output = np.zeros(self.output)
        for i in range(self.output):
            delta_output[i] = 2 * (self.activations_output[i] - target[i])

        # deltas for hidden
        delta_hidden = np.zeros(self.hidden)
        for i in range(self.hidden):
            sum = delta_output @ self.weights_output[i]
            delta_hidden[i] = der_sigmoid(self.activations_hidden[i]) * sum

        # update weights for output
        for i in range(self.hidden):
            change = delta_output[0] * self.activations_hidden[i]
            self.weights_output[i][0] -= rate * change
        
        # update weights for hidden
        for i in range(self.input):
            for j in range(self.hidden):
                change = delta_hidden[j] * self.activations_input[i]
                self.weights_hidden[i][j] -= rate * change
        
        # calculate the error
        error = ((target - self.activations_output) ** 2).mean()
        return error
    
    def make_patterns(self, n):
        patterns = np.random.random((n, 2)) * 2 - 1
        results = np.sign(patterns[:,0] * patterns[:,1])
        results[results == -1] = 0
        return patterns, results
    
    def train(self, pattern_count, epochs=100, rate=0.1):
        # keep track of the errors and misclassifications per epoch
        errors = []
        missc = []
        
        # do the training
        for i in range(epochs):
            if i % (epochs // 10) == 0:
                print("Epoch {}".format(i))
            error = 0.0
            misses = 0
            # generate patterns and targets
            patterns, targets = self.make_patterns(pattern_count)
            # go through each pattern once
            for j in range(len(patterns)):
                # forward pass
                res = self.forward(patterns[j])
                # computing whether this is a misclassification
                bin_res = 0 if res < 0.5 else 1
                misses += 1 if bin_res != targets[j] else 0
                # backward propagation
                error += self.backward([targets[j]], rate)
            # saving the error and misclassification (averaged)
            error /= len(patterns)
            misses /= len(patterns)
            errors.append(error)
            missc.append(misses)
        
        # finished training, now test
        # results = [(self.forward(patterns[i]), targets[i]) for i in \
        #     range(len(patterns))]
        results = [self.forward(patterns[i]) for i in range(len(patterns))]
        colors = []
        for i in range(len(patterns)):
            a, b = self.forward(patterns[i])[0], targets[i]
            c = 0 if a < 0.5 else 1
            print("{:.4f} {} {}".format(a, b, c==b))
            if c == b:
                colors.append("blue")
            else:
                colors.append("red")

        # plot the errors
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        print(patterns[:,0].shape)
        # ax.plot_surface(X=np.arange(-1,1,0.1), Y=np.arange(-1,1,0.1), Z=patterns)
        ax.scatter(patterns[:,0], patterns[:,1], results, c=colors)
        plt.show()

        plt.plot(errors, label="Errors")
        plt.plot(missc, label="Missclassifications")
        plt.legend()
        plt.show()

mlp = Sign_MLP(2, 10, 1)
mlp.train(int(input("Patterns: ")), int(input("Epochs: ")), 0.1)