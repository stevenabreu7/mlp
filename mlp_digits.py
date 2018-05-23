import sys
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def der_sigmoid(y):
    return y * (1.0 - y)

class XOR_MLP(object):
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
    
    def train(self, patterns, targets, epochs=100, rate=0.1):
        assert len(patterns) == len(targets)
        # keep track of the errors and misclassifications per epoch
        errors = []
        missc = []
        
        # do the training
        for i in range(epochs):
            if i % (epochs // 10) == 0:
                print("Epoch {}".format(i))
            error = 0.0
            misses = 0
            # go through each pattern once
            for j in range(len(patterns)):
                # forward pass
                res = self.forward(patterns[j])
                # computing whether this is a misclassification
                bin_res = 0 if res < 0.5 else 1
                misses += 1 if bin_res != targets[j] else 0
                # backward propagation
                error += self.backward(targets[j], rate)
            # saving the error and misclassification (averaged)
            error /= len(patterns)
            misses /= len(patterns)
            errors.append(error)
            missc.append(misses)
        
        # finished training, now test
        results = [(self.forward(patterns[i]), targets[i]) for i in \
            range(len(patterns))]
        print(results)

        # plot the errors
        plt.plot(errors, label="Errors")
        plt.plot(missc, label="Missclassifications")
        plt.legend()
        plt.show()



patterns = np.array([[0,1], [1,0], [0,0], [1,1]])
targets = np.array([[1],[1],[0],[0]])

mlp = XOR_MLP(2, 2, 1)
mlp.train(patterns, targets, 10*1000, 0.1)