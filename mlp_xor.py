import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def der_sigmoid(y):
    return y * (1.0 - y)

class MLP(object):
    def __init__(self, input, hidden, output, S):
        """
        Initializes MLP with one hidden layer.
        input:  # of input units
        hidden: # of units in hidden layer
        output: # of output units
        """
        # set number of units (include bias)
        self.input = input + 1
        self.hidden = hidden + 1
        self.output = output

        # activations (first entry is bias)
        self.act_input = [1.0] * self.input
        self.act_hidden = [1.0] * self.hidden
        self.act_output = [1.0] * self.output

        # randomize weights
        self.wei_input = (np.random.random((self.input, self.hidden)) * 2 - 1) * 2
        self.wei_output = (np.random.random((self.hidden, self.output)) * 2 - 1) * 2

    def feed_forward(self, inputs):
        """
        One feed forward iteration on input vector.
        inputs: input vector
        return: output vector
        """
        if len(inputs) != self.input - 1:
            raise ValueError("Wrong input dimensionality.")

        # set activations of input layer to the input vector
        for i in range(1, self.input):
            # don't change the first input unit (it's the bias)
            self.act_input[i] = inputs[i-1]

        # compute activations for hidden layer
        for i in range(1, self.hidden):
            # don't change the first hidden unit (it's the bias)
            sum = 0.0
            for j in range(1, self.input):
                sum += self.act_input[j] * self.wei_input[j][i]
            # add the bias (weight only, since value is one)
            sum += self.wei_input[0][i]
            self.act_hidden[i] = sigmoid(sum)

        # compute activations for output layer
        for i in range(self.output):
            self.act_output[i] = 0.0
            for j in range(1, self.hidden):
                self.act_output[i] += self.act_hidden[j] * self.wei_output[j][i]
            # add the bias (weight only, since value is one)
            self.act_output[i] += self.wei_input[0][i]

            # # changing the output to be binary
            # self.act_output[i] = 0 if self.act_output[i] < 0.5 else 1

        return self.act_output

    def back_propagation(self, targets, N = 0.0002):
        """
        targets: target values
        N: learning rate
        return: current error
        """
        if len(targets) != self.output:
            raise ValueError("Invalid target dimensionality.")

        # deltas for output units
        output_deltas = [0.0] * self.output
        for i in range(self.output):
            # using quadratic loss with no regularizer -> eq. 61 in the LN
            output_deltas[i] = 2 * (self.act_output[i] - targets[i])

        # deltas for hidden units
        hidden_deltas = [0.0] * self.hidden
        for i in range(self.hidden):
            sum = 0.0
            for j in range(self.output):
                sum += output_deltas[j] * self.wei_output[i][j]
            hidden_deltas[i] = der_sigmoid(self.act_hidden[i]) * sum

        # update weights for output layer
        for i in range(self.hidden):
            for j in range(self.output):
                change = output_deltas[j] * self.act_hidden[i]
                self.wei_output[i][j] -= N * change

        # update weights for hidden layer
        for i in range(self.input):
            for j in range(self.hidden):
                change = hidden_deltas[j] * self.act_input[i]
                self.wei_input[i] -= N * change

        # calculate error
        error = 0.0
        for i in range(self.output):
            error += 0.5 * (targets[i] - self.act_output[i]) ** 2
        return error

    def print_weights(self):
        print("Weights (3 input, 1 output)")
        for i in range(3):
            for j in range(3):
                print("{:<15.5}".format(self.wei_input[i][j]), end=" ")
            print()
        for i in range(3):
            print("{:<15.5}".format(self.wei_output[i][0]), end=" ")
        print()

    def train(self, patterns, iterations = 100, N = 0.0002, S=5):
        errors = []
        misscs = []
        misses = 1
        i = 0
        self.print_weights()
        for i in range(5000):
            error = 0.0
            misses = 0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                x = self.feed_forward(inputs)
                error += self.back_propagation(targets, N)
                true_out = 0 if x[0] < 0.5 else 1
                misses += 0 if true_out == targets[0] else 1
            error /= len(patterns)
            errors.append(error)
            misscs.append(misses / len(patterns))
        self.print_weights()
        print("{:<3} {:8.5f} {:2}\n".format(i, error, misses))
        # print(x[0], targets[0], end="\t")

        # plot errors and misses
        plt.plot(errors, label="Errors")
        plt.plot(misscs, label="Missclassifications")
        plt.legend()
        plt.savefig("N-{}-S-{}.png".format(N,S))

N = float(input("N: "))
# S = int(input("Range of initializations: 1/"))
S=1

patterns = np.array([
        [[0,0], [0]],
        [[1,1], [0]],
        [[0,1], [1]],
        [[1,0], [1]]
    ])

mlp = MLP(2, 2, 1, S)
mlp.train(patterns, 10000, N, S)
