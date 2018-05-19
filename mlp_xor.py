import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def inv_sigmoid(x):
    return np.log(x/(1-x))

def der_sigmoid(y):
    return y * (1.0 - y)

class MLP(object):
    def __init__(self, input, hidden, output):
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
        # activations (last entry is bias)
        self.act_input = [1.0] * self.input
        self.act_hidden = [1.0] * self.hidden
        self.act_output = [1.0] * self.output
        # randomize weights
        self.wei_input = abs(np.random.randn(self.input, self.hidden)) / 50
        self.wei_output = abs(np.random.randn(self.hidden, self.output)) / 50
        # # change of weights
        # self.change_input = np.zeros((self.input, self.hidden))
        # self.change_output = np.zeros((self.hidden, self.output))

    def feed_forward(self, inputs):
        """
        One feed forward iteration on input vector.
        inputs: input vector
        return: output vector
        """
        if len(inputs) != self.input - 1:
            raise ValueError("Wrong input dimensionality.")
        # activations for input layer
        for i in range(self.input - 1):
            self.act_input[i] = inputs[i]
        # activations for hidden layer
        for i in range(self.hidden - 1):
            sum = 0.0
            for j in range(self.input):
                sum += self.act_input[j] * self.wei_input[j][i]
            self.act_hidden[i] = sigmoid(sum)
        # activations for output layer
        for i in range(self.output):
            self.act_output[i] = 0.0
            for j in range(self.hidden):
                self.act_output[i] += self.act_hidden[j] * self.wei_output[j][i]
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
                self.wei_output[i][j] -= N * change # + self.change_output[i][j]
                # self.change_output[i][j] = change
        # update weights for hidden layer
        for i in range(self.input):
            for j in range(self.hidden):
                change = hidden_deltas[j] * self.act_input[i]
                self.wei_input[i] -= N * change # + self.change_input[i][j]
                # self.change_input[j] = change
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

    def train(self, patterns, iterations = 100, N = 0.0002):
        for i in range(iterations):
            self.print_weights()
            error = 0.0
            misses = 0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                x = self.feed_forward(inputs)
                error += self.back_propagation(targets, N)
                true_out = 0 if x[0] < 0.5 else 1
                misses += 0 if true_out == targets[0] else 1
                print(x[0], targets[0])
            error /= len(patterns)
            print("{:<3} {:8.5f} {:2}\n".format(i, error, misses))

N = 0.0001

patterns = np.array([
        [[0,0], [0]],
        [[1,1], [0]],
        [[0,1], [1]],
        [[1,0], [1]]
    ])

mlp = MLP(2, 2, 1)
mlp.train(patterns, 10000, N)

# for i in range(4):
#     print(mlp.feed_forward(patterns[i][0]))
