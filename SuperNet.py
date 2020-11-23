import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():

    '''
    This is a simple deep neural network with
    L2 and dropout regularization if specified.

    layer_dims = [n_x, n_h1, n_h2, ... , n_y]
    and L = len(layer_dims[1:])
    '''

    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.L = len(layer_dims) - 1

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    # NOT SURE WITH THIS SO JUST CHANGE LATER
    def relu_derivative(z):
        dadz = np.zeros(z.shape)
        dadz[z > 0] = 1
        return dadz

    def get_cost(self, Y, C=0):
        m = Y.shape[1]
        logprobs = np.multiply(Y, np.log(self.A['A' + str(self.L)])) + np.multiply(1 - Y, np.log(1 - self.A['A' + str(self.L)]))
        log_term = -1/m * np.sum(logprobs)
        # IF WITH REGULARIZATION
        if C != 0:
            Wl_norm_vals = []
            for l in range(1, self.L+1):
                Wl_norm = np.sum(np.square(self.W['W' + str(l)]))
                Wl_norm_vals.append(Wl_norm)
            reg_term = C/(2*m) * np.sum(Wl_norm_vals)
            return log_term + reg_term
        # IF NO REGULARIZATION
        return log_term

    def forward_propagation(self, X, keep_prob=1, random_state=None):
        self.Z = {}
        self.A = {}
        self.D = {}
        self.A['A0'] = X
        for l in range(1, self.L):
            self.Z['Z' + str(l)] = np.dot(self.W['W' + str(l)], self.A['A' + str(l-1)]) + self.b['b' + str(l)]
            Al = self.relu(self.Z['Z' + str(l)])
            self.D['D' + str(l)] = np.random.rand(Al.shape[0], Al.shape[1]) < keep_prob
            Al = np.multiply(Al, self.D['D' + str(l)]) / keep_prob
            self.A['A' + str(l)] = Al
        self.Z['Z' + str(self.L)] = np.dot(self.W['W' + str(self.L)], self.A['A' + str(self.L-1)]) + self.b['b' + str(self.L)]
        self.A['A' + str(self.L)] = self.sigmoid(self.Z['Z' + str(self.L)])

    def backward_propagation(self, Y, C=0, keep_prob=1):
        m = Y.shape[1]
        self.dZ = {}
        self.dA = {}
        self.dW = {}
        self.db = {}
        self.dA['dA' + str(self.L)] = -np.divide(Y, self.A['A' + str(self.L)]) + np.divide(1 - Y, 1 - self.A['A' + str(self.L)])
        self.dZ['dZ' + str(self.L)] = np.multiply(self.dA['dA' + str(self.L)], self.sigmoid_derivative(self.Z['Z' + str(self.L)]))
        self.dW['dW' + str(self.L)] = 1/m * np.dot(self.dZ['dZ' + str(self.L)], self.A['A' + str(self.L-1)].T) + C/m * self.W['W' + str(self.L)]
        self.db['db' + str(self.L)] = 1/m * np.sum(self.dZ['dZ' + str(self.L)], axis=1, keepdims=True)

        for l in reversed(range(1, self.L)):
            dAl = np.dot(self.W['W' + str(l + 1)].T, self.dZ['dZ' + str(l + 1)])
            dAl = np.multiply(dAl, self.D['D' + str(l)]) / keep_prob
            self.dA['dA' + str(l)] = dAl
            self.dZ['dZ' + str(l)] = np.multiply(self.dA['dA' + str(l)], self.relu_derivative(self.Z['Z' + str(l)]))
            self.dW['dW' + str(l)] = 1/m * np.dot(self.dZ['dZ' + str(l)], self.A['A' + str(l-1)].T) + C/m * self.W['W' + str(l)]
            self.db['db' + str(l)] = 1/m * np.sum(self.dZ['dZ' + str(l)], axis=1, keepdims=True)

    def update_parameters(self, alpha):
        for l in range(1, self.L+1):
            self.W['W' + str(l)] = self.W['W' + str(l)] - alpha * self.dW['dW' + str(l)]
            self.b['b' + str(l)] = self.b['b' + str(l)] - alpha * self.db['db' + str(l)]

    def fit(self, X_train, y_train, epochs, alpha=0.01, activation='relu', C=0, keep_prob=1, random_state=None, verbose=0):
        # SET RANDOM STATE
        if random_state != None:
            np.random.seed(random_state)

        # WEIGHT INITIALIZATION
        self.W = {}
        self.b = {}
        for l in range(1, len(self.layer_dims)):
            self.W['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2 / self.layer_dims[l-1])
            self.b['b' + str(l)] = np.zeros((self.layer_dims[l], 1))

        self.costs = []

        # GRADIENT DESCENT
        for i in range(1, epochs+1):
            self.forward_propagation(X_train, keep_prob=keep_prob)
            self.backward_propagation(y_train, C=C, keep_prob=keep_prob)
            self.update_parameters(alpha)
            cost = self.get_cost(y_train, C)
            self.costs.append(cost)

            # PRINT COST FOR FEEDBACK WHILE TRAINING
            if verbose != 0:
                verbose = int(verbose)
                if i % verbose == 0:
                    print(f'Epoch: {i}/{epochs}. Cost: {cost}')
        print(f'Training done! Cost after {epochs} epochs: {self.costs[-1]}')

    def predict(self, X_test):
        self.forward_propagation(X_test)
        y_pred = self.A['A' + str(self.L)]
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        return y_pred

    def plot_cost(self):
        plt.figure(dpi=100)
        plt.plot(self.costs)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()
