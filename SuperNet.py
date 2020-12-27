import math
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

    def initialize_parameters(self, optimizer='adam', random_state=0):
        np.random.seed(random_state)

        self.W = {}
        self.b = {}
        for l in range(1, self.L + 1):
            self.W['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2 / self.layer_dims[l-1])
            self.b['b' + str(l)] = np.zeros((self.layer_dims[l], 1))

        if optimizer == 'adam':
            self.V = {}
            self.S = {}
            for l in range(1, self.L + 1):
                self.V['dW' + str(l)] = np.zeros(self.W['W' + str(l)].shape)
                self.V['db' + str(l)] = np.zeros(self.b['b' + str(l)].shape)
                self.S['dW' + str(l)] = np.zeros(self.W['W' + str(l)].shape)
                self.S['db' + str(l)] = np.zeros(self.b['b' + str(l)].shape)

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

    def update_parameters(self, alpha, optimizer='adam'):
        if optimizer.lower() == 'gd':
            for l in range(1, self.L+1):
                self.W['W' + str(l)] = self.W['W' + str(l)] - alpha * self.dW['dW' + str(l)]
                self.b['b' + str(l)] = self.b['b' + str(l)] - alpha * self.db['db' + str(l)]

        elif optimizer.lower() == 'adam':
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8

            V_corrected = {}
            S_corrected = {}

            for l in range(1, self.L + 1):
                self.V['dW' + str(l)] = beta1 * self.V['dW' + str(l)] + (1 - beta1) * self.dW['dW' + str(l)]
                self.V['db' + str(l)] = beta1 * self.V['db' + str(l)] + (1 - beta1) * self.db['db' + str(l)]
                V_corrected['dW' + str(l)] = self.V['dW' + str(l)] / (1 - np.power(beta1, self.t))
                V_corrected['db' + str(l)] = self.V['db' + str(l)] / (1 - np.power(beta1, self.t))

                self.S['dW' + str(l)] = beta2 * self.S['dW' + str(l)] + (1 - beta2) * np.square(self.dW['dW' + str(l)])
                self.S['db' + str(l)] = beta2 * self.S['db' + str(l)] + (1 - beta2) * np.square(self.db['db' + str(l)])
                S_corrected['dW' + str(l)] = self.S['dW' + str(l)] / (1 - np.power(beta2, self.t))
                S_corrected['db' + str(l)] = self.S['db' + str(l)] / (1 - np.power(beta2, self.t))

                self.W['W' + str(l)] = self.W['W' + str(l)] - alpha * np.divide(V_corrected['dW' + str(l)], np.sqrt(S_corrected['dW' + str(l)]) + epsilon)
                self.b['b' + str(l)] = self.b['b' + str(l)] - alpha * np.divide(V_corrected['db' + str(l)], np.sqrt(S_corrected['db' + str(l)]) + epsilon)

    def random_batches(self, X, Y, batch_size=128, random_state=0):
        np.random.seed(random_state)
        m = X.shape[-1]
        self.batches = []

        permutation = list(np.random.permutation(m))
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation].reshape(1, m)

        num_batches = math.floor(m / batch_size)
        for k in range(num_batches):
            X_batch = X_shuffled[:, k * batch_size:(k + 1) * batch_size]
            Y_batch = Y_shuffled[:, k * batch_size:(k + 1) * batch_size]
            batch = (X_batch, Y_batch)
            self.batches.append(batch)

        if m % batch_size != 0:
            X_batch = X_shuffled[:, batch_size * num_batches:]
            Y_batch = Y_shuffled[:, batch_size * num_batches:]
            batch = (X_batch, Y_batch)
            self.batches.append(batch)

    def fit(self, X_train, y_train, epochs, alpha=0.01, C=0, keep_prob=1, batch_size=128, optimizer='adam', random_state=0, verbose=0):
        # TEMPORARY FIX FOR MINI BATCH GD
        m = X_train.shape[1]
        batch_size = m

        self.initialize_parameters(optimizer=optimizer, random_state=random_state)
        self.costs = []

        self.t = 1
        seed = 10
        # GRADIENT DESCENT
        if batch_size == X_train.shape[1]:
            for i in range(1, epochs+1):
                self.forward_propagation(X_train, keep_prob=keep_prob)
                self.backward_propagation(y_train, C=C, keep_prob=keep_prob)
                self.update_parameters(alpha, optimizer=optimizer)
                cost = self.get_cost(y_train, C)
                self.costs.append(cost)

                # PRINT COST FOR FEEDBACK WHILE TRAINING
                if verbose != 0:
                    verbose = int(verbose)
                    if i % verbose == 0:
                        print(f'Epoch: {i}/{epochs}. Cost: {cost}')

        else:
            for i in range(1, epochs+1):
                seed = seed + 1
                self.random_batches(X_train, y_train, batch_size=batch_size, random_state=seed)
                for batch in self.batches:
                    (X_batch, Y_batch) = batch
                    self.forward_propagation(X_batch, keep_prob=keep_prob)
                    self.backward_propagation(Y_batch, C=C, keep_prob=keep_prob)
                    self.update_parameters(alpha, optimizer=optimizer)
                    self.t = self.t + 1
                    cost = self.get_cost(Y_batch, C)
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
