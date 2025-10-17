class NeuralNetwork:
    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.weights = []
        self.biases = []

        for i in range(len(self.layer_dims) - 1):
            w = np.random.randn(self.layer_dims[i+1], self.layer_dims[i]) * np.sqrt(2.0 / self.layer_dims[i])
            b = np.zeros((self.layer_dims[i+1], 1))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        ez = np.exp(x - np.max(x))
        return ez / np.sum(ez, axis=0, keepdims=True)
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    
    def cross_entropy_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[y_true, range(m)] + 1e-9)
        loss = np.sum(log_likelihood) / m
        return loss
        
    def forward_pass(self, x):
        x = np.array(x)

        if x.ndim == 1:
            out = x.reshape(-1, 1)
        else:
            out = x.T 

        self.cache = {"A0": out}  

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(w, out) + b
            self.cache[f"Z{i+1}"] = z

            if i == len(self.weights) - 1:
                out = self.softmax(z)
            else:
                out = self.relu(z)

            self.cache[f"A{i+1}"] = out

        return out
    
    def backward_pass(self, y_true, learning_rate=0.01):
        m = y_true.shape[0] 
        grads_w, grads_b = [], []

        num_classes = self.weights[-1].shape[0]
        y = np.zeros((num_classes, m))
        y[y_true, np.arange(m)] = 1

        dA = self.cache[f"A{len(self.weights)}"] - y  

        for i in reversed(range(len(self.weights))):
            dZ = dA
            A_prev = self.cache[f"A{i}"]
            dW = (1/m) * np.dot(dZ, A_prev.T)
            dB = (1/m) * np.sum(dZ, axis=1, keepdims=True)

            grads_w.insert(0, dW)
            grads_b.insert(0, dB)

            if i > 0: 
                dA = np.dot(self.weights[i].T, dZ)
                dA[self.cache[f"Z{i}"] <= 0] = 0 

        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grads_w[i]
            self.biases[i] -= learning_rate * grads_b[i]


    def train(self, X, y, epochs=10, batch_size=128, lr=0.01):
        n = X.shape[0]
        for epoch in range(epochs):
            idx = np.random.permutation(n)

            X, y = X[idx], y[idx]

            for i in range(0, n, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                y_pred = self.forward_pass(X_batch)
                self.backward_pass(y_batch, learning_rate=lr)

            y_pred = self.forward_pass(X)
            loss = self.cross_entropy_loss(y_pred, y.T)
            acc = self.accuracy(X, y)*100
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Acc: {acc:.4f}%")

    def accuracy(self, X, y):
        y_pred = self.forward_pass(X)
        preds = np.argmax(y_pred, axis=0)
        return np.mean(preds == y)
