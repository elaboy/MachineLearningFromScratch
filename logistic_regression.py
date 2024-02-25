class LogisticRegressionEstimator:
    '''
    A logistic regression estimator for classification tasks.
    '''

    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-6):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        '''
        Fit the logistic regression model to the training data.
        '''
        # Initialize the weights
        self.weights = np.zeros(X.shape[1])

        # Perform gradient descent
        for i in range(self.max_iter):
            # Compute the predicted probabilities
            p = self.predict_proba(X)

            # Compute the gradient
            gradient = X.T @ (p - y)

            # Update the weights
            self.weights -= self.learning_rate * gradient

            # Check for convergence
            if np.linalg.norm(gradient) < self.tol:
                break

    