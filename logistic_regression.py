import csv
import math
import random
class LogisticRegressionEstimator:
    '''
    A logistic regression estimator for classification tasks.
    '''

    def __init__(self, learning_rate=0.1, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None
        self.likelihoods = []


    def fit(self, features, labels):
        '''
        Fit the logistic regression model to the training data.
        '''
        # Initialize the weights
        self.weights = []
        for i in range(len(features[0])):
            self.weights.append(random.uniform(0, 1))
        self.bias = random.uniform(0, 1)

        for iter in range(self.max_iter):
            dot = []
            for i in range(len(features)):
                row = []
                for j in range(len(features[i])):
                    row.append(features[i][j] * self.weights[j])
                dot.append(row)
            
            y_pred = self.sigmoid(dot)

            grad = []
            for i in range(len(y_pred)):
                grad.append([(labels[i] - y_pred[i]) * k for k in range(len(features[i]))])
            
            #get the mean of the gradient
            grad = [sum(x) / len(y_pred) for x in zip(*grad)]

            self.weights = [self.weights[i] + self.learning_rate * grad[i] for i in range(len(self.weights))]

            likeligood = self.logLikehood(labels, y_pred)
        #for iteration in range(self.max_iter):
        #    print("iteration: " + str(iteration))
        #    dot_product = 0
        #    predictions = []
        #    for i in range(len(features)):
        #        p = self.theta_mult(features[i])
        #        # Apply the sigmoid function
        #        prediction = self.sigmoid_function(p)
#
        #        #Append prediction to the listtgood pass car 10 keep going
        #        predictions.append(prediction)
        #        
        #        #calculate the gradient for the weights and bias
        #    gradient_weights, gradient_bias = self.gradient_ascent(features, labels, predictions)
        #    self.weights = gradient_weights
        #    self.bias = gradient_bias
        #    # #calculate the gradient for the weights and bias
        #    # gradient_weights = []
        #    # for j in range(len(features[i])):
        #    #     gradient_weights.append((labels[i] - prediction) * features[i][j])
        #    # gradient_bias = float(labels[i] - prediction)
#
        #    # # Update the weights and bias
        #    # newWeights = []
        #    # for j in range(len(self.weights)):
        #    #     newWeights.append(self.weights[j] + self.learning_rate * gradient_weights[j])
        #    # self.weights = newWeights
#
        #    # self.bias = self.bias + self.learning_rate * gradient_bias
        #    l = self.logLikehood(labels, features)
        #    if l == None:
        #        continue
        #    self.likelihoods.append(l)
        #    print("Log likelihood: " + str(l) + "\n")  

    def logLikehood(self, true_y, x):
        '''
        Calculate the log likelihood of the model.
        '''
        log_likelihood = 0
        for i in range(0, len(true_y)):
            log_likelihood += ((true_y[i] * math.log(1/1+(1/self.theta_mult(x[i]))))) + ((1 - true_y[i]) * math.log(1/1+(1/self.theta_mult(x[i]))))
            
            
        return log_likelihood/len(true_y)

    def gradient_ascent(self, features, true_y, y_hat):
        '''
        Perform gradient ascent.
        '''
        gradient_weights = self.weights
        randomWeight = random.randint(0, len(self.weights) - 1)
        gradient_weights[randomWeight] += self.learning_rate
        #for i in range(len(self.weights)):
         #   gradient_weights[i] = self.weights[i] + self.learning_rate 

        gradient_bias = self.bias * self.learning_rate
        #for i in range(len(features)):
        #    for j in range(0, 6):
        #        gradient_weights[j] += (true_y[i] - (1/1+(1/(self.theta_mult(features[i]))))) * features[i][j]
        #    gradient_bias += float(true_y[i] - y_hat[i])
        return gradient_weights, gradient_bias
    
    def predict(self, features):
        '''
        Predict the labels for the input features.
        '''
        #features * weights + bias
        y_hat = 0
        for i in range(len(self.weights)):
            dot_product = self.weights[i] * features
            dot_product += self.bias
            y_hat = self.sigmoid_function(dot_product)

        return y_hat
    
    def theta_mult(self, x):
        '''
        Multiply the weights and features.
        '''
        result = 0
        for i in range(len(self.weights)):
            result += self.weights[i] * x[i]
        return result
    
    def theta_mult_float(self, x):
        '''
        Multiply the weights and features.
        '''
        result = 0
        for i in range(len(self.weights)):
            result += self.weights[i] * x

        return result

    def sigmoid_function(self, x):
        '''
        The sigmoid function.
        '''
        result = 0
        for i in range(len(self.weights)):
            result += self.weights[i] * x

        return 1 / (1 + math.exp(result * -1))

    def sigmoid(self, dot):
        '''
        The sigmoid function for the list.
        '''
        sigmoidResults = []
        for i in range(len(dot)):
            sigmoidResult = 0
            for j in range(len(dot[i])):
                sigmoidResult += self.weights[j] * dot[i][j]
            sigmoidResults.append(1 / (1 + math.exp(sigmoidResult * -1)))
        return  sigmoidResults
        
if(__name__ == '__main__'): 
    #Load training data
    #cols = Survived,Pclass,Sex,Age,Siblings/Spouses, Aboard,Parents/Children, Aboard,Fare
    #labels = Survived
    X = []
    y = []
    with open('Files\\titanic_data.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for index, row in enumerate(spamreader):
            if index == 0:
                continue
            data = row[0].split(',')
            #gets the first column of data and appends it to the labels list
            y.append(float(data[0]))
            #gets each row of data and appends it to the features list
            x = []
            for i in data[1:]:
                x.append(float(i))
            X.append(x)
    # Fit the model
    estimator = LogisticRegressionEstimator()
    estimator.fit(X, y)
    #print each weight with the corresponding feature name 
    print("PClass: " + str(estimator.weights[0]) + "\n"
          "Sex: " + str(estimator.weights[1]) + "\n"
            "Age: " + str(estimator.weights[2]) + "\n"
            "Siblings/Spouses Aboard: " + str(estimator.weights[3]) + "\n"
            "Parents/Children Aboard: " + str(estimator.weights[4]) + "\n"
            "Fare: " + str(estimator.weights[5]) + "\n")
    print("Bias: "+ str(estimator.bias))