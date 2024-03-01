import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

def load_titanic() -> pd.DataFrame:
    '''
    Loads the titanic data and returns it as a dataframe 
    ontaining the following information about 887 passengers:
    1) whether they survived or not (1 = survived, 0 = deceased),
    2) passenger class, 
    3) gender (0 = male, 1 = female),
    4) age,
    5) number of siblings/spouses aboard, 
    6) number of parents/children aboard, 
    and 7) fare
    '''
    data = pd.read_csv('Files/titanic_data.csv')
    return data

def transform_features(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Transforms the features of the titanic data into binary variables.
    '''
    #Transform the Pclass feature into a binary variable
    data['Pclass'] = data['Pclass'].apply(
        lambda x: True if x >= 2.305524 else False)
    
    #Transform tbe Sex feature into a binary variable
    data['Sex'] = data['Sex'].apply(
        lambda x: True if x == 1 else False)
    
    #Transform the Age feature into a binary variable
    data['Age'] = data['Age'].apply(
        lambda x: True if x >= 29.471443 else False)
    
    #Transform the Siblings/Spouses Aboard feature into a binary variable
    data['Siblings/Spouses Aboard'] = data['Siblings/Spouses Aboard'].apply(
        lambda x: True if x > 0 else False)
    
    #Transform the Parents/Children Aboard feature into a binary variable
    data['Parents/Children Aboard'] = data['Parents/Children Aboard'].apply(
        lambda x: True if x > 0 else False)
    
    #Transform the Fare feature into a binary variable
    data['Fare'] = data['Fare'].apply(
        lambda x: True if x >= 34.30542 else False)
    
    return data

def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Computes the mutual information between two random variables X and Y.
    X: Feature variable
    y: Target variable (survived or not)
    '''
    N = len(x)
    
    # Count occurrences for each combination of x and y
    joint_counts = np.zeros((2, 2))
    for i in range(N):
        joint_counts[x[i], y[i]] += 1

    # Compute probabilities
    joint_probs = joint_counts / N
    marginal_x = np.sum(joint_probs, axis=1)
    marginal_y = np.sum(joint_probs, axis=0)

    # Compute mutual information using the definition: 
    #I(xj, y) = sum(p(xj, y) * log2(p(xj, y) / (p(xj) * p(y))))
    mutual_information = 0
    for i in range(2):
        for j in range(2):
            if joint_probs[i, j] > 0:
                mutual_information += joint_probs[i, j] * np.log2(joint_probs[i, j] / 
                                                                  (marginal_x[i] * marginal_y[j]))

    return mutual_information


from enum import Enum

class Tree():
    def __init__(self, features: np.ndarray, target: np.ndarray):
        self.features = features
        self.target = target
        self.tree = self.sort_features_by_mutual_information()

    def sort_features_by_mutual_information(self) -> np.ndarray:
        '''
        Sorts the features by mutual information with the target variable.
        '''
        mutual_informations = []

        transform = lambda x: 1 if x == True else 0
        for i in range(len(self.features[1])):
            transformed_numerical = []
            for j in self.features[:,i]:
                transformed_numerical.append(transform(j))
            mutual_informations.append((i, mutual_information(transformed_numerical, self.target)))
        
        return mutual_informations

    def print_tree(self) -> None:
        '''
        Prints the tree.
        '''
        tree = sorted(self.tree, key=lambda x: x[1], reverse=True)
        for category, I in tree:
            category = FeatureType(category)
            print(f'{category}: {I}')
    
    def build_tree(self) -> np.ndarray:
        '''
        Builds a decision tree using the mutual information of the features.
        '''
        
    
class FeatureType(Enum):
    PassengerClass = 0
    Gender = 1
    Age = 2
    SiblingsSpousesAboard = 3
    ParentsChildrenAboard = 4
    Fare = 5

# def mutual_information_estimation(X: np.ndarray, Y: np.ndarray, bins: int = 10) -> float:
#     '''
#     Estimates the mutual information between two random variables X and Y.
#     '''
#     #Create a 2D histogram of the two variables
#     H, x_edges, y_edges = np.histogram2d(X, Y, bins = bins)
    
#     #Compute the joint probability distribution
#     Pxy = H / float(np.sum(H))
    
#     #Compute the marginal probability distributions
#     Px = np.sum(Pxy, axis = 1)
#     Py = np.sum(Pxy, axis = 0)
    
#     #Compute the mutual information
#     I = 0
#     for i in range(Pxy.shape[0]):
#         for j in range(Pxy.shape[1]):
#             if Pxy[i, j] > 0:
#                 I += Pxy[i, j] * np.log(Pxy[i, j] / (Px[i] * Py[j]))
#     return I
    