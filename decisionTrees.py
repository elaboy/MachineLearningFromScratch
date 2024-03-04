import numpy as np
from enum import Enum

class TitanicDecisionTree():
    def __init__(self, p_class: np.ndarray, sex: np.ndarray, age: np.ndarray,
                siblings_spouses_aboard: np.ndarray, parents_children_aboard: np.ndarray, 
                fare: np.ndarray, survived: np.ndarray, get_Is=False) -> None:
        
        self.p_class = ("PassengerClass", p_class)
        self.sex = ("Sex", sex)
        self.age = ("Age", age)
        self.siblings_spouses_aboard = ("SiblingsSpousesAboard", siblings_spouses_aboard)
        self.parents_children_aboard = ("ParentsChildrenAboard", parents_children_aboard)
        self.fare = ("Fare", fare)
        self.survived = ("survived", survived)
        self.root = self.get_root(get_Is)
        self.built_tree = None

    def get_root(self, print_Is=False) -> None:
        '''
        Returns the root of the tree.
        '''
        #calculate all the mutual information and return the highest one
        features = [self.p_class, self.sex, self.age, self.siblings_spouses_aboard,
                    self.parents_children_aboard, self.fare]
        mutual_informations = []
        for feature in features:
            mutual_informations.append(self.calculate_mutual_information(feature[1], self.survived[1]))
        highest_mutual_information = max(mutual_informations)
        if print_Is:
            for i in range(len(mutual_informations)):
                print(f'{FeatureType(i).name}: {mutual_informations[i]}')
        return (FeatureType(mutual_informations.index(highest_mutual_information)).name, highest_mutual_information)

    def build_tree(self, subset):
        '''
        Builds a decision tree using the mutual information of the leafs.
        '''
        root = TreeNode(self.root[0], self.root[1])
        self.build_subtree(root, subset)
        self.built_tree = root
        return root

    def build_subtree(self, node, subset):
        '''
        Recursively builds subtrees.
        '''
        if not subset:
            return
        feature, _ = subset.pop(0)
        node.add_child(feature, 1)
        node.add_child(feature, 0)
        self.build_subtree(node.children[0], subset[:])
        self.build_subtree(node.children[1], subset[:])

    @staticmethod
    def calculate_mutual_information(x: np.ndarray, y: np.ndarray) -> float:
        '''
        Computes the mutual information between two random variables X and Y.
        X: Feature variable
        y: Target variable (survived or not)
        '''
        N = x.shape[0]
        
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
    
    @staticmethod
    def calculate_mutual_information_tuple(x: np.ndarray, y: np.ndarray) -> float:
        '''
        Computes the mutual information between two random variables X and Y.
        X: Feature variable
        y: Target variable (survived or not)
        '''
        N = len(x.tolist())
        
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

   
    
    def vertical_print(self):
        def print_node(node, prefix='', is_tail=True):
            print(prefix + ('└── ' if is_tail else '├── ') + f"Is {node.feature[0]}?")
            if node.value == 0:
                print(prefix + '    └── Didn\'t survive')
            else:
                prefix += '    ' if is_tail else '│   '
                for i, child in enumerate(node.children):
                    print_node(child, prefix, i == len(node.children) - 1)

        root = self.build_tree([(feature, self.calculate_mutual_information_tuple(feature[1], self.survived[1]))
                                for feature in [self.p_class, self.sex, self.age, self.siblings_spouses_aboard,
                                                self.parents_children_aboard, self.fare]])
        print_node(root)

class TreeNode():
    def __init__(self, feature, value):
        self.feature = feature
        self.value = value
        self.children = []

    def add_child(self, feature, value):
        self.children.append(TreeNode(feature, value))

def cross_validation(tree: TreeNode, k: int, features: np.ndarray, target: np.ndarray) -> float:
    '''
    Performs 10-fold cross validation on the decision tree.
    '''
    features = features
    target = target
    #divide features and target into k folds
    print(features.shape, target.shape)
    features = np.array_split(features, k, 1)
    target = np.array_split(target, k, 1)
    print(features[0].shape, target[0].shape)
    #initialize accuracy
    accuracy_scores = []

    #perform cross validation
    for i in range(1):
        # Determine the indices for the current fold
        start_index = i * k
        end_index = start_index + k

        # Extract the test data and labels for this fold
        testing_features = features[start_index:end_index]
        testing_target = target[start_index:end_index]

        # Extract the training data and labels for this fold
        training_features = np.concatenate([features[:start_index], target[end_index:]])
        training_target = np.concatenate([features[:start_index], target[end_index:]])

        y_pred = []
        for i in testing_features:    
            for row in i:
                print(row)
                y_pred.append(predict(tree, row))

        # print(y_pred, testing_target[0])

        # Calculate accuracy for this fold
        correct_predictions = np.sum(np.array(y_pred) == np.array(testing_target))
        accuracy = correct_predictions / len(testing_target)

        # Store accuracy score for this fold
        accuracy_scores.append(accuracy)

    return accuracy / k

def predict(tree, sample):
    """
    Predicts the outcome for a single sample using the decision tree.
    
    Parameters:
        - tree (TreeNode): The root node of the decision tree.
        - sample (numpy.ndarray): The sample to predict the outcome for.
    
    Returns:
        - str: The predicted outcome.
    """
    # Traverse the tree until a leaf node is reached
    current_node = tree.built_tree
    while current_node.children:
        # Get the feature index and value of the current node
        if current_node.feature == "Sex":
            feature_index = 1
        else:
            feature_index = FeatureType(current_node.feature[0]).value
        
        # print(sample)
        feature_value = sample[feature_index]
        
        # Determine which child node to follow based on the feature value
        if feature_value == 1:
            current_node = current_node.children[0]
        elif feature_value ==0:
            current_node = current_node.children[1]
        else:
            raise ValueError("Invalid feature value")
    
    # Return the predicted class label of the leaf node
    return current_node.value

class FeatureType(Enum):
    PassengerClass = 0
    Sex = 1
    Age = 2
    SiblingsSpousesAboard = 3
    ParentsChildrenAboard = 4
    Fare = 5