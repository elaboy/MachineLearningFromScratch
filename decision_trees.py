import numpy
import pandas
import matplotlib.pyplot as plt
from utils import load_titanic

class DecisionTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.root = None

class Node:
    def __init__(self, data, depth, max_depth):
        self.data = data
        self.depth = depth
        self.max_depth = max_depth
        self.left = None
        self.right = None

if __name__ == "__main__":
    titanic_dataframe = load_titanic()
    print(titanic_dataframe)