import numpy
import pandas
import matplotlib.pyplot as plt
from utils import load_titanic

class DecisionTree:
    def __init__(self):
        self.root = None

class Leaf:
    def __init__(self, data, depth, max_depth):
        self.data = data
        self.depth = depth
        self.max_depth = max_depth
        self.left = None
        self.right = None
