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