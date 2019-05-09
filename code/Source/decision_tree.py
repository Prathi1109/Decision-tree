#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 15:35:46 2019

@author: prathibha
"""



import numpy as np
import pandas as pd

header_row=["att0","att1","att2","att3","att4","att5","Class"]
df = pd.read_csv("/Users/prathibha/Desktop/decisiontree/car.csv", delimiter=";", names=header_row)
print(df)

def entropy(probs):
    '''
    Takes a list of probabilities and calculates their entropy
    '''

    import math
    return sum( [-prob*math.log(prob, len(df.Class.unique())) for prob in probs] )
    

def entropy_of_list(a_list):
    '''
    Entropy of the tree
    '''
    from collections import Counter
    
    # Getting  unique items and their counts in the dict format:
    unique = Counter(x for x in a_list)
    
    # Convert to Proportion
    num_instances = len(a_list)
    probs = [x / num_instances for x in unique.values()]
    
    # Calculate Entropy:
    return entropy(probs)
    
tree_entropy = entropy_of_list(df['Class'])
print(tree_entropy)

def information_gain(df, split_attribute_name, target_attribute_name, trace=0):
    
    
    # Split Data by Possible Vals of Attribute:
    df_split = df.groupby(split_attribute_name)
    
    # Calculate Entropy for Target Attribute, as well as Proportion of Obs in Each Data-Split
    totalobs = len(df.index) * 1.0
    df_splitattr = df_split.agg({target_attribute_name : [entropy_of_list, lambda x: len(x)/totalobs] })[target_attribute_name]
    df_splitattr.columns = ['Entropy', 'Proportion']
    if trace: # helps understand what fxn is doing:
        print(df_splitattr)
    
    # Calculate Information Gain:
    sub_entropy = sum( df_splitattr['Entropy'] * df_splitattr['Proportion'] )
    root_entropy = entropy_of_list(df[target_attribute_name])
    return root_entropy-sub_entropy

def id3(df, target_attribute_name, attribute_names, max_class=None):
    
    ## target attribute:
    from collections import Counter
    target_class = Counter(x for x in df[target_attribute_name])
    
    # If there is only one class 
    if len(target_class) == 1:
        return list(target_class.keys())[0]
    
    # If there are no feature values
    elif df.empty or (not attribute_names):
        return max_class 
    
    # Grow the tree
    else:
        # Recursive call of this function:
        index_of_max = list(target_class.values()).index(max(target_class.values())) 
        max_class = list(target_class.keys())[index_of_max] # Maximum value of target attribute in dataset
        
        # Choose Best Attribute to split on:
        gain = [information_gain(df, attr, target_attribute_name) for attr in attribute_names]
        index_of_max = gain.index(max(gain)) 
        best_attr = attribute_names[index_of_max]
        
        # Create an empty tree and build the tree
        tree = {best_attr:{}}
        remaining_attribute_names = [i for i in attribute_names if i != best_attr]
        
        # Split dataset
        # On each split, recursively call ID3.
        # populate the empty tree with subtrees, which are the result of the recursive call
       
        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset,
                        target_attribute_name,
                        remaining_attribute_names,
                        max_class)
            tree[best_attr][attr_val] = subtree
        return tree

attribute_names = list(df.columns)
attribute_names.remove('Class')

from pprint import pprint
tree = id3(df, 'Class',attribute_names)
pprint(tree)
