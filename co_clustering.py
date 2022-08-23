# This code to predict ratings for user-item pairs from the Yelp dataset 
# using the co-clustering algorithm provided by the Surprise library
# Written by Brie Hoffman, Nov 4 2017 for CMPT 741 final project


import sys
import operator
# import re, string
import csv
import math
import numpy as np
from scipy import sparse
import pandas as pd
from scipy.sparse import lil_matrix
from surprise import NMF
from surprise import SVD
from surprise import SVDpp
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import Reader
from surprise import CoClustering,Dataset, Reader

# import training set as a pandas dataframe
# train_file_path = sys.argv[1]
# dftrain = pd.read_csv(train_file_path)
# dftrain = dftrain.drop(['train_id', 'date'], axis=1)
if __name__ == '__main__':

	df = pd.read_csv("train_rating.txt", sep=",")

	# Delete unused columns
	del df['date']

# import test set as a pandas dataframe
# test_file_path = sys.argv[2]
# dftest = pd.read_csv(test_file_path)
# dftest = dftest.drop(['test_id', 'date'], axis=1)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'business_id', 'rating']], reader)

# create a trainset object 
# reader = Reader()
# data = Dataset.load_from_df(dftrain, reader)
trainset = data.build_full_trainset()

"""
# The code here in quotes was for the cross-validation gridsearch for the best
# hyperparamters


#param_grid = {'n_cltr_u': [2,3,4,5,6,7,8,9,10],
#              'n_cltr_i': [2,3,4,5,6,7,8,9,10],
#              'n_epochs': [10,20,30,40,50,60,70,80,90,100]}


# Evaluate the model with 5-fold cross validation
#data.split(5)

#grid_search = GridSearch(CoClustering, param_grid, measures=['RMSE'])
#grid_search.evaluate(data)
#print ("after grid_search.evaluate(data)")
#print_perf(perf)

#results_df = pd.DataFrame.from_dict(grid_search.cv_results)
#print(results_df) """


# create a co-clustering algorithm
algo = CoClustering(n_cltr_u=3, n_cltr_i=3, n_epochs=100)
algo.fit(trainset)


# use the trained algorithm to predict ratings for every user in the test set
f = open('testOutput.csv','w')
f.write("test_id,rating\n")
dfTest = pd.read_csv("test_rating.txt", sep=",")
for i in range(len(dfTest)) :
    prediction = algo.predict(dfTest.at[i,'user_id'],dfTest.at[i,'business_id'],r_ui=4,verbose=True)
    predRating = prediction.est
    f.write(str(i)+","+str(predRating)+'\n')

f.close()





