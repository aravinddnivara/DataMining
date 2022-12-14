import sys
import operator
import re, string
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


numUsers = 693208
numItems = 145302

# Main program
if __name__ == '__main__':

	df = pd.read_csv("train_rating.txt", sep=",")

	# Delete unused columns
	del df['date']
	# del df['train_id']
	#
	# del df['test_id']


	# Set the rating scale and create the data for Surprise to use
	reader = Reader(rating_scale=(1, 5))
	data = Dataset.load_from_df(df[['user_id', 'business_id', 'rating']], reader)

	factors = 50

	train_set = data.build_full_trainset()

	# Use SVD with surprise
	algo = SVD(n_factors=factors)
	algo.fit(train_set)

	f = open('SVDOutput.csv','w')
	f.write("test_id,rating\n")
	dfTest = pd.read_csv("test_rating.txt", sep=",")
	for i in range(len(dfTest)) :
		prediction = algo.predict(dfTest.at[i,'user_id'],dfTest.at[i,'business_id'],r_ui=4,verbose=True)
		predRating = prediction.est
		f.write(str(i)+","+str(predRating)+'\n')

# testdata = Dataset.load_from_df(dfTest[['user_id', 'business_id']], reader)
# result=algo.test(testdata)
# print(result)
f.close()