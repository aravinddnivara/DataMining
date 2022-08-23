import sys
import operator
import re, string
import csv
import math
import sklearn
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# from sklearn.cross_validation import train_test_split
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
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'business_id', 'rating']], reader)


X_train , y_train = test_train_split(data ,test_size=0.3)
# train_set = data.build_full_trainset()
algo = RandomForestClassifier(n_estimators=100)
algo.fit(X_train, y_train)

#Random Forest Model-

#Import Random Forest Model



# Create a Random Forest Regressor with 100 trees (default)
# rfm-RandomForestClassifier (n_estimators=100)

#Train the model using the training sets y pred=clf.predictx_test)

# rfm.fit(x_train,y_train)
f = open('ROutput.csv','w')
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