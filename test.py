#!/usr/bin/env python

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

#number of times to test
iterations = 100

#load the data file.
df = pd.read_csv("feature.csv", header = 0)

#shuffle
df = df.sample(frac=1).reset_index(drop=True)

#uncomment to remove the frequency feature.
#df = df.drop(['frequency'],1)

#Test for different K values
for k in range (1, 8):
	rate = 0
	for i in range (0, iterations):
		#training set
		#https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas
		mask = np.random.rand(len(df)) < 0.8
		train = df[mask]
		#test set
		test = df[~mask]

		#separate x(features) and y (expected result)
		y = train['class']
		x = train.drop(['class'],1)

		#train
		knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric="euclidean", metric_params=None, n_jobs=1, n_neighbors=k, p=2, weights='uniform')

		knn.fit(x,y)

		#remove y from the test data
		test_no_class = test.drop(['class'],1)

		#predict
		a = knn.predict(test_no_class)
		
		#measure the accuracy
		rate += sum(a == test['class'])/float(len(test) * iterations)
	print k
	print rate
