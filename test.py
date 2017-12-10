#!/usr/bin/env python

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

#n value
n = 3
iterations = 100

df = pd.read_csv("feature.csv", header = 0)
#shuffle
df = df.sample(frac=1).reset_index(drop=True)

rate = 0
for i in range (0, iterations):
	#training set
	#https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas
	mask = np.random.rand(len(df)) < 0.8
	train = df[mask]
	#test set
	test = df[~mask]

	y = train['class']
	x = train.drop(['class'],1)

	knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric="euclidean", metric_params=None, n_jobs=1, n_neighbors=n, p=2, weights='uniform')

	knn.fit(x,y)

	test_no_class = test.drop(['class'],1)

	a = knn.predict(test_no_class)
	rate += sum(a == test['class'])/float(len(test) * iterations)
	
print rate
