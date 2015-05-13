""" Logistic Regression
This is an implementation of a Logistic Regression model using Newton's Method to iterate
through a data set until convergence on model parameters.  The target data set is
adult demographic information containing ~48k data points.  The goal is to predict whether
the person has an annual income less than or greater than/equal to $50k per year.
For simplification, I used only the features concerning age, final weight (accounts for
race, age, and gender), years of education, and work hours per work.  I then used 70% of
data for training and 30% for testing.  """

#!/usr/bin/python
import fileinput
import csv
import math
import random
import numpy as np
import pandas as pd

''' Constants '''
INPUT_FILE = 'adult.data'
HEADER = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
TRAINING_DATA = [0,1,3,4,6,7,9]
MIN_CHG = 0.1

''' This function is for scrubbing data cells reported as "None" to be equal to "0". '''
def scrub_none(string):
	if(string == None):
		return 0
	else:
		return int(string)

''' This function initializes the main dataframe to a slimmed down dataframe for the
logistic regression model.  See notes at top for information on what features are used. '''
def init_df(df):
	data = []
	for row in df.iterrows():
		age = scrub_none(row[1]['age'])
		fnlwgt = scrub_none(row[1]['fnlwgt'])
		ed_num = scrub_none(row[1]['education-num'])
		hours_pw = scrub_none(row[1]['hours-per-week'])
		
		# Output values for training / testing
		if(row[1]['income'] == '>50K'):
			income_group = 1
		else:
			income_group = 0
		
		# Add training (1) vs. testing (0) column
		if ((row[0] % 10) in TRAINING_DATA):
			training = 1
		else:
			training = 0
	
		data.append([age,fnlwgt,ed_num,hours_pw,income_group, training])
	
	new_headers = ['age','fnlwgt','education-num','hours-per-week','income','training']
	scrubbed_df = pd.DataFrame(data, columns = new_headers)
	return scrubbed_df

''' Initializes theta at 0 for the beginning of the algorithm. '''
def init_theta(headers):
	# eliminates the income column
	if(headers[-1] == 'income'):
		headers = headers[:-1]
	
	new_headers = []
	for each in headers:
		new_headers.append(0)

	return new_headers

''' initialize gradient as array of 0's '''
def init_grad(len):
	grad = []
	
	for i in range(len):
		grad.append(0)

	return grad

''' initialize hessian as array of arrays of 0's '''
def init_hess(len):
	hess = []
	
	for i in range(len):
		hess.append(init_grad(len))

	return hess

''' Logistic function for calculating g(theta_transpose * x) '''
def logistic_function(theta, x):
	dot = np.dot(theta[:-2], x[:-2]) # do not compute dot product on income or training columns
	
	# To prevent math overflow errors
	if dot < -100:
		return 0
	else:
		return (1/(1+math.exp(-dot)))

''' Construct gradient of logistic function '''
def gradient(theta, df):
	length = len(df.columns)
	grad = init_grad(length - 2)
	
	for row in df.iterrows():  
		row_index = row[0]
		if(df.ix[row_index]['training'] == 1):
			y = df.ix[row_index][length - 1] 
			exp = y - logistic_function(theta, df.ix[row_index])
			i = 0
			while i < (length - 2):
				x = df.ix[row_index][i]
				grad[i] += (exp * x)
				i += 1
	return grad

''' Construct Hessian of logistic function '''
def hessian(theta, df):
	length = df.columns.size
	hess = init_hess(length - 2)
	
	for row in df.iterrows():
		row_index = row[0]
		if(df.ix[row_index]['training'] == 1):
			y = df.ix[row_index][length - 1]
			hx = logistic_function(theta, df.ix[row_index])
			exp = -1 * hx * (1 - hx)
			i = 0
			while i < length - 2:
				xi = df.ix[row_index][i]
				j = 0
				while j < length - 2:
					xj = df.ix[row_index][j]
					hess[i][j] += (exp * xi * xj)
					j += 1
				i += 1
	return hess

''' Iteratively calculates 
Theta(t+1) = Theta(T) - (gradient(theta_transpose * x) * inverse(hessian(theta_transpose * x)))
until convergence is reached.  At convergence, returns the final values of theta parameters. '''
def newtons_method(df, theta):
	length = len(df.index)
	y_delta = 1
	
	while y_delta > MIN_CHG:
		g = gradient(theta, df)
		h = hessian(theta, df)
		h_inv = np.linalg.inv(h)
		delta = np.dot(h_inv, g)
		delta = np.append(delta, [0,0])
		new_theta = theta + delta
		
		diff_sum = 0
		for row in df.iterrows():
			y = logistic_function(theta, row[1].values)
			new_y = logistic_function(new_theta, row[1].values)
			diff_sum += abs(new_y - y)
		
		y_delta = diff_sum / length # take average change in y as y_delta
		theta += delta
	
	return theta

""" --- MAIN FUNCTION --- """
''' Loading the csv data into a pandas DataFrame '''
data = []
for row in csv.reader(fileinput.input(INPUT_FILE), delimiter=','):
	data.append(row)

''' Main scrubbing of data and initialization of theta '''
main_df = pd.DataFrame(data, columns = HEADER)
scrubbed_df = init_df(main_df)
headers = list(scrubbed_df.columns)
theta = init_theta(headers)
final_theta = newtons_method(scrubbed_df, theta)

''' Use final parameters to make predictions '''
prediction = []
for row in scrubbed_df.iterrows():
	if (row[1]['training'] == 0):
		prediction.append(float(logistic_function(final_theta, list(row[1].values))))
	else:
		prediction.append(5.0)
scrubbed_df['prediction'] = prediction

''' Compute number of correct predictions '''
correct = 0
predictions = 0
for row in scrubbed_df.iterrows():
	if(row[1]['training'] == 0):
		predictions += 1
		if(row[1]['income'] == 1 and row[1]['prediction'] > 0.99):
			correct += 1
		elif(row[1]['income'] == 0 and row[1]['prediction'] < 0.01):
			correct += 1

''' Print Results '''	
print "Number of correct predictions: " + str(correct)
print "Number of total predictions: " + str(predictions)
print "Percent Correct: " + str(round(float(correct * 100) / predictions, 2)) + "%"