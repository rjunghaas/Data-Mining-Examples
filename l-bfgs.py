""" Logistic Regression: Limited Memory BFGS
This is an implementation of a Logistic Regression model using the Limited Memory BFGS
algorithm to iterate through a data set until convergence on model parameters.  
The target data set is adult demographic information containing ~48k data points.  
The goal is to predict whether the person has an annual income less than or greater 
than/equal to $50k per year.  For simplification, I used only the features concerning 
age, final weight (accounts for race, age, and gender), years of education, and work 
hours per work.  I then used 70% of data for training and 30% for testing.  """

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
TRAINING_DATA = [0,1,3,4,6,7,9] # 70% training / 30% testing
MIN_CHG = 0.05
NUM_VALS_STORED = 5

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
		if(row[1]['income'] == ' >50K'):
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

''' Logistic function for calculating g(theta_transpose * x) '''
def logistic_function(theta, x):
	dot = np.dot(theta[:-2], x[:-2]) # do not compute dot product on income or training columns
	
	# To prevent math overflow errors
	if dot < -100.0:
		return 0
	else:
		return (1/(1+math.exp(-dot)))

''' Calculates logistic function over entire range of dataframe and takes average y value '''
def log_f(theta, df):
	sum_y = 0.0
	for row in df.iterrows():
		y = logistic_function(theta, row[1].values)
		sum_y += y
	
	return sum_y / len(df)

''' Construct gradient of logistic function '''
def gradient(theta, df):
	length = len(df.columns)
	grad = init_grad(length - 2)
	
	for row in df[df['training'] == 1].iterrows():  
		row_index = row[0]
		y = df.ix[row_index][length - 1] 
		exp = y - logistic_function(theta, df.ix[row_index])
		i = 0
		while i < (length - 2):
			x = df.ix[row_index][i]
			grad[i] += (exp * x)
			i += 1

	return grad

''' Function for calculating initial estimate for Hessian at each iteration of the algorithm based on stored values '''
def choose_h0(s, y):
	s_trans = np.transpose(s)
	y_trans = np.transpose(y)
	gamma = list(np.multiply(list(np.multiply(s_trans,y)), (y_trans * y)))
	return np.dot(gamma, np.identity(len(gamma)))

''' Function to estimate step size of next iteration of L-BFGS algorithm based on stored values from previous iterations '''
def estimate_step(vals, g, alpha, h0, k):
	# initialize variables
	q = g # start with q = gradient
	rho = []
	qa = []
	i = 0 # iteration counter over number of stored values

	# iterate over each stored value
	while i < (len(vals)):
		# iterating over stored vals and getting s & y
		s = vals[i][0] 
		y = vals[i][1] 
		
		# calculating transpose of stored values
		s_trans = np.transpose(s)
		y_trans = np.transpose(y)
		
		qa_numerator = q * s_trans
		qa_denominator = y_trans * s
		qa_denom_inv = list(np.divide(1, qa_denominator))
		rho.append(qa_denom_inv)
		
		qan = np.dot(qa_numerator, qa_denom_inv)
		qa.append(qan)
		
		delta = (-qan * y)
		q += delta
		i += 1

	r = np.dot(h0, q) # r = estimate for step size
	
	# iterate over each stored value
	i = 0
	while i < (len(vals)):
		# iterating over stored vals and getting s & y
		s = vals[i][0]
		y = vals[i][1] 
		
		# calculate transpose of y
		y_trans = np.transpose(y)
		
		beta_numerator = y_trans * r
		beta_denom_inv = rho[i]
		beta = list(np.multiply(beta_numerator, beta_denom_inv))

		alpha_beta = qa[i] - beta
		r += alpha_beta * s # update r
		i += 1

	return r

''' Main function for L-BFGS algorithm '''
def l_bfgs(df, theta):
	k = 0 # iteration counter
	vals = [] # stored values of previous thetas; to be stored as [[s, y],...]
	x = list(np.zeros(shape=len(theta)).astype(float))
	y = log_f(theta, df)
	
	y_delta = 1.0
	while y_delta > MIN_CHG:
		g = gradient(theta, df)
		
		# for first iteration, set initial step size to be alpha * gradient
		if k == 0:
			s0 = list(np.multiply(y, g))
			vals.append([s0, y])
			s = np.multiply(0.1, g)

		# for remaining iterations, proceed with L-BFGS algorithm to get step size
		else:
			h0 = choose_h0(s, y) # s is matrix
			
			# Per Wolfe Conditions, set alpha small for first iterations
			if k < 2:
				alpha = 0.1
			else:
				alpha = 1.0
			
			# estimate step direction and step size
			p = estimate_step(vals, g, alpha, h0, k)
			s = alpha * p

		# calculates new x value by incrementing x by s and adds two placeholder 0's for income and training.
		x_new = []
		i = 0
		for each in theta[:-2]:
			x_new.append(each + s[i])
			i += 1
		x_new.append(0.0)
		x_new.append(0.0)
		
		# Compute new y value from new x value
		y_new = log_f(x_new, df)
		
		# Append new y value to our stored list of values.  If we are at our limit for stored values, pop oldest stored value from list.
		if k > NUM_VALS_STORED:
			vals.pop(0)
			vals.append([s, y_new])
		else:
			vals.append([s, y_new])
		
		# set values for next iteration of algorithm
		theta = x_new
		y_delta = abs(y - y_new)
		y = y_new
		k += 1

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
final_theta = l_bfgs(scrubbed_df, theta)
print "Theta"
print final_theta

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
for row in scrubbed_df[scrubbed_df['training'] == 0].iterrows():
	predictions += 1
	if(row[1]['income'] == 1 and row[1]['prediction'] > 1.00):
		correct += 1
	elif(row[1]['income'] == 0 and row[1]['prediction'] <= 1.00):
		correct += 1

''' Print Results '''	
print "Number of correct predictions: " + str(correct)
print "Number of total predictions: " + str(predictions)
print "Percent Correct: " + str(round(float(correct * 100) / predictions, 2)) + "%"