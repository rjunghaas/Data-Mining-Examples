""" Logistic Regression
This is an implementation of a Logistic Regression model using Newton's Method to iterate
through a data set until convergence on model parameters.  The target data set is
adult demographic information containing ~48k data points.  The goal is to predict whether
the person has an annual income less than or greater than/equal to $50k per year.

For simplification, I used only the features concerning age, final weight (accounts for
race, age, and gender), years of education, and work hours per work.  I then used 80% of
data for training and 20% for testing.  """

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
TRAINING_DATA = [0,1,2,3,4,5,6,7]
MIN_CHG = 0.00000001

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
		
		# Add training (1) vs. testing (2) column
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

''' Uses Euclidean distance to estimate the change in parameter values. '''
def delta_dist(delta):
	sum = 0
	for each in delta:
		sum += (each**2)
	
	return math.sqrt(sum)

''' Returns difference between two vectors '''
def diff_iterables(i, j):
	new = []
	for a in range(len(i)-2):
		new.append(i[a] - j[a])	
	
	# adds two '5's to end of vector as substitute for 'income' and 'training'
	new.append(5)
	new.append(5)
	return new

''' Logistic function for calculating g(theta_transpose * x) '''
def logistic_function(theta, x):
	dot = np.dot(theta[:-2], x[:-2]) # do not compute dot product on income or training columns
	return (1/(1+math.exp(-dot)))

''' First Derivative of Logistic function for Newton's method '''
def first_deriv(x , y, theta):
	return (y[:-2] - logistic_function(theta[:-2], x[:-2]) * x[:-2])

''' Second Derivative of Logistic function for Newton's method '''
def second_deriv(x, y, theta):
	return (-logistic_function(theta[:-2], x[:-2])) * (1 - logistic_function(theta[:-2], x[:-2])) * np.dot(x[:-2].values, x[:-2].values)

''' Iteratively calculates 
Theta(t+1) = Theta(T) - (first_deriv(theta_transpose * x) / second_deriv(theta_transpose * x))
until convergence is reached.  At convergence, returns the final values of theta parameters. '''
def newtons_method(df, theta):
	length = len(df.index)
	delta = [0.00001, 0.00001, 0.00001, 0.00001]
	
	while delta_dist(delta) > MIN_CHG:
		theta = diff_iterables(theta, delta)
	
		index = random.randint(0, length-1)
		if(df.ix[index]['training'] == 1):
			y = df.ix[index]
			x = df.ix[index]
			delta = first_deriv(x, y, theta) / second_deriv(x, y, theta)
		else:
			pass
	
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
		if(row[1]['income'] == 1 and row[1]['prediction'] == 1):
			correct += 1
		elif(row[1]['income'] == 0 and row[1]['prediction'] < 1.0):
			correct += 1

''' Print Results '''	
print "Number of correct predictions: " + str(correct)
print "Number of total predictions: " + str(predictions)
print "Percent Correct: " + str(round(float(correct * 100) / predictions, 2)) + "%"