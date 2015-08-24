''' Naive Bayes
This script runs a Naive Bayes classification algorithm on the UC Irvine adult data set.
The goal is to classify users as having annual income greater than or less than/equal to
$50k/year.  I am using the following features to predict the classification of income: age,
final weight (accounts for race, age, and gender), years of education, and hours per week
of work.  I have also divided the data set into 70% training and 30% testing.
 '''
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

''' This function is for scrubbing data cells reported as "None" to be equal to "0". '''
def scrub_none(string):
	if(string == None):
		return 0
	else:
		return int(string)

''' This function initializes the main dataframe to a slimmed down dataframe for the
naive bayes model.  See notes at top for information on what features are used. '''
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
	
	# Add headers and construct dataframe
	new_headers = ['age','fnlwgt','education-num','hours-per-week', 'income','training']
	scrubbed_df = pd.DataFrame(data, columns = new_headers)
	
	# eliminate NaN values
	scrubbed_df.fillna(0)
	
	# normalize scrubbed_df
	for col in new_headers[:-2]: # do not adjust income or training columns
		min = scrubbed_df[col].min()
		max = scrubbed_df[col].max()
		scrubbed_df[col] = (scrubbed_df[col] - min) / (max - min)
	
	return scrubbed_df

''' Function for calculating conditional posterior probabilities using Bayes' Theorem.  Must enter parameter to predict on as y and 1 or 0 for predicted value '''
def calc_cond_prob(training_df, param, y, value):
    prob_param_given_y = (len(training_df[(training_df[param] == 1) & (training_df[y] == value)]) + 1) / float(len(training_df[training_df[y] == value]) + 1)
    prob_y = (len(training_df[training_df[y] == value]) + 1) / float((len(training_df) + 1))
    prob_param = (len(training_df[training_df[param] == 1]) + 1) / float((len(training_df) + 1))
    return (prob_param_given_y * prob_y) / prob_param

''' Function for calculating probability of outcome for a specific data point based on conditional probability '''
def row_predictor(row, cond_probs):
	total_prob = 1.0
	for key in cond_probs:
		if row[key] == 0:
			total_prob *= (1 - cond_probs[key])
		else:
			total_prob *= (float(row[key] * cond_probs[key]))

	return total_prob

''' Function that maximizes likelihood of classification for each data point '''
def predict(row, yes_cond_probs, no_cond_probs):
	yes_pred = row_predictor(row, yes_cond_probs)
	no_pred = row_predictor(row, no_cond_probs)
	
	if yes_pred >= no_pred:
		training_df.loc[row.name,'prediction'] = 1
		return 1
	else:
		training_df.loc[row.name,'prediction'] = 0
		return 0

""" --- MAIN FUNCTION --- """
''' Loading the csv data into a pandas DataFrame '''
data = []
for row in csv.reader(fileinput.input(INPUT_FILE), delimiter=','):
	data.append(row)

''' Main scrubbing of data and initialization of theta '''
main_df = pd.DataFrame(data, columns = HEADER)
training_df = init_df(main_df)
training_headers = list(training_df.columns)
training_headers = training_headers[:-2]

''' Calculate probabilities of being "yes" '''
yes_cond_probs = {}
# incorporates naive assumption
for param in training_headers:
	ycp = calc_cond_prob(training_df, param, 'income', 1)
	yes_cond_probs[param] = ycp		

''' Calculate probabilities of being "no" '''
no_cond_probs = {}
# incorporates naive assumption
for param in training_headers:
	ncp = calc_cond_prob(training_df, param, 'income', 0)
	no_cond_probs[param] = ncp

''' Output results '''
print "Yes Conditional Probabilities: " + str(yes_cond_probs)
print "No Conditional Probabilities: " + str(no_cond_probs)

''' Use final parameters to make predictions '''
prediction = []
for row in training_df.iterrows():
	if (row[1]['training'] == 0):
		prediction.append(predict(row[1], yes_cond_probs, no_cond_probs))
	else:
		prediction.append(5.0)

training_df['prediction'] = prediction

''' Compute number of correct predictions '''
correct = 0
predictions = 0
for row in training_df[training_df['training'] == 0].iterrows():
	predictions += 1
	if(row[1]['income'] == 1 and row[1]['prediction'] > 1.00):
		correct += 1
	elif(row[1]['income'] == 0 and row[1]['prediction'] <= 1.00):
		correct += 1

''' Print Results '''	
print "Number of correct predictions: " + str(correct)
print "Number of total predictions: " + str(predictions)
print "Percent Correct: " + str(round(float(correct * 100) / predictions, 2)) + "%"
