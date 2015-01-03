""" This is a demonstration of a decision tree algorithm for the UCI Mushroom
Classification Data set found at: https://archive.ics.uci.edu/ml/datasets/Mushroom"""
#!/usr/bin/python
import fileinput
import csv
import copy
import pandas as pd

""" Constants """
INPUT_FILE = 'expanded.txt'
MIN_LEAF_SIZE = 20

""" Gini = 1 - sum(proportion**2) is used as the measure of purity for determining
which attribute to divide the decision on next. """
def calc_gini(df, attribute):
	vals = set(df[attribute].values)
	total = len(df)

	fracs = {}
	for each in vals:
		subtotal = float(len(df[df[attribute] == each]))
		frac = subtotal / total
		fracs[each] = frac
	
	fracs_sum = 0
	for each in fracs.values():
		fracs_sum += (each**2)
	return 1 - fracs_sum

""" This function iterates through the currently available attributes, calculates the 
Gini, and returns the attribute with the highest Gini score. """
def find_best_attr(df, curr_headers):
	max_gini = 0.0
	for each in curr_headers:
		gini = float(calc_gini(main_df, each))
		if gini > max_gini:
			max_gini = gini
			attr = each
	return attr

""" This is the recursive function that continues to divide the tree until the 
MIN_LEAF_SIZE condition is met. """
def split_tree(df, attr, headers, headers_used):
	if len(df[attr]) > MIN_LEAF_SIZE:
		attr_vals = set(df[attr].values)
		for each in attr_vals:
			curr_headers = headers[:] # make copy of headers for this branch of tree
			curr_headers.remove(attr)
			curr_headers_used = copy.deepcopy(headers_used)
			curr_headers_used[attr] = each
			
			best = find_best_attr(df[df[attr] == each], curr_headers)
			split_tree(df[df[attr] == each], best, curr_headers, curr_headers_used)
			curr_headers_used = headers_used
	else:
		print "----------"
		print "Headers: " + str(headers_used)
		print "Leaf value: " + str(len(df))
		print "=========="
		return df


""" MAIN FUNCTION """
header = ['edible?','cap-shape','cap-surface','cap-color','bruises?','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']
data = []

""" Loading the csv data into a pandas DataFrame. """
for row in csv.reader(fileinput.input(INPUT_FILE), delimiter=','):
	data.append(row)

main_df = pd.DataFrame(data, columns = header)
headers = list(main_df.columns)

attr = find_best_attr(main_df, headers)
split_tree(main_df, attr, headers, {})