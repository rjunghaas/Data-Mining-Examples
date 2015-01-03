import fileinput
import csv
import pandas as pd
import random
import math
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np

try:
    import json
except ImportError:
    import simplejson as json

# CONSTANTS
DATA_FILE = "" # Need to add in file to read data from
SECONDS_IN_MONTH = 60 * 60 * 24 * 30
NUM_CLUSTERS = 5

# In order to not over-weight values that are large or under-weight smaller values,
# each property should be normalized.  We have used the following formula for this: 
# Mean-centering:  (Val - Mean) / (Max - Min).
# This function recreates the DataFrame by mean-centering every value in the main_df.
def mean_center(df, means):
	headers = df.columns
	max = df.max()
	min = df.min()
	mean_centered_list = []
	
	for col in headers:
		dict_entry = {}
		mean_centered_list.append((df[col] - means.ix[col]) / (max[col] - min[col]))
	
	mean_centered_df = pd.DataFrame(mean_centered_list)
	return mean_centered_df.T # mean_centered_df is made from list of cols; have to transpose to return cols to columns

# This function randomly selects k (for number of clusters) data points to be the
# initial centroids.  These will be the start of our k-means clustering algorithm.
def pick_centroids(df, k):
	centroids = []
	i = 0
	indices = list(df.index.values) # get index values of filtered_df rows
	index = random.sample(indices, k)
	
	for i in index:
		centroids.append(df.ix[i])
	
	return centroids

# Calculates Euclidean Distance multiplied by correlation
# I wanted to use the correlation of each term in order to provide weighting for properties
# that were more tied to revenue.  This is multiplied by the distance of the terms from
# each other, then squared and summed.  Finally, the square root of the sum is taken
# to calculate the total distance between two points.
def calc_distance(corr_arr, user, centroid):
	i = 0
	radical = 0
	try:
		while i < (len(corr_arr.index) - 4): # don't include revenue, revenue per month, and training data which are at end of Series
			term = float(round(corr_arr.ix[i] * ((float(centroid.ix[i]) - float(user.ix[i]))),10))
			term_sq = term**2
			radical += term_sq
			i += 1
	
		return math.sqrt(radical)
	except IndexError: # captures case where len < 2
		return 0

# Calculates the mean for the values in a cluster. This value can be used as the new
# centroid once clusters are formed. If there are no values in the cluster, the centroid
# value defaults to 0. Depending on the application, we could use a min, max, or average
# calculation rather than the mean calculation.
def centroid(xs, keys):
    if not xs: # if no users in cluster, make centroid = DataFrame of zeros.
    	length = len(keys)
    	zeros = pd.DataFrame(np.zeros((1, length)), columns = keys).transpose()
    	zeros_list = [zeros] # list of pandas object(s)
    	cluster_df = pd.concat(zeros_list, axis = 1, ignore_index = True) # axis = 1 means concatenate along columns
    else:
    	cluster_df = pd.concat(xs, axis = 1, ignore_index = True)
    	
    return cluster_df.mean(axis=1)


# This function initializes a list of lists to represent the clusters. 
# Then, it iterates through the values in the clusters, estimates the distance between
# each point and the centroids for the clusters. It then places the point in the nearest
# cluster. After iterating through all of the points, it returns the clusters.
def cluster(xs, centroids, correlation_array):
    print "Clustering"
    clusters = [[] for c in centroids]

    for row_index, row in xs.iterrows(): 
        # find the closest cluster to row
        distances = {}
        
        for cluster_id, c in enumerate(centroids):
        	dist = calc_distance(correlation_array, row, c)
        	distances[cluster_id] = dist
        cluster_id = min(distances, key = distances.get)
        
        # place row in cluster
        clusters[cluster_id].append(row)
	
    return clusters # list of list of Series


# This function initializes values for the terminating condition and an initial observed
# error. It then runs clusters to calculate distances between points and centroids and
# move data points to the nearest cluster. The new centroid is calculated for
# new_clusters. Lastly, observed_error is calculated by summing the difference between
# the new and old centroids. Centroids is then replaced with new_centroids, and the
# process iterates again until the terminating condition is met.
def iterate_centroids(xs, centroids, correlation_array):
    err = .001 # minimum amount of allowed centroid movement
    #err = .1
    observed_error = 10000000000 # Initialize: maxiumum amount of centroid movement
    new_clusters = [[] for c in centroids] # Initialize: clusters
    iterations = 0
    
    while (observed_error > err):
        new_clusters = cluster(xs, centroids, correlation_array) # moves points to nearest clusters
        keys = correlation_array.keys()
        new_centroids = []
        for each in new_clusters:
        	cent = centroid(each, keys)
        	new_centroids.append(cent)
        
        obs_err = []
        for new, old in zip(new_centroids, centroids):
        	obs_err.append(float(calc_distance(correlation_array, new, old)))	
        observed_error = max(obs_err) 
        
        print "Observed Error"
        print observed_error
        
        centroids = new_centroids
        iterations += 1
        print "iteration: %s" % str(iterations)
		
    return (centroids, new_clusters)

# Function to select the color when visualizing the clusters
def select_color(num):
	if num == 0:
		return 'b'
	elif num == 1:
		return 'g'
	elif num == 2:
		return 'r'
	elif num == 3:
		return'm'
	elif num == 4:
		return 'y'
	else:
		return 'k'

# MAIN FUNCTION
data = []
count = 0

""" Need to add code here to read through rows of CSV and convert to list of lists. """
	data.append(insert_data)
	
print str(count) + " records captured."

# Convert list of lists to Pandas DataFrame and run correlation.  Output will be Pandas Series
headers = [] # insert headers for DataFrame here.
main_df = pd.DataFrame(data, columns = headers)
mean_df = main_df.mean(0)

mean_centered_df = mean_center(main_df, mean_df)
corr_df = main_df_copy[main_df['Training Data'] == 1] # select rows that were marked as Training Data
correlation_array = corr_df.corrwith(corr_df[]) # select one column that we want to weight more strongly the correlation with for our distance function

# Main functions of k-means algorithm
initial_centroids = pick_centroids(mean_centered_df, NUM_CLUSTERS)
final_centroids, final_clusters = iterate_centroids(mean_centered_df, initial_centroids, correlation_array)

# Re-map mean_centered final clusters to main_df and make new final clusters
new_final_clusters = [[] for c in final_clusters]
clus_num = 0
for clus in final_clusters:
	for cust in clus:
		num = int(cust.name)
		new_final_clusters[clus_num].append(main_df.ix[num])
	clus_num += 1

# Recalculate final_centroids from re-mapped final_clusters
final_centroids = []
for each in new_final_clusters:
	fin_cent = centroid(each, headers)
	final_centroids.append(fin_cent)

# Display results
for centroid, cluster in zip(final_centroids, new_final_clusters):
    print "Cluster size: %r" % str(len(cluster))

# Plot results
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
cluster_num = 0
for clus in new_final_clusters:
	for each in clus:
		color = select_color(cluster_num)
		ax1.scatter(each.ix[''], each.ix[''], color = color) # add columns for x and y axes
	cluster_num += 1
ax1.set_xlim([0, 500000000]) # set scale for x axis
ax1.set_ylim([0, 50000]) # set scale for y axis
ax1.set_xlabel('') # set label for x axis
ax1.set_ylabel('') # set label for y axis
plt.show()