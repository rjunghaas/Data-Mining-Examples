import random

###
# Implement simple k-means clustering using 1 dimensional data
#
##/

dataset = [
    -13.65089255716321, -0.5409562932238607, -88.4726466247223,
    39.30158828358612, 4.066458182574449, 64.64143300482378,
    38.68269424751338, 33.42013676314311, 31.18603331719732,
    -0.2027616409406292, 45.13590038987272, 30.791899783552395,
    61.1727490302448, 18.167220741624856, 88.88077709786394,
    -1.3808002119514704, 50.14991362212521, 55.92029956281276,
    -6.759813255299466, 34.28290084421072
]

k = 2 # number of clusters


# Uses random to select random ints between 0 and length-1. These are the index
# numbers of items in the dataset which will be the initial centroids for our algorithm.
# Reference these items in the dataset, assemble into an array and return the array.
def pick_centroids(xs, num):
    length = len(xs)
    centroids = []
    
    for i in range(num):
     centroids.append(xs[random.randint(0,length-1)])
    
    return centroids


# Calculates the distance between two points which is used in our algorithm to determine
# if points should remain in a cluster or be moved. This function just uses absolute
# distance. For multidimensional spaces, Euclidean or Manhattan distances could be used
# instead.
def distance(a, b):
    dist = abs(a-b)
    return dist


# Calculates the mean for the values in a cluster. This value can be used as the new
# centroid once clusters are formed. If there are no values in the cluster, the centroid
# value defaults to 0. Depending on the application, we could use a min, max, or average
# calculation rather than the mean calculation.
def centroid(xs):
    if len(xs) == 0:
     return 0
    else:
     return sum(xs) / len(xs)


# This function was provided as part of the class. It initializes a list of lists to
# represent the clusters. Then, it iterates through the values in the clusters, estimates
# the distance between each point and the centroids for the clusters. It then places the
# point in the nearest cluster. After iterating through all of the points, it returns
# the clusters.
def cluster(xs, centroids):
    clusters = [[] for c in centroids]

    for x in xs:
        # find the closest cluster to x
        dist, cluster_id = min(
            (distance(x, c), cluster_id) for cluster_id, c in enumerate(centroids)
        )
        # place x in cluster
        clusters[cluster_id].append(x)

    return clusters


# This function was provided as part of the class.
# This function initializes values for the terminating condition and an initial observed
# error. It then runs clusters to calculate distances between points and centroids and
# move data points to the nearest cluster. The new centroid is calculated for
# new_clusters. Lastly, observed_error is calculated by summing the difference between
# the new and old centroids. Centroids is then replaced with new_centroids, and the
# process iterates again until the terminating condition is met.
def iterate_centroids(xs, centroids):
    err = 0.001 # minimum amount of allowed centroid movement
    observed_error = 1 # Initialize: maxiumum amount of centroid movement
    new_clusters = [[] for c in centroids] # Initialize: clusters

    while observed_error > err:
        new_clusters = cluster(xs, centroids) # moves points to nearest clusters
        new_centroids = map(centroid, new_clusters) # calculate the new central point

        observed_error = max(abs(new - old) for new, old in zip(new_centroids, centroids))
        centroids = new_centroids

    return (centroids, new_clusters)


# Main part of program:
# Pick initial centroids
# Iterative to find final centroids
# Print results
initial_centroids = pick_centroids(dataset, k) # selects initial starting centroids.
final_centroids, final_clusters = iterate_centroids(dataset, initial_centroids)

for centroid, cluster in zip(final_centroids, final_clusters):
    print "Centroid: %s" % centroid
    print "Cluster contents: %r" % cluster