import math
import copy

# Initialize Constants
INPUT = [0, 1, 2]
WEIGHTS = [[],[0,0,0,-3,2,4],[0,0,0, 2,-3,0.5],0.2,0.7,1.5]
EXPECTED_OUTPUT = 0
LEARNING_RATE = 10
START_HIDDEN_LAYER = 3
START_OUTPUT_LAYER = 6

# Terminology:  Each node is a two function step.  The first function is referred to as
# combination function.  It is a sum product of the prior nodes' outputs with weights.
# Next, the transfer function is a sigmoid function defined in function called sigmoid.
# Sum product is fed into sigmoid to yield node's output.

# This program assumes a three layer (input, hidden, output) neural network.

# Function to calculate transfer function.  Input should be the weighted combination function.
def sigmoid(x):
	if x == 0:
		return 1
	else:
		return (1 / (1+math.exp(-x)))

# Forward Propagation of Neural Network - per problem as defined in class slides
# Calculate combination function of the hidden layer.  Output is a list of sum products
hidden_input = [0,0,0,0,0,0] # initializes a list for results of hidden layer combination function.
i=START_HIDDEN_LAYER # iterator for weights
while i<START_OUTPUT_LAYER:
	j=1 # iterator for input
	while j<START_HIDDEN_LAYER:
		hidden_input[i] += INPUT[j]*WEIGHTS[j][i]
		j+=1
	i+=1
      
# Calculate output of hidden layer.  This section applies transfer function to list of results
# from combination function.  Result is a list of outputs from hidden layer.  This will
# make up the inputs of the output layer.
hidden_output = [0,0,0,0,0,0]
index = START_HIDDEN_LAYER
while index < len(hidden_input):
	hidden_output[index] = sigmoid(hidden_input[index])
	index+=1

# Calculate combination function for output layer.  Applies weights for hidden layer nodes
# to hidden layer output list to get sum product for output node.
output_input = 0
m = START_HIDDEN_LAYER # iterator for hidden layer outputs
while m<START_OUTPUT_LAYER:
	output_input += hidden_output[m]*WEIGHTS[m]
	m+=1

#Calculate transfer function for Output layer.  Applies sigmoid to output's combination function.
final_output = sigmoid(output_input)

# Assemble all of outputs in single list.
new_hidden_output = hidden_output[START_HIDDEN_LAYER:]
output = INPUT + new_hidden_output
output.append(final_output)

# Calculation of Error at output node.
# The formula for this is err(6) = output(6) * (1 - output(6)) * (Expected Output - output(6)).
error_6 = final_output * (1-final_output) * (EXPECTED_OUTPUT - final_output)

error = [0,0,0,0,0,0,0] # initialize a list of all errors for each node.
error[START_OUTPUT_LAYER]=error_6

# Calculation of error for Hidden Layer nodes.  Adds these values to error list.
# The formula for each of these errors is: err(j) = output(j) * (1 - output(j)) * sum(error(6)*WEIGHTS[j]).
hidden_i = START_HIDDEN_LAYER
while hidden_i < START_OUTPUT_LAYER:
	error[hidden_i] = hidden_output[hidden_i] * (1 - hidden_output[hidden_i]) * error_6 * WEIGHTS[hidden_i]
	hidden_i+=1

# Calculation of error in input layer.
# The formula for each of these errors is:  err(j) = output(j) * (1 - output(j)) * 
# sum(error(hidden_layer)*weight(hidden_layer)).
input_i = 1
while input_i < START_HIDDEN_LAYER:
	error_const = INPUT[input_i] * (1 - INPUT[input_i])
	error_i = 3
	while error_i < len(error)-1:
		error[input_i] += error_const * error[error_i] * WEIGHTS[input_i][error_i]
		error_i += 1
	input_i += 1

new_weights = copy.copy(WEIGHTS) # initialize a new weights list to which we will add the
								 # adjustments to the weights.

# Calculates the adjustments to the weights for the hidden layer nodes.
# The formula for this is:  new_weight(j) = LEARNING RATE * error(j) * output(j).
# Starts with the weights from input layer which are lists of lists in new_weights.
# Iterates through items 1 and 2 of new_weights and adjusts weights.
new_weights_i = 1
while new_weights_i < START_HIDDEN_LAYER:
	new_weights_j = 0
	while new_weights_j < len(new_weights[new_weights_i]):
		new_weights[new_weights_i][new_weights_j] += LEARNING_RATE * error[new_weights_j] * output[new_weights_i]
		new_weights_j += 1
	new_weights_i += 1

# This block adjusts the weights for hidden layer using same formula as input nodes.
# Since each of these weights is just a single float item, not a list, I have separated
# them into two separate blocks of code.
i = 0
float_array = [0,0,0]
while i < START_HIDDEN_LAYER:
	float_array[i] = WEIGHTS[i+START_HIDDEN_LAYER] + LEARNING_RATE * error[START_OUTPUT_LAYER] * output[i+START_HIDDEN_LAYER]
	i += 1

# Combines weights from the prior two blocks into one master list.
new_weights = new_weights[:START_HIDDEN_LAYER]
for item in float_array:
	new_weights.append(item)
	
# Prints output as required by assignment
print "err_6 = %s" % error[6]
print "err_5 = %s" % error[5]
print "err_4 = %s" % error[4]
print "err_3 = %s" % error[3]
print ""
print "w_56 = %s" % new_weights[5]
print "w_46 = %s" % new_weights[4]
print "w_36 = %s" % new_weights[3]
print ""
print "err_2 = %s" % error[2]
print "err_1 = %s" % error[1]
print "w_25 = %s" % new_weights[2][5]
print "w_24 = %s" % new_weights[2][4]
print "w_23 = %s" % new_weights[2][3]
print "w_15 = %s" % new_weights[1][5]
print "w_14 = %s" % new_weights[1][4]
print "w_13 = %s" % new_weights[1][3]