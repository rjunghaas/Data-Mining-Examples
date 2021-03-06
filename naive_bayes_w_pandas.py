import pandas as pd

data = [[0, [1,0,0,1]], [1, [0,1,1,0]], [0, [0,0,0,1]], [1, [0,1,0,0]]]

# Function that adds up each attribute within class and calculates conditional probs as list
def make_probs(df, num_samples, num_attr):
    # initialize counts to be 1's (LaPlace Smoothing)
    i = 0
    counts = []
    while i < num_attr:
        counts.append(1)
        i += 1
    
    # First sum up number of instances of each attribute in class
    for samp in df.iterrows():
        s = samp[1]['attributes']
        for index in range(num_attr):
            if len(counts) == num_attr:
                c = counts[index]
                new_c = int(counts[index]) + int(s[index])
                counts[index] = new_c
            else:
                counts.append(s[index])
    
    # Then calculate conditional probability of each attribute
    probs = []
    for each in counts:
        c = counts[each]
        prob = c / float(num_samples + 1)
        probs.append(prob)
    
    return probs

# Function to make a prediction of class of an instance
def predict_class(test, cond_probs_1, cond_probs_0, p_1, p_0):
    # Calculate conditional probability of being class 1
    cond_prob_1 = 1
    i = 0
    while i < len(test):
        this_prob = cond_probs_1[i]
        tp = test[i] * this_prob
        if (tp == 0):
            i += 1
            continue
        else:
            cond_prob_1 *= tp
            i += 1
    
    # Calculate conditional probability of being class 2
    cond_prob_0 = 1
    j = 0
    while j < len(test):
        this_prob = cond_probs_0[j]
        tp = test[j] * this_prob
        if (tp == 0):
            j += 1
            continue
        else:
            cond_prob_0 *= tp
            j += 1
    
    # Argmax posterior possibility of each class
    if (cond_prob_0 * p_0) >= (cond_prob_1 * p_1):
        return 0
    else:
        return 1

# Construct DataFrame and partition into class 0 and 1 Labeled Points
df = pd.DataFrame(data=data, columns = ['class','attributes'])
df0 = df[df['class'] == 0]
df1 = df[df['class'] == 1]

# Get number of samples in each class
num_attr = len(df['attributes'])
num_samples_1 = len(df1)
num_samples_0 = len(df0)

# Calculate conditional probs
cond_probs_1 = make_probs(df1, num_samples_1, num_attr)
cond_probs_0 = make_probs(df0, num_samples_0, num_attr)

# Calculate Class Probabilities
p_1 = float(num_samples_1) / (num_samples_1 + num_samples_0)
p_0 = (1 - p_1)

# Test case of predictor
test = [0,0,1,0]
x = predict_class(test, cond_probs_1, cond_probs_0, p_1, p_0)
print x