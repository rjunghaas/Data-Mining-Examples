import hashlib
import time
import random
from datetime import datetime
import pandas as pd
try:
    import json
except ImportError:
    import simplejson as json

""" convert list of dict objects to DataFrame object. """
event_df = pd.DataFrame(event_data)

""" convert list of dict objects to DataFrame object """
people_df = pd.DataFrame(people_data)

""" Create a list of duplicate columns between event and people DataFrames. """
dropped = []
for people_key in people_df.keys():
	if (people_key in event_df.keys()):
		dropped.append(str(people_key))

""" Drop duplicate columns from people DataFrame and then merge the DataFrames into a master DataFrame."""
people_df2 = people_df.drop(dropped, axis=1)
master = pd.merge(event_df, people_df2, left_on='distinct_id', right_on='$distinct_id')

""" Example 3: Naive Bayes Model for Success Variable.
1. Add column to turn Outcome = yes / no into binary variable.  Add column for people_property and predictor_event.
2. Select training (n = 40 with 50%/50% split) population, plus testing population (n = 70 with 90% yes / 10% no split)
3. Calculate P(success), P(property), P(property | success) from training set
4. Calculate P(success | property) using Bayes's Theorem
5. Display results on test group
"""

""" STEP 1 """
""" Create new column for binary value for success property """
OUTCOME = 'outcome'
PEOPLE_PROP = 'property'
outcome = master[OUTCOME].values
outcome_list = outcome.tolist()
new_col = []
yes_index = []
no_index = []
yes_count = 0
no_count = 0
i = 0
# Add "1" to new column for "Yes", "0" for "No", "2" otherwise
for each in outcome_list:
	if each == "Yes":
		new_col.append(1)
		yes_index.append(i)
		yes_count += 1
	elif each == "No":
		new_col.append(0)
		no_index.append(i)
		no_count += 1
	else:
		new_col.append(2)
		no_count += 1 
	i += 1

master['outcome_bin'] = new_col # add column to DataFrame

""" Create new column for whether user has people_prop """
people_prop = master[PEOPLE_PROP].values
people_prop_list = people_prop.tolist()
people_prop_col = []
# if people_prop is True, add column value = 1; else column value = 0
for each in people_prop_list:
	if each == "True":
		val = 1
	else:
		val = 0
	people_prop_col.append(val)

master['people_prop_col'] = people_prop_col # add column to DataFrame

""" Create new column for binary variable for users who have attempted predictor_event """
PREDICTOR_EVENT = 'event'
pred_event_list = master[master.whatStatus == PREDICTOR_EVENT]['$distinct_id'].values
master['pred_event_col'] = 0

# if user has attempted to do predictor_event, set pred_event_col = 1 in DataFrame
for id in pred_event_list:
	master.loc[(master['$distinct_id'] == id), 'pred_event_col'] = 1

pred_event_count = len(master[master.whatStatus == EVENT])

""" STEP 2 """
""" Identify training and testing data """
YES_TRAIN_SAMPLE = 20
NO_TRAIN_SAMPLE = 20

YES_TEST_SAMPLE = 63
NO_TEST_SAMPLE = 7

yes_len = len(yes_index)
no_len = len(no_index)

y = 0
yes_train_index = []
while y < YES_TRAIN_SAMPLE:
	rand = random.randint(0, yes_len-1)
	if yes_index[rand] not in yes_train_index:
		yes_train_index.append(yes_index[rand])
	y += 1

n = 0
no_train_index = []
while n < NO_TRAIN_SAMPLE:
	rand = random.randint(0, no_len-1)
	if no_index[rand] not in no_train_index:
		no_train_index.append(no_index[rand])
	n += 1

z = 0
yes_test_index = []
while z < YES_TEST_SAMPLE:
	rand = random.randint(0, yes_len-1)
	if yes_index[rand] not in yes_train_index:
		yes_test_index.append(yes_index[rand])
	z += 1

i = 0
no_test_index = []
while i < len(no_index):
	if no_index[i] not in no_train_index:
		no_test_index.append(no_index[i])
	i += 1	
	
""" Copy in training rows to new DataFrame called train_group; replace master below with train_group """	
train_group_index = sorted(yes_train_index + no_train_index)
group_index = []
i = 0
while i < len(master):
	if i not in train_group_index:
		group_index.append(i)
	i += 1
train_group = master.drop(group_index)

""" Copy in testing rows to new DataFrame called test_group """
test_group_index = sorted(yes_test_index + no_test_index)
group_index = []
j = 0
while j < len(master):
	if j not in test_group_index:
		group_index.append(j)
	j+=1
test_group = master.drop(group_index)

""" STEP 3 """
""" Calculate P(outcome) """
total_rows = float(len(master))
p_outcome = yes_count / float(total_rows)

""" Calculate P(not outcome) = 1 - P(outcome) """
p_not_outcome = 1 - p_outcome

""" Calculate P(people_prop) """
people_prop_list = master[master.people_prop_col == 1].people_prop_col.values.tolist()
people_prop_len = len(people_prop_list)
p_people_prop = people_prop_len / float(total_rows)

""" Calculate P(pred_event) """
p_pred_event = pred_event_count / float(total_rows)

""" Calculate P(people_prop | outcome) """
outcome_people_prop_series = master[master.outcome_bin == 1].people_prop_col
p_people_prop_given_outcome = outcome_people_prop_series.count() / float(yes_count)

""" Calculate P(pred_event | outcome) """
outcome_pred_event_series = master[master.outcome_bin == 1].pred_event_col
p_pred_event_given_outcome = outcome_pred_event_series.sum() / float(yes_count)

""" Calculate P(people_prop | not outcome) """
not_outcome_people_prop_series = master[master.outcome_bin != 1].people_prop_col
p_people_prop_given_not_outcome = not_outcome_people_prop_series.sum() / float(no_count)

""" Calculate P(pred_event | not outcome) """
not_outcome_pred_event_series = master[master.outcome_bin != 1].pred_event_col
p_pred_event_given_not_outcome = not_outcome_pred_event_series.sum() / float(no_count)

""" STEP 4 """
""" Calculate P(outcome | people_prop) """
p_outcome_given_people_prop = (p_people_prop_given_outcome * p_outcome) / float(p_people_prop)

""" Calculate P(outcome | pred_event) """
p_outcome_given_pred_event = (p_pred_event_given_outcome * p_outcome) / float(p_pred_event)

""" Calculate P(outcome | people_prop AND pred_event) """
p_outcome_given_people_prop_and_pred_event = (p_people_prop_given_outcome*p_pred_event_given_outcome*p_outcome) / ((p_people_prop_given_outcome*p_pred_event_given_outcome*p_outcome)+(p_people_prop_given_not_outcome*p_pred_event_given_not_outcome*p_not_outcome))

""" STEP 5 """
""" Add column with predicted probabilities """
test_group['test_result'] = 0.0
test_group.loc[(test_group.people_prop_col == 1) & (test_group.pred_event_col == 1),'test_result'] = float(p_outcome_given_people_prop_and_pred_event)
test_group.loc[(test_group.people_prop_col == 1) & (test_group.pred_event_col == 0),'test_result'] = float(p_outcome_given_people_prop)
test_group.loc[(test_group.people_prop_col == 0) & (test_group.pred_event_col == 1),'test_result'] = float(p_outcome_given_pred_event)

""" Create summarized DataFrame with results; measure whether results were correct against actual results of testing group """
final_results = pd.DataFrame(test_group['test_result'], columns=['test_result'])
final_results['outcome_bin'] = test_group['outcome_bin']
final_results['correct'] = 0.0
final_results.loc[(final_results.test_result > 0.01) & (final_results.success_bin == 1),'correct'] = 1
final_results.loc[(final_results.test_result == 0) & (final_results.success_bin == 0), 'correct'] = 1
print "Prediction Results"
right_answers = len(final_results[final_results.correct == 1])
print "Total correct: " + str(right_answers) + " out of " + str(len(final_results))