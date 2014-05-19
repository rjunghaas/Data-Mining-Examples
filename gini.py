#!/usr/bin/python
"""Script can be used to calculate the Gini Index of a column in a CSV file.

Classes are strings."""

from __future__ import division
import fileinput
import csv
import collections

(
    CMTE_ID, AMNDT_IND, RPT_TP, TRANSACTION_PGI, IMAGE_NUM, TRANSACTION_TP,
    ENTITY_TP, NAME, CITY, STATE, ZIP_CODE, EMPLOYER, OCCUPATION,
    TRANSACTION_DT, TRANSACTION_AMT, OTHER_ID, CAND_ID, TRAN_ID, FILE_NUM,
    MEMO_CD, MEMO_TEXT, SUB_ID
) = range(22)

CANDIDATES = {
    'P80003338': 'Obama',
    'P80003353': 'Romney',
}

############### Set up variables
# TODO: declare datastructures
# Data structure contains a hash of zip codes as key and number of donors for candidate as
# value.  Total number of donors also held within num_donors property.
class Candidate:
	def __init__(self, name):
		self.name = name
		self.zips = {}
		self.num_donors = 0

# Initializing two Candidate Objects for each candidate
Obama = Candidate("Obama")
Romney = Candidate("Romney")

############### Read through files
for row in csv.reader(fileinput.input(), delimiter='|'):
    candidate_id = row[CAND_ID]
    if candidate_id not in CANDIDATES:
        continue

    candidate_name = CANDIDATES[candidate_id]
    zip_code = row[ZIP_CODE]
    ###
    # TODO: save information to calculate Gini Index
    ##/
    if candidate_name == 'Obama': # takes Obama donations and adds them to hash within Obama
		if zip_code not in Obama.zips:
			Obama.zips[zip_code] = 1
		else:
			Obama.zips[zip_code] += 1
		Obama.num_donors += 1
	
    if candidate_name == 'Romney': # takes Romney donations and adds them to hash within Romney
		if zip_code not in Romney.zips:
			Romney.zips[zip_code] = 1
		else:
			Romney.zips[zip_code] += 1
		Romney.num_donors += 1

# Calculating overall Gini Index.  Total = total number of donors for both candidates.	
total = float(Obama.num_donors + Romney.num_donors)
obama_frac = float(Obama.num_donors/total)
romney_frac = float(Romney.num_donors/total)
gini = 1 - ((obama_frac**2) + (romney_frac**2))

# use defaultdict to create hash of unique Obama zip codes as keys and count of donors from that zip as value
o = collections.defaultdict(int)
for i in Obama.zips:
	o[i] = Obama.zips[i]

# same procedure for Romney using defaultdict.
r = collections.defaultdict(int)
for j in Romney.zips:
	r[j] = Romney.zips[j]

# Create a master_hash with zip code as key and [Obama donors, Romney donors, total donors]
# as value
master_hash = {}
grand_total = Obama.num_donors + Romney.num_donors
for zip in o:
	obama_zip = zip[:5]
	o_donors = o[zip]
	for x in r:
		romney_zip = x[:5]
		r_donors = r[x]
		if obama_zip == romney_zip: # add all zip codes with donors for both candidates
			donor_count = o_donors + r_donors
			master_hash[obama_zip] = [o_donors, r_donors, donor_count]

# add zip codes with only Obama donors
for y in o:
	obama_zip = y[:5]
	o_donors = o[y]
	if master_hash.has_key(obama_zip):
		continue
	else:
		master_hash[obama_zip] = (o_donors, 0, o_donors)

# add zip codes with only Romney donors
for x in r: 
	romney_zip = x[:5]
	r_donors = r[x]
	if master_hash.has_key(romney_zip):
		continue
	else:
		master_hash[romney_zip] = [0, r_donors, r_donors]
# master_hash is now completely constructed.

# loop through master_hash to calculate split_gini
split_gini = 0.0
keys_list = master_hash.keys()
for a in keys_list:
	o_donors = master_hash[a][0]
	r_donors = master_hash[a][1]
	donor_count = master_hash[a][2]
	o_frac = float(o_donors) / float(donor_count)
	r_frac = float(r_donors) / float(donor_count)
	weight = float(donor_count) / float(grand_total)
	zip_gini = 1 - ((o_frac)**2) - ((r_frac)**2)
	zip_contrib = weight * zip_gini
	split_gini += zip_contrib

###
# TODO: calculate the values below:
#gini  # current Gini Index using candidate name as the class
#split_gini  # weighted average of the Gini Indexes using candidate names, split up by zip code
##/

print "Gini Index: %.6f" % gini
print "Gini Index after split: %.6f" % split_gini
