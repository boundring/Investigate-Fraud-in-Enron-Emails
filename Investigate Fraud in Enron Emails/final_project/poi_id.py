#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

'''
Project: Investigate Fraud in Enron Emails

Richard Smith
March 2021

poi.id.py

See readme.txt for description of Python version and package environment used,
  as well as rationale for the decision.

Contents: (use "find" to navigate)
Pre-processing
  Task 1: Select what features you'll use
    1-A - features_list
  Task 2: Remove outliers 
    2-A - Removing 'TOTAL', 'THE TRAVEL AGENCY IN THE PARK'
  Task 2.5: Data cleaning, checking, cleaning
    2-B - Data cleaning
      2-B1 - Removing feature 'email_address'
    2-C - Data checking
      2-C1 - Cleaning "shifted" entries
      2-C2 - Verifying fixed entries
  Task 3: Create new feature(s)
    3-A - feature creation
Classifier testing
  Task 4: Try a variety of classifiers
    4-A - Function definition for classifier testing, evaluation, validation
    4-B - Iteration over a list of classifiers
    4-C - Checking feature importances
      4-C1 - Extracting features and labels from dataset for local testing
      4-C2 - by DecisionTree feature_importances_ ('Gini' impurity method)
      4-C3 - by SelectKBest scores_ ('ANOVA' f-values)
  Task 5: Tune your classifier to achieve better than .3 precision and recall
    5-A - Using GridSearchCV with SelectKBest, DecisionTreeClassifier
    5-B - Testing tuned parameters with 1000-fold cross validation
    5-C - Manual tuning and testing informed by previous results
Output production
  Task 6: Dump your classifier, dataset, and features list
    6-A - Dumping "my_classifier.pkl", "my_dataset.pkl", "my_feature_list.pkl"
            via tester.dump_classifier_and_data

'''
# console output title splash
print("                        __          __      __")
print("                       /_/         /_/     / /")
print("      ______  ______  __          __  ____/ /     ______  __  _ ")
print("     / __  / / __  / / /         / / / __  /     / __  / / / / /")
print("    / /_/ / / /_/ / / / ______  / / / /_/ / __  / /_/ / / (_/ /")
print("   / ____/ /_____/ /_/ /_____/ /_/ /_____/ /_/ / ____/  L__  /")
print("  / /                                         / /       __/ /")
print(" /_/                                         /_/       /___/\n")
print("Edited to meet project requirements for:")
print(" 'Investigate Fraud in Enron Emails'\n\n")

###############################################################################
###   Task 1: Select what features you'll use.
###############################################################################
# 1-A - features_list

# note: this list is only used by featureFormat(), targetFeatureSplit() in
#         this script for testing feature importances and parameter tuning.
# list for features used by final classifier and Pickle dump will be defined
#   according to classifier performance and feature importances
features_list = ['poi',
                 'salary',
                 'bonus',
                 'long_term_incentive',
                 'expenses',
                 'director_fees',
                 'other',
                 'loan_advances',
                 'deferred_income',
                 'deferral_payments',
                 'total_payments',
                 'restricted_stock_deferred',
                 'exercised_stock_options',
                 'restricted_stock',
                 'total_stock_value',
                 'from_messages',
                 'to_messages',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi',
                 'shared_receipt_with_poi',
                 'from_poi_to_messages_ratio',
                 'to_poi_from_messages_ratio',
                 'shared_receipt_to_messages_ratio']

### Load the dictionary containing the dataset
print("Loading data... ", end='')
with open("final_project_dataset.pkl", "rb") as data_file:
  data_dict = pickle.load(data_file)
print("done.\n")

print(" ---------------------------------------------------")
print("--- Data cleaning, checking, and feature creation ---")
print(" ---------------------------------------------------\n")

###############################################################################
### Task 2: Remove outliers 
###############################################################################
# 2-A - Removing 'TOTAL', 'THE TRAVEL AGENCY IN THE PARK'

print("Removing bad rows... ", end='')
# removing aggregate row
data_dict.pop('TOTAL', 0)
# removing non-person row
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
print("done.")

###############################################################################
### Task 2.5: Data cleaning, checking, cleaning
###############################################################################
# 2-B - Data cleaning

# 2-B1 - Removing email addresses prior to data check (unused feature)
print("Removing email addresses... ", end='')
for k in data_dict.keys():
  if 'email_address' in data_dict[k].keys():
    data_dict[k].pop('email_address')
print("done.")

# 2-C - Data Checking

print("Checking data for issues related to 'total_payments'... ", end='')
payment_financial_features = ['salary',
                              'bonus',
                              'long_term_incentive',
                              'expenses',
                              'director_fees',
                              'other',
                              'loan_advances',
                              'deferred_income',
                              'deferral_payments']
problem_entries = {}
# Iterate over each row, check sum of above features against total_payments,
#   rows with mismatch added to problem_entries
for k in data_dict.keys():
  total_payments_check = 0
  for d in data_dict[k]:
    if d in payment_financial_features and data_dict[k][d] != 'NaN':
      total_payments_check += data_dict[k][d]
  if data_dict[k]['total_payments'] != 'NaN' and \
                        total_payments_check != data_dict[k]['total_payments']:
    problem_entries[k] = data_dict[k]
from pprint import pprint as pp
if len(problem_entries):
  print("found!")
  print("  Rows with issues related to 'total_payments' found:")
  pp(problem_entries)
else:
  print("none.")
print('')

# 2-C1 - Cleaning "shifted" entries

# Due to problems found with two entries, 'BELFER ROBERT' and
#   'BHATNAGAR SANJAY', manually referencing /tools/enron61702insiderpay.pdf to
#   check data by eye.

# For 'BELFER ROBERT', lines marked with '#' were affected by an apparent shift
#   in values, corrected here with values from reference.
# Email data left as-is.
belfer_corrected = {'bonus': 'NaN',
                    'deferral_payments': 0,                   #
                    'deferred_income': -102500,               #
                    'director_fees': 102500,                  #
                    'exercised_stock_options': 0,             #
                    'expenses': 3285,                         #
                    'from_messages': 'NaN',
                    'from_poi_to_this_person': 'NaN',
                    'from_this_person_to_poi': 'NaN',
                    'loan_advances': 'NaN',
                    'long_term_incentive': 'NaN',
                    'other': 'NaN',
                    'poi': False,
                    'restricted_stock': 44093,                #
                    'restricted_stock_deferred': -44093,      #
                    'salary': 'NaN',
                    'shared_receipt_with_poi': 'NaN',
                    'to_messages': 'NaN',
                    'total_payments': 3285,                   #
                    'total_stock_value': 0}                   #

# Likewise, for 'BHATNAGAR SANJAY', lines marked with '#' were affected by an
#   apparent shift in data, corrected here with values from reference.
# Email data left as-is.
bhatnagar_corrected = {'bonus': 'NaN',
                       'deferral_payments': 'NaN',
                       'deferred_income': 'NaN',
                       'director_fees': 0,                    #
                       'exercised_stock_options': 15456290,   #
                       'expenses': 137864,                    #
                       'from_messages': 29,
                       'from_poi_to_this_person': 0,
                       'from_this_person_to_poi': 1,
                       'loan_advances': 'NaN',
                       'long_term_incentive': 'NaN',
                       'other': 0,                            #
                       'poi': False,
                       'restricted_stock': 2604490,           #
                       'restricted_stock_deferred': -2604490, #
                       'salary': 'NaN',
                       'shared_receipt_with_poi': 463,
                       'to_messages': 523,
                       'total_payments': 137864,              #
                       'total_stock_value': 15456290}         #

# Assigning corrected rows to dataset
print('Updating data with corrections... ', end='')
data_dict['BELFER ROBERT'] = belfer_corrected
data_dict['BHATNAGAR SANJAY'] = bhatnagar_corrected
print("done.")

# 2-C2 - Verifying fixed entries
print("Re-checking data for issues related to 'total_payments'... ", end='')
problem_entries = {}
for k in data_dict.keys():
  total_payments_check = 0
  for d in data_dict[k]:
    if d in payment_financial_features and data_dict[k][d] != 'NaN':
      total_payments_check += data_dict[k][d]
  if data_dict[k]['total_payments'] != 'NaN' and \
                        total_payments_check != data_dict[k]['total_payments']:
    problem_entries[k] = data_dict[k]
if len(problem_entries):
  print("found!")
  print("  Rows with issues related to 'total_payments' found:")
  pp(problem_entries)
else:
  print("none.")

###############################################################################
### Task 3: Create new feature(s) 
###############################################################################
# 3-A - Feature Creation

# Presence of 'NaN' for any component features results in 'NaN' values for
#   created features which require them.
print("Creating features 'to_poi_from_messages_ratio'\n\
                         'from_poi_to_messages_ratio'\n\
                         'shared_receipt_to_messages_ratio'...", end='')
for k in data_dict.keys():
  from_messages = True if \
    (data_dict[k]['from_messages'] != 'NaN') else False
  to_messages = True if \
    (data_dict[k]['to_messages'] != 'NaN') else False
  to_poi = True if \
    (data_dict[k]['from_this_person_to_poi'] != 'NaN') else  False
  from_poi = True if \
    (data_dict[k]['from_poi_to_this_person'] != 'NaN') else False
  shared_receipt = True if \
    (data_dict[k]['shared_receipt_with_poi'] != 'NaN') else False

  # ratio of emails sent to PoIs to emails sent generally:
  # to_poi_from_messages_ratio = from_this_person_to_poi / from_messages
  if to_poi and from_messages:
    data_dict[k]['to_poi_from_messages_ratio'] = \
       data_dict[k]['from_this_person_to_poi'] / data_dict[k]['from_messages']
  else:
    data_dict[k]['to_poi_from_messages_ratio'] = 'NaN'

  # ratio of emails received from PoIs to emails received generally:
  # from_poi_to_messages_ratio = from_poi_to_this_person / to_messages
  if from_poi and to_messages:
    data_dict[k]['from_poi_to_messages_ratio'] = \
          data_dict[k]['from_poi_to_this_person'] / data_dict[k]['to_messages']
  else:
    data_dict[k]['from_poi_to_messages_ratio'] = 'NaN'
  
  # ratio of emails having shared recipt with PoIs to emails received generally:
  # shared_receipt_to_messages_ratio = shared_receipt_with_poi / to_messages
  if shared_receipt and to_messages:
    data_dict[k]['shared_receipt_to_messages_ratio'] = \
       data_dict[k]['shared_receipt_with_poi'] / data_dict[k]['to_messages']
  else:
    data_dict[k]['shared_receipt_to_messages_ratio'] = 'NaN'
print(" done.\n")

###############################################################################
### Task 4: Try a variety of classifiers 
###############################################################################
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.tree              import DecisionTreeClassifier
from sklearn.naive_bayes       import GaussianNB
from sklearn.ensemble          import AdaBoostClassifier
from sklearn.model_selection   import StratifiedShuffleSplit

# 4-A - Function definition for classifier testing, validation, evaluation
def classifier_test(clf, dataset, feature_list, folds = 1000):
  '''
  Based on code used in tester.py, with
equivalent functionality, this function
evaluates classifier performance through
cross-validation via
StratifiedShuffleSplit(), default 1000
splits for training and testing sets.
  Written primarily for personal
comprehension of the testing method used
in grading results, and to apply the
same metrics used in grading to
validation and evaluation of classifiers.

parameters:

clf:
  sklearn classifier, must support *.fit,
    *.predict
  
dataset:
  object compatible with Python dict,
    must have key entries containing
    features and values compatible with
    feature_list.

feature_list:
  Python list, must contain strings
    matching features present in dict
    passed to 'dataset'.
  
folds:
  integer, default 1000, controls splits
    applied for cross validation via
    StratifiedShuffleSplit

output:
  Displays predictions made and
    performance results:
    Accuracy, Precision, Recall, F1, F2
  '''
  data = featureFormat(dataset, feature_list, sort_keys = True)
  labels, features = targetFeatureSplit(data)
  cv = StratifiedShuffleSplit(n_splits=folds, random_state = 42)
  true_neg  = 0
  false_neg = 0
  true_pos  = 0
  false_pos = 0
  for train_idx, test_idx in cv.split(features, labels):
    features_train = []
    labels_train   = []
    features_test  = []
    labels_test    = []
    for ii in train_idx:
      features_train.append(features[ii])
      labels_train.append(labels[ii])
    for jj in test_idx:
      features_test.append(features[jj])
      labels_test.append(labels[jj])

    # fit the classifier using training set, and test on test set
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    for prediction, truth in zip(predictions, labels_test):
      if prediction == 0 and truth == 0:
        true_neg += 1
      elif prediction == 0 and truth == 1:
        false_neg += 1
      elif prediction == 1 and truth == 0:
        false_pos += 1
      elif prediction == 1 and truth == 1:
        true_pos += 1
      else:
        print("Warning: Found a predicted label not == 0 or 1.")
        print("All predictions should take value 0 or 1.")
        print("Evaluating performance for processed predictions:")
        break
  try:
    total_pred = true_neg + false_neg + false_pos + true_pos
    accuracy = 1.0 * (true_pos + true_neg) / total_pred
    precision = 1.0 * true_pos / (true_pos + false_pos)
    recall = 1.0 * true_pos / (true_pos + false_neg)
    f1 = 2.0 * true_pos / (2 * true_pos + false_pos + false_neg)
    f2 = (1 + 2.0 * 2.0) * precision * recall / (4 * precision + recall)
    print("Testing", clf, "locally...")
    print("  Predictions: %d" % total_pred)
    print("  Accuracy: %.5f\n  Precision: %.5f  Recall: %.5f" % \
          (accuracy, precision, recall))
    print("  F1: %.5f  F2: %.5f" % (f1, f2), "\n")
  except:
    print("Performance calculations failed.")
    print("Precision or recall may be undefined (no true positives).")
    print("Or else you've forgotten 'poi' in param feature_list")

print(" ------------------------")
print("--- Classifier testing ---")
print(" ------------------------\n")

# 4-B - Iteration over a list of classifiers
# (see references.txt for code example source)
classifiers = [KNeighborsClassifier(),
               DecisionTreeClassifier(),
               GaussianNB()]

print("Trying several classifiers with default settings for comparison...\n")
for classifier in classifiers:
  classifier_test(classifier, data_dict, features_list)

###############################################################################
### 4-C - Checking feature importances
###############################################################################

print(" ---------------------------------")
print("--- Checking feature importance ---")
print(" ---------------------------------\n")

# 4-C1 - Extracting features and labels from dataset for local testing
print("Extracting features and labels... ", end='')
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
print("done.\n")

# 4-C3 - by mutual_info_classif ('mutual information')

from sklearn.feature_selection import mutual_info_classif, SelectKBest
print("\nFeature importance by mutual_info_classif:")
print(" (\"mutual information\" with regard to target, 'poi')")

# sorting feature names by magnitude of mutual information with 'poi'
# (see references.txt for code example used with zip() sorting of two lists)
mutual_info = sorted(zip(list(mutual_info_classif(features, labels)),
                         features_list[1:]), reverse = True)
for i in range(len(mutual_info)):
  print(" ", i+1, "- '%s'" % mutual_info[i][1],
        "\n        %.5f"   % mutual_info[i][0])

print('')

###############################################################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall
###############################################################################
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline        import Pipeline

print(" --------------------------------------")
print("--- Optimizing classifier parameters ---")
print(" --------------------------------------\n")

# 5-A - Using GridSearchCV with SelectKBest, DecisionTreeClassifier

# Using mutual information as feature selection metric
selector = SelectKBest(mutual_info_classif)
# Using information gain as splitting criterion
classifier = DecisionTreeClassifier(criterion = 'entropy')

tune_pipe = Pipeline(steps=[('skb', selector),
                            ('clf', classifier)])

# Optimizing number of features and minimum number of samples for splitting
grid_params = {'skb__k' : (3, 4, 5, 6, 7, 8, 9),
                'clf__min_samples_split' : (3, 4, 5, 6, 7, 8, 9)}

print("Trying GridSearchCV with")
pp(tune_pipe)
print("over parameters:")
pp(grid_params)

# Optimizing for maximized F1 in order to maximize precison and recall
grid = GridSearchCV(tune_pipe, grid_params, scoring = 'f1', cv = 10,
                      n_jobs = -1)
grid.fit(features, labels)

print("\nResulting 'best' parameters for maximizing 'f1':")
pp(grid.best_params_)

# sorting features by paired information gain scores
grid_ftrs = sorted(zip(list(grid.best_estimator_.named_steps['skb'].scores_),
                             features_list[1:]), reverse = True)
# creating featuer list to pass to k-fold testing function
best_features = ['poi']
print("\nFeatures used:")
for i in range(grid.best_params_['skb__k']):
  best_features.append(grid_ftrs[i][1])
  # displaying features for inspection of GridSearchCV's varying results
  print(" ", i+1, "- '%s'" % grid_ftrs[i][1],
        "\n        %.5f"   % grid_ftrs[i][0])
print('')

# 5-B - Testing tuned parameters with 1000-fold cross validation
classifier_test(grid.best_estimator_.named_steps['clf'],data_dict,
                best_features)

###############################################################################
### Final Algorithm
###############################################################################
print(" -----------------------------------------------")
print("--- Testing classifiers with tuned parameters ---")
print(" -----------------------------------------------\n")
# 5-C - Manual tuning and testing informed by previous results
manual_features = ['poi',
                   'expenses',
                   'bonus',
                   'other',
                   'to_poi_from_messages_ratio',
                   'shared_receipt_with_poi']
clf = DecisionTreeClassifier(criterion = 'entropy',
                             min_samples_split = 5)

print("Trying DecisionTreeClassifier with parameter settings and feature")
print("  selection based on 'best' of varying results from optimization...")
print("  (features *reliably* top-ranked by 'mutual information' with 'poi')")
print("Features used:")
pp(manual_features[1:])
classifier_test(clf, data_dict, manual_features)

print("--------------------------------------------------------------------\n")

###############################################################################
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
###############################################################################
# 6-A - Dumping "my_classifier.pkl", "my_dataset.pkl", "my_feature_list.pkl"
import tester
print("Testing final classifier via tester.py...")
tester.dump_classifier_and_data(clf, data_dict, manual_features)
tester.main()
