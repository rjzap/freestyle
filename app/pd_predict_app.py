from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier as etc
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier as gbc
from operator import itemgetter
import pandas as pd
import numpy as np
import pyodbc
from datetime import datetime

datetime.now().strftime('%Y%m%d_%H%M%S')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# paramerters
#server = "localhost"
#db = "lc_loan_stats"
#user = "zappari"
#assword = "@N3wY0rkC4rp"

#configure connection
#cnxn = pyodbc.connect("DRIVER ={SQL Server};SERVER=" + server + ";DATABASE=" + db + "Trusted_Connection=yes")

#query data
#sql = "SELECT * FROM LoanStats3a_20170620_v12"
#create dataframe
#df = pd.read_sql(sql, cnxn)
#df.head()

### READ IN DATA
df = pd.read_csv('data\LoanStats3a_20170620_v12.csv')

### UNCOMMENT FOLLOWING LINES TO CONFIRM DATA LOAD
#print df.shape ## returns (rows, columns)
#print df.head() ## returns first 5 rows for each column
#print df.groupby('loan_status').size() ## aggregates by values in column 'loan status and returns count'

#PREP VARIABLES TO FACILITATE DATA SAMPLING AND MODELING
ftr_names = df.columns.values
exclude_ftr = ['loan_status', 'member_id']
indvar_ftr = 'loan_status'
trees_ftr_elim = exclude_ftr
gbst_ftr_elim = exclude_ftr

cls_rpt_help = """
    The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative\n.
    The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples\n.
    The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall, where an F-beta score reaches its best value at 1 and worst score at 0.\n
    The F-beta score weights recall more than precision by a factor of beta. beta == 1.0 means recall and precision are equally important.
    """ ##http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html

def depvar_select(x, y):
    return [c for c in x if c not in y]


def extra_trees_prediction(x_train, y_train, x_test, y_test):
    trees = etc()
    trees.fit(x_train, y_train)
    trees_pred = trees.predict(x_test)
    trees_ftr_imp = list(trees.feature_importances_)
    trees_ftr_eval =[]
    for feature, importance in zip(depvar_ftrs, trees_ftr_imp):
        ftr_update = {"name": feature, "score": importance}
        trees_ftr_eval.append(ftr_update)
        if importance == 0.0:
            trees_ftr_elim.append(feature)
    trees_ftr_eval = sorted(trees_ftr_eval, key=itemgetter("score"), reverse=True)
    for i in range(len(trees_ftr_eval)):
        print "{}. {}: %.5f".format(i+1, trees_ftr_eval[i]["name"].title()) % trees_ftr_eval[i]["score"]
    print "\nNumber of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test != trees_pred).sum())
    print classification_report(y_test, trees_pred)
    return trees_ftr_elim

def log_regr_prediction(x_train, y_train, x_test, y_test):
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    logreg_pred = logreg.predict(x_test)
    logreg_accuracy = logreg.score(x_test, y_test)
    print "\nNumber of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test != logreg_pred).sum())
    print logreg_accuracy
    print classification_report(y_test, logreg_pred)

def recursive_ftr_elim(model, x_train, y_train, desired_nmbr_ftrs):
    rfe = RFE(model, desired_nmbr_ftrs)
    rfe = rfe.fit(x_train, y_train)
    rfe_rank = list(rfe.ranking_)
    rfe_ftr_eval = []
    for feature, rank in zip(depvar_ftrs, rfe_rank):
        rfe_ftr_update = {"name": feature, "rank": rank}
        rfe_ftr_eval.append(rfe_ftr_update)
    rfe_ftr_eval = sorted(rfe_ftr_eval, key=itemgetter("rank"))
    for i in range(len(rfe_ftr_eval)):
        print "{}. {}".format(rfe_ftr_eval[i]["rank"], rfe_ftr_eval[i]["name"].title())

def gaussian_nb_prediction(x_train, y_train, x_test, y_test):
    gnb = GaussianNB()
    clf = gnb.fit(x_train, y_train)
    gnb_pred = clf.predict(x_test)
    gnb_accuracy = clf.score(x_test, y_test)
    print "\nNumber of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test != gnb_pred).sum())
    print classification_report(y_test, gnb_pred)
    print cls_rpt_help
    print gnb_accuracy

def grad_bst_prediction(x_train, y_train, x_test, y_test):
    gb = gbc()
    gbst = gb.fit(x_train, y_train)
    gbst_pred = gbst.predict(x_test)
    print "\nNumber of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test != gbst_pred).sum())
    print classification_report(y_test, gbst_pred)
    gbst_ftr_imp = list(gbst.feature_importances_)
    gbst_ftr_eval =[]
    for feature, importance in zip(depvar_ftrs, gbst_ftr_imp):
        ftr_update = {"name": feature, "score": importance}
        gbst_ftr_eval.append(ftr_update)
        if importance == 0.0:
            gbst_ftr_elim.append(feature)
    gbst_ftr_eval = sorted(gbst_ftr_eval, key=itemgetter("score"), reverse=True)
    for i in range(len(gbst_ftr_eval)):
        print "{}. {}: %.5f".format(i+1, gbst_ftr_eval[i]["name"].title()) % gbst_ftr_eval[i]["score"]


### SPLIT LOADED DATA FRAME INTO TRAINING AND TESTING SUBSETS
df_train, df_test = train_test_split(df, test_size=0.3) ##test size parameter determins the size of the test as as a percentage
depvar_ftrs = depvar_select(ftr_names, exclude_ftr)

x_train = df_train[depvar_ftrs]
x_test = df_test[depvar_ftrs]

y_train = df_train[indvar_ftr]
y_test = df_test[indvar_ftr]

###uncomment to check data sample dimensions
#print x_train.shape
#print x_test.shape
#print y_train.shape
#print y_test.shape

##FIT AND EMPLOY ExtraTreesClassifier TO PREDICT DEFAULT AND EVALUATE FEATURE IMPORTANCE

#extra_trees_prediction(x_train, y_train, x_test, y_test)

#print trees_ftr_elim ##check features with 0 importance factor are added to the exlcusion list

###RUN ExtraTreesClassifier again with refined feature list to mitigate overfitting
depvar_ftrs_post_trees_fe = depvar_select(ftr_names, trees_ftr_elim)
x_train2 = df_train[depvar_ftrs_post_trees_fe]
x_test2 = df_test[depvar_ftrs_post_trees_fe]

#extra_trees_prediction(x_train2, y_train, x_test2, y_test)

###RUN LogisticRegression prediction
#log_regr_prediction(x_train, y_train, x_test, y_test)
#recursive_ftr_elim(etc(), x_train, y_train, 10)

###RUN Gaussian Naive Bayes classifier to predict default and evaluate feature importance
#gaussian_nb_prediction(x_train, y_train, x_test, y_test)

###RUN GradientBoostingClassifier to predict defaul and evaluate feature importance

grad_bst_prediction(x_train, y_train, x_test, y_test)

depvar_ftrs_post_gbst_fe = depvar_select(ftr_names, gbst_ftr_elim)
x_train3 = df_train[depvar_ftrs_post_gbst_fe]
x_test3 = df_test[depvar_ftrs_post_gbst_fe]

grad_bst_prediction(x_train3, y_train, x_test3, y_test)

df_out = df_test
gbst_predict = pd.DataFrame(gbst_pred)
df_out = df_out.assign(predicted_status_gbc = gbst_predict.values)
df_out.to_csv("data\output\LoanStats_predict_"+timestamp+".csv")
