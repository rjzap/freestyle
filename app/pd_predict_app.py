from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier as etc
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier as gbc
from operator import itemgetter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import getpass

uid = getpass.getuser()
datetime.now().strftime('%Y%m%d_%H%M%S')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filepath = 'data\LoanStats3a_20170620_v13.csv'

### READ IN DATA
df = pd.read_csv(filepath)

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

welcome = """
    \nWelcome {}, your loan data input file '{}' has been loaded into a data frame with dimensions:
    {}.

    These loan statistics will be evaluated via the following machine learning algorithms for both prediction and feature evaluation purposes:

        + Extra Trees classifier
        + Gradient Boosting classifier
        + Logistic Regression
        + Gaussian Naive Bayes Classifier

    This applicaiton will generate three '.csv' output files:

        + 'LoanStats_predict_' file
        + 'ML_Ensemble_FeatureEval_'
        + 'ML_RFE_FeatureEval_'

    You may review results on screen, or in the output files.\n""".format(uid, filepath, df.shape)


divider = "---------------------------------------------------------------------"
cls_rpt_help = """
    The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative\n.
    The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples\n.
    The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall, where an F-beta score reaches its best value at 1 and worst score at 0.\n
    The F-beta score weights recall more than precision by a factor of beta. beta == 1.0 means recall and precision are equally important.
    """ ##http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html

###Function Groups

###Variable selection
def depvar_select(x, y):
    return [c for c in x if c not in y]

###Confusion matrix creation and visualization
def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.show()

def cm(y_test, prediction_list):
    df_confusion = pd.crosstab(y_test, prediction_list, rownames=['Actual'], colnames=['Predicted'], margins = False)
    print "\n Confustion Matrix, without normalization:\n"
    print df_confusion
    #df_confusion_norm = df_confusion / df_confusion.sum(axis = 1)
    df_confusion_norm = pd.crosstab(y_test, prediction_list, rownames=['Actual'], colnames=['Predicted'], margins = False, normalize = "index" )
    print "\n Confustion Matrix, with normalization:\n"
    print df_confusion_norm
    plot_confusion_matrix(df_confusion)
    plot_confusion_matrix(df_confusion_norm)

###Prediction functions
def extra_trees_prediction(x_train, y_train, x_test, y_test):
    trees = etc()
    trees = trees.fit(x_train, y_train)
    trees_pred = trees.predict(x_test)
    if on_scrn == "Yes":
        print divider
        print "\nExtra Trees Classifier mislabeled %d points out of a total of %d points" % ((y_test != trees_pred).sum(), x_test.shape[0])
        print "\nExtra Trees Classification Report:\n"
        print classification_report(y_test, trees_pred)
        print divider
    return trees_pred

def grad_bst_prediction(x_train, y_train, x_test, y_test):
    gb = gbc()
    gbst = gb.fit(x_train, y_train)
    gbst_pred = gbst.predict(x_test)
    if on_scrn == "Yes":
        print divider
        print "\nGradient Boosting Classifier mislabeled %d points out of a total of %d points" % ((y_test != gbst_pred).sum(), x_test.shape[0])
        print "\nGradient Boosting Classification Report:\n"
        print classification_report(y_test, gbst_pred)
        print divider
    else: pass
    return gbst_pred

def log_regr_prediction(x_train, y_train, x_test, y_test):
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    logreg_pred = logreg.predict(x_test)
    logreg_accuracy = logreg.score(x_test, y_test)
    if on_scrn.title() == "Yes":
        print divider
        print "\nLogistic Regression mislabeled %d points out of a total of %d points" % ((y_test != logreg_pred).sum(), x_test.shape[0])
        print "\nThe Logistic Regression accuracay score is: ", logreg_accuracy
        print "\nLogistic Regression Classification Report:\n"
        print classification_report(y_test, logreg_pred)
        print divider
    else: pass
    return logreg_pred

def gaussian_nb_prediction(x_train, y_train, x_test, y_test):
    gnb = GaussianNB()
    clf = gnb.fit(x_train, y_train)
    gnb_pred = clf.predict(x_test)
    gnb_accuracy = clf.score(x_test, y_test)
    if on_scrn.title() == "Yes":
        print divider
        print "\nGaussian Naive Bayes Classifier mislabeled %d points out of a total %d points" % ((y_test != gnb_pred).sum(), x_test.shape[0])
        print "\nThe Gaussian Naive Bayes Classifier accuracy score is: ", gnb_accuracy
        print "\nGaussian Naive Bayes Classification Report:\n"
        print classification_report(y_test, gnb_pred)
        print divider
    else: pass
    return gnb_pred

###Ensemble model Feature Evaluation functions
def extra_trees_ftr_eval(x_train, y_train, x_test, y_test):
    trees = etc()
    trees_ftr_imp = list(trees.fit(x_train, y_train).feature_importances_)
    trees_ftr_eval =[]
    for feature, importance in zip(depvar_ftrs, trees_ftr_imp):
        ftr_update = {"name": feature, "score": importance}
        trees_ftr_eval.append(ftr_update)
        if importance == 0.0:
            trees_ftr_elim.append(feature)
    trees_ftr_eval = sorted(trees_ftr_eval, key=itemgetter("score"), reverse=True)
    if eval_on_scrn == "Yes":
        print divider
        print "\nExtra Trees Classifier evaluated dependent variable data features as follows:\n"
        for i in range(len(trees_ftr_eval)):
            print "{}. {}: %.5f".format(i+1, trees_ftr_eval[i]["name"].title()) % trees_ftr_eval[i]["score"]
        print divider
    else: pass
    return trees_ftr_eval

def grad_bst_ftr_eval(x_train, y_train, x_test, y_test):
    gb = gbc()
    gbst = gb.fit(x_train, y_train)
    gbst_ftr_imp = list(gbst.feature_importances_)
    gbst_ftr_eval =[]
    for feature, importance in zip(depvar_ftrs, gbst_ftr_imp):
        ftr_update = {"name": feature, "score": importance}
        gbst_ftr_eval.append(ftr_update)
        if importance == 0.0:
            gbst_ftr_elim.append(feature)
    gbst_ftr_eval = sorted(gbst_ftr_eval, key=itemgetter("score"), reverse=True)
    if eval_on_scrn == "Yes":
        print divider
        print "\nGradient Boosting Classifier evaluated dependent variable data features as follows:\n"
        for i in range(len(gbst_ftr_eval)):
            print "{}. {}: %.5f".format(i+1, gbst_ftr_eval[i]["name"].title()) % gbst_ftr_eval[i]["score"]
        print divider
    else: pass
    return gbst_ftr_eval

###Recursive feature elimination function for feature evaluation
def recursive_ftr_elim(model, x_train, y_train, desired_nmbr_ftrs):
    rfe = RFE(model, desired_nmbr_ftrs)
    rfe = rfe.fit(x_train, y_train)
    rfe_rank = list(rfe.ranking_)
    rfe_ftr_eval = []
    for feature, rank in zip(depvar_ftrs, rfe_rank):
        rfe_ftr_update = {"name": feature, "rank": rank}
        rfe_ftr_eval.append(rfe_ftr_update)
    rfe_ftr_eval = sorted(rfe_ftr_eval, key=itemgetter("rank"))
    if rfe_on_scrn == "Yes":
        print divider
        print "RFE results for model: ",model, "\n"
        for i in range(len(rfe_ftr_eval)):
            print "{}. {}".format(rfe_ftr_eval[i]["rank"], rfe_ftr_eval[i]["name"].title())
        else: pass
        print divider
    return rfe_ftr_eval

###Output functions
def output(df_test, x_train, y_train, x_test, y_test):
    df_out = df_test
    etc_predict = pd.DataFrame(extra_trees_prediction(x_train, y_train, x_test, y_test))
    df_out = df_out.assign(predicted_status_etc = etc_predict.values)
    gbst_predict = pd.DataFrame(grad_bst_prediction(x_train, y_train, x_test, y_test))
    df_out = df_out.assign(predicted_status_gbc = gbst_predict.values)
    logreg_predict = pd.DataFrame(log_regr_prediction(x_train, y_train, x_test, y_test))
    df_out = df_out.assign(predicted_status_logreg = logreg_predict.values)
    gnb_predict = pd.DataFrame(gaussian_nb_prediction(x_train, y_train, x_test, y_test))
    df_out = df_out.assign(predicted_status_nb = gnb_predict.values)
    df_out.to_csv("data\output\prediction\LoanStats_predict_"+timestamp+".csv")
    return df_out

def confusion_review(x_train, y_train, x_test, y_test):
    etc_predict = extra_trees_prediction(x_train, y_train, x_test, y_test)
    print divider
    cm(y_test, etc_predict)
    print divider
    gbst_predict = grad_bst_prediction(x_train, y_train, x_test, y_test)
    print divider
    cm(y_test, gbst_predict)
    print divider
    logreg_predict = log_regr_prediction(x_train, y_train, x_test, y_test)
    print divider
    cm(y_test, logreg_predict)
    print divider
    gnb_predict = gaussian_nb_prediction(x_train, y_train, x_test, y_test)
    print divider
    cm(y_test, gnb_predict)
    print divider

def ens_feature_output(x_train, y_train, x_test, y_test):
    etc_ftr_rank = pd.DataFrame(extra_trees_ftr_eval(x_train, y_train, x_test, y_test))
    etc_ftr_rank.columns = ["etc_ftr_nm", "etc_ftr_scr"]
    gbst_ftr_rank = pd.DataFrame(grad_bst_ftr_eval(x_train, y_train, x_test, y_test))
    gbst_ftr_rank.columns = ["gbst_ftr_nm", "gbst_ftr_scr"]
    pd.concat([etc_ftr_rank, gbst_ftr_rank], axis = 1).to_csv("data\output\Feature_eval\ensemble_feature_eval\ML_Ensemble_FeatureEval_"+timestamp+".csv")

def rfe_feature_output(x_train, y_train, x_test, y_test):
    rfe_tree = pd.DataFrame(recursive_ftr_elim(etc(), x_train, y_train, 1))
    rfe_tree.columns = ["etc_ftr_nm", "etc_ftr_rank"]
    rfe_gbst = pd.DataFrame(recursive_ftr_elim(gbc(), x_train, y_train, 1))
    rfe_gbst.columns = ["gbst_ftr_nm", "gbst_ftr_rank"]
    rfe_log = pd.DataFrame(recursive_ftr_elim(LogisticRegression(), x_train, y_train, 1))
    rfe_log.columns = ["log_regr_ftr_nm", "log_regr_ftr_rank"]
    pd.concat([rfe_tree, rfe_log, rfe_gbst], axis = 1).to_csv("data\output\Feature_eval\RFE_feature_eval\ML_RFE_FeatureEval_"+timestamp+".csv")

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

###RUN ExtraTreesClassifier again with refined feature list to mitigate overfitting
depvar_ftrs_post_trees_fe = depvar_select(ftr_names, trees_ftr_elim)
x_train2 = df_train[depvar_ftrs_post_trees_fe]
x_test2 = df_test[depvar_ftrs_post_trees_fe]

depvar_ftrs_post_gbst_fe = depvar_select(ftr_names, gbst_ftr_elim)
x_train3 = df_train[depvar_ftrs_post_gbst_fe]
x_test3 = df_test[depvar_ftrs_post_gbst_fe]

print welcome

on_scrn = raw_input("\nWould you like to review prediction results on screen? Type 'Yes' or 'No'\n").title()

output(df_test, x_train, y_train, x_test, y_test)

if on_scrn == "Yes":
    c_rpt_hlp = raw_input("\nDo you require an explanation of the Classification Report? Type 'Yes' or 'No'\n").title()
    if c_rpt_hlp == "Yes":
        print cls_rpt_help
    else: pass
else: pass

matrix_on_scrn = raw_input("\nWould you like to review confusion matrices for predictions on screen? Type 'Yes' or 'No'\n").title()

if matrix_on_scrn == "Yes":
    confusion_review(x_train, y_train, x_test, y_test)
else: pass

eval_on_scrn = raw_input("\nWould you like to review feature evaluation results on screen? Type 'Yes' or 'No'\n").title()

#ens_feature_output(x_train, y_train, x_test, y_test)

rfe_on_scrn = raw_input("\nWould you like to review feature elimination results on screen? Type 'Yes' or 'No'\n").title()

#rfe_feature_output(x_train, y_train, x_test, y_test)
