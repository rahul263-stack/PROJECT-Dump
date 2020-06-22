
# all_in_one is a fuction, created for splitiing of a dataset inot 3 parts and to do the repatative tasks namely, draw  learning curves, ROC curves and model classification analysis(Error Analysis).
# Import basic libraries
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import time


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

#To get probabilty of SGD 
from sklearn.calibration import CalibratedClassifierCV
base_model = SGDClassifier()


from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve
#################################################################################################################################

# Import libraries for performance analysis (Error analysis)
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#################################################################################################################################

### We want to split our dataset into following three subsets.
# 1st: Training dataset ~ 60 % of total dataset
# 2nd: Cross-Validation dataset ~20% of tatal dataset
# 3rd: Testing dataset ~ 20% of total dataset

from sklearn.cross_validation import train_test_split

def all_in_one_test(features,target):
    #Split X and y in  training and testing data by 80:20 ratio.
    X_train1, X_test, y_train1, y_test = train_test_split(features,target, test_size=0.2, random_state=1)

    #Again,Split training1 dataset into training and cross-validation datasets by 80:20 ratio.
    X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=0.25, random_state=1)
    
    print ("Training Dataset :", X_train.shape, y_train.shape)
    print ("Testing Dataset:", X_test.shape, y_test.shape)
    print ("Validation Dataset:", X_val.shape, y_val.shape)


    # Create a list of classifiers
    classifiers = [LogisticRegression() , DecisionTreeClassifier() ,RandomForestClassifier(), SVC(probability=True), GaussianNB(), KNeighborsClassifier(), GradientBoostingClassifier(),CalibratedClassifierCV(base_model) , MLPClassifier()]

    # Create a dictionary of the classifiers 
    classifiers_dict = {'Logistic Classifier': LogisticRegression(), 'Decision_Tree Classifier': DecisionTreeClassifier(), 'Random_Forest Classifier': RandomForestClassifier(),
                           'SVM Classifier': SVC(probability=True), "GaussianNB Classifier":GaussianNB(), "KNN Classifiers": KNeighborsClassifier(),"XGB Classifier": GradientBoostingClassifier(),
                            "SGD Classifier":CalibratedClassifierCV(base_model) , 'MLP Classifier':MLPClassifier()}

    # All Learning Curves in one figure
    from sklearn.model_selection import learning_curve
    fig, axs = plt.subplots(3,3, figsize=(15, 10))
    fig.subplots_adjust(hspace = 0.25, wspace=0.25)
    axs = axs.ravel()

    List = ['Logistic Regression', 'Decision Tree', 'Random Forest' ,'SVM' , 'Gaussian NB' , 'KNN', 'XGB', 'SGD', 'MLP']
    k = 0
    for  i in range(len(classifiers)):
         train_sizes, train_scores, test_scores = learning_curve(classifiers[i], features,target)
         train_scores_mean = np.mean(train_scores, axis=1)
         train_scores_std = np.std(train_scores, axis=1)
         test_scores_mean = np.mean(test_scores, axis=1)
         test_scores_std = np.std(test_scores, axis=1)
         axs[i].fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
         axs[i].fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
         axs[i].plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
         axs[i].plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation score")
         axs[i].legend(loc= 'lower right')
         axs[i].set_ylim([0.0,1.1])
         axs[i].set_title(str(List[k]))
         k = k+1
    plt.show()

    # All  Classification reports + Accuracy reports + Confusion matrices 
    results = pd.DataFrame([[0, 0,0,0, 0,0 ,0]],columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 ','ROC', 'Time'])
    for name, classifier in classifiers_dict.items(): 
        print(name)
        start = time.time()
        clf = classifier.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        end = time.time()
        print(classification_report(y_test, y_pred))
        conf_mat = confusion_matrix(y_test,y_pred)
        print('Confusion matrix:\n',conf_mat)
        labels =['Class0','Class 1']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(conf_mat,cmap = plt.cm.Blues)
        fig.colorbar(cax)
        ax.set_xticklabels(['']+ labels)
        ax.set_yticklabels(['']+ labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
		
        plt.show()
        print ("\n",accuracy_score(y_test, y_pred))
        print("\n Time taken by the algorithm to get trained and for prediction :",end-start)
        print ('\n==========================================================================\n')
        roc=roc_auc_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        t = end-start
        #Model_results = pd.DataFrame(columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
        model_results =  pd.DataFrame([[name, acc,prec,rec, f1,roc, t]],columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 ','ROC','Time'])
        results = results.append(model_results, ignore_index = True)
    print(results.loc[1:,:])  
    print ('\n==========================================================================\n')


    # All in one  Receiver Operating Characteristic (ROC) curve
    
    plt.figure(figsize=(15,10))
    for name, classifier in classifiers_dict.items():
        fit = classifier.fit(X_train, y_train)
        y_pred = classifier.predict_proba(X_test)[:,1]
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        # Roc curve
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
    
        fpr = false_positive_rate
        tpr = true_positive_rate
 
        plt.plot(fpr, tpr,lw=2 ,hold = True,label =name) #'ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))
    
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',hold= True,label='ROC curve (area = %0.2f)' % roc_auc)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve of all the classifiers')
        plt.grid(True)
        plt.legend(loc="lower right")
    plt.show()

#########################################################################################################################
### ROC Curve 

def plot_roc_curve(algorithm,X_train,y_train, X_val,y_val,name):
    plt.figure(figsize=(6,4))
    fit = algorithm.fit(X_train, y_train)
    y_pred = fit.predict_proba(X_val)[:,1]
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val,y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    #stop = timeit.default_timer()
    
    # Roc curve
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val,y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    
    fpr = false_positive_rate
    tpr = true_positive_rate
    plt.plot(fpr, tpr,lw=2 ,hold = True) #'ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--',hold= True,label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name)
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.show()

###############################################################################################################

# Draw learning curve for our classification problem:
# Create a plot_learning_curve function
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.ylim(0,1.1)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

##################################################################################

# Performance analysis (Error analysis)
# Print the classification results
def classification_accuracy (y_val, y_pred):
	print('classification_report:\n',classification_report(y_val, y_pred))
	print ('accuracy_score:\n',accuracy_score(y_val, y_pred))
	conf_mat = confusion_matrix(y_val,y_pred)
	print('Confusion matrix:\n', conf_mat)
	labels = ['Class 0', 'Class 1']
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
	fig.colorbar(cax)
	ax.set_xticklabels([''] + labels)
	ax.set_yticklabels([''] + labels)
	plt.xlabel('Predicted')
	plt.ylabel('Actual')
	plt.show()

################################################################################################################

def split_my_data(features,target):
    #Split X and y in  training and testing data by 80:20 ratio.
    X_train1, X_test, y_train1, y_test = train_test_split(features,target, test_size=0.2, random_state=1)

    #Again,Split training1 dataset into training and cross-validation datasets by 80:20 ratio.
    X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=0.25, random_state=1)
    
    print ("Training Dataset :", X_train.shape, y_train.shape)
    print ("Testing Dataset:", X_test.shape, y_test.shape)
    print ("Validation Dataset:", X_val.shape, y_val.shape)
    return X_train, X_val, y_train, y_val, X_test, y_test

###################################################################################################################################
