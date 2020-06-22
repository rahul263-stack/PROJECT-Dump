
# all_in_one is a fuction, created for splitiing of a dataset inot 3 parts and to do the repatative tasks namely, draw  learning curves, ROC curves and model classification analysis(Error Analysis).
# Import basic libraries
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import time
#ignore warning messages 
import warnings
warnings.filterwarnings('ignore') 


#Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

#################################################################################################################################

# Import libraries for performance analysis (Error analysis)
from sklearn.metrics import roc_auc_score , precision_score, recall_score, f1_score, classification_report,accuracy_score,confusion_matrix,roc_curve, auc

#################################################################################################################################


from sklearn.model_selection import train_test_split

def all_in_one(features,target):

        ### We want to split our dataset into following three subsets.
        # 1st: Training dataset ~ 60 % of total dataset
        # 2nd: Cross-Validation dataset ~20% of tatal dataset
        # 3rd: Testing dataset ~ 20% of total dataset
        #Split X and y in  training and testing data by 80:20 ratio.
    X_train1, X_test, y_train1, y_test = train_test_split(features,target, test_size=0.2, random_state=1)

    #Again,Split training1 dataset into training and cross-validation datasets by 80:20 ratio.
    X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=0.25, random_state=1)
    
    print ("Training Dataset :", X_train.shape, y_train.shape)
    print ("Testing Dataset:", X_test.shape, y_test.shape)
    print ("Validation Dataset:", X_val.shape, y_val.shape)


    # Create a list of classifiers
    #classifiers = [LogisticRegression(class_weight='balanced') , DecisionTreeClassifier(class_weight='balanced') ,RandomForestClassifier(class_weight='balanced'), SVC(probability=True,gamma='scale'), GaussianNB(), KNeighborsClassifier(), 
     #             GradientBoostingClassifier(n_estimators=10, learning_rate=1.0,
      #             max_depth=1, random_state=42,loss = 'deviance')]

    # Create a dictionary of the classifiers 
    classifiers_dict = {'Logistic Classifier': LogisticRegression(class_weight='balanced'), 
                       'Decision_Tree Classifier': DecisionTreeClassifier(class_weight='balanced'), 
                        'Random_Forest Classifier': RandomForestClassifier(class_weight='balanced'),
                        'SVM Classifier': SVC(probability=True,gamma='scale'), 
                        "GaussianNB Classifier":GaussianNB(), 
                        "KNN Classifiers": KNeighborsClassifier(),
                        "GB Classifier": GradientBoostingClassifier(loss = 'deviance'),
                         "XGB Classifier" : XGBClassifier(scale_pos_weight = 2)}

    # All Learning Curves in one figure
    from sklearn.model_selection import learning_curve
    fig, axs = plt.subplots(3,3, figsize=(15, 10))
    fig.subplots_adjust(hspace = 0.25, wspace=0.25)
    axs = axs.ravel()

    List = ['Logistic Regression', 'Decision Tree', 'Random Forest' ,'SVM' , 'Gaussian NB' , 'KNN', 'GB','XGB']
    k = 0
    for name, classifier in classifiers_dict.items(): 
         train_sizes, train_scores, test_scores = learning_curve(classifier, features,target)
         train_scores_mean = np.mean(train_scores, axis=1)
         train_scores_std = np.std(train_scores, axis=1)
         test_scores_mean = np.mean(test_scores, axis=1)
         test_scores_std = np.std(test_scores, axis=1)
         axs[k].fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
         axs[k].fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
         axs[k].plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
         axs[k].plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation score")
         axs[k].legend(loc= 'lower right')
         axs[k].set_ylim([0.5,1.1])
         axs[k].set_title(name)
         k = k+1
    plt.show()

    # All  Classification reports + Accuracy reports + Confusion matrices 
    results = pd.DataFrame([[0, 0,0,0, 0,0 ,0]],columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 ','ROC', 'Time'])
    for name, classifier in classifiers_dict.items(): 
        #print(name)
        start = time.time()
        clf = classifier.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        end = time.time()
        roc=roc_auc_score(y_val, y_pred)
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
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
        y_pred = classifier.predict_proba(X_val)[:,1]
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val,y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        # Roc curve
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val,y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
    
        fpr = false_positive_rate
        tpr = true_positive_rate
 
        plt.plot(fpr, tpr,lw=2 ,label =name) #'ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))
    
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',label='ROC curve (area = %0.2f)' % roc_auc)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve of all the classifiers')
        plt.grid(True)
        plt.legend(loc="lower right")
    plt.show()



#################################################################################################################################

from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
def Optimization (features,target):
    #Split X and y in  training and testing data by 70:30 ratio.
    X_train, X_test, y_train, y_test = train_test_split(features,target, test_size=0.25)


    # Create a list of classifiers
    classifiers = {'Logistic Regression':LogisticRegression(class_weight='balanced'), 
                'Decision Tree':DecisionTreeClassifier(class_weight='balanced'),
                'Random Forest':RandomForestClassifier(class_weight='balanced'), 
                'SVM':SVC(),  
                'Gaussian NB':GaussianNB(), 
                'K-Nearest Neighbor':KNeighborsClassifier(), 
                'Gradient Boosting':GradientBoostingClassifier(loss = 'deviance'),
                'Exteme Gradient Boosting':XGBClassifier(class_weight='balanced',loss = 'deviance')}

    kfold = KFold(n_splits = 5, random_state = 123)

    # Parameters grids for that model
    models_and_parameters = {'Logistic Regression': (LogisticRegression(),{'C': [1,2,3,4,5],'penalty' : ['l1','l2'],
                            'class_weight' : ['balanced',None]}),
                          
     
                            'Decision Tree':(DecisionTreeClassifier(class_weight='balanced'),{'criterion':['gini','entropy'], 
                            'min_samples_split': [2,3,4],'min_samples_leaf': [12,14,16,18,20,22],'max_depth':[2,3,4,5]}),
                         
                            'Random Forest': (RandomForestClassifier(class_weight='balanced'), {'criterion':['gini','entropy'], 
                            'min_samples_split': [8,10,12,14],'min_samples_leaf': [12,14,16,18,20,22], 'max_depth':[2,3,4,5],
                            'n_estimators':[300,400,450,500,550,600]}),
                        
                         
                            'Support Vector Machine':(SVC(probability=True),{'C' : [0.001, 0.005],'gamma' :['scale','auto']}),

                            'K-Nearest Neighbor':(KNeighborsClassifier(),{'n_neighbors':[15,20,25],'algorithm':['auto', 'ball_tree', 'kd_tree'],
                             'leaf_size':[5,10,15,20]}),
                         
                                        
                            'Gradient Boosting':(GradientBoostingClassifier(),{'loss': ['deviance', 'exponential'],
                                            'learning_rate': [.03, .1, .3, 1, 3],'n_estimators':[150,200,300],'max_depth':[2,3,4,5]}),
                       
                        
                        
                        
                            'Extreme Gradient Boosting':(XGBClassifier(),{'learning_rate': [.03, .1, .3, 1, 3], 
                                              'n_estimators':[50,70,100,150,200,250],'max_depth':[2,3,4,5],'scale_pos_weight' :[1,1.5,2,2.5,3,4,5,6]})

                        
                            }


    Best_parameters = pd.DataFrame([[0,0]],columns=['Model','Best_Parameters'])
    results = pd.DataFrame([[0, 0,0,0, 0,0 ]],columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 ','ROC' ])
    for name, (model, params) in models_and_parameters.items():
        clf = RandomizedSearchCV(estimator = model, param_distributions = params, cv = kfold, n_jobs = -1, verbose = 2, scoring="balanced_accuracy").fit(X_train,y_train)
        print(name,":")
        print(clf.best_params_)
        y_pred = clf.predict(X_test)
        roc = roc_auc_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)
        Bestparameters = pd.DataFrame([[name, clf.best_params_]],columns=['Model','Best_Parameters'])
        Best_parameters = Best_parameters.append(Bestparameters, ignore_index = True)
        model_results =  pd.DataFrame([[name, acc,prec,rec, f1,roc]],columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 ','ROC'])
        results = results.append(model_results, ignore_index = True)
    #plt.show()        
    print(results.loc[1:,:]) 
    print("==============================================================================") 





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
    plt.plot(fpr, tpr,lw=2 ) #'ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--',label='ROC curve (area = %0.2f)' % roc_auc)
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
	'''labels = ['Class 0', 'Class 1']
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
	fig.colorbar(cax)
	ax.set_xticklabels([''] + labels)
	ax.set_yticklabels([''] + labels)
	plt.xlabel('Predicted')
	plt.ylabel('Actual')
	plt.show()'''

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
