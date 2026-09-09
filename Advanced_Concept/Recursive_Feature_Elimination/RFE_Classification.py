import pandas as pd
from sklearn.model_selection import train_test_split 
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier   
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# def rfeFeature(indep_X,dep_Y,n):
#         rfelist=[]
        
#         log_model = LogisticRegression(solver='lbfgs')
#         RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
#        # NB = GaussianNB()
#         DT= DecisionTreeClassifier(criterion = 'gini', max_features='sqrt',splitter='best',random_state = 0)
#         svc_model = SVC(kernel = 'linear', random_state = 0)
#         #knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
#         rfemodellist=[log_model,svc_model,RF,DT] 
#         for i in   rfemodellist:
#             print(i)
#             log_rfe = RFE(i, n)
#             log_fit = log_rfe.fit(indep_X, dep_Y)
#             log_rfe_feature=log_fit.transform(indep_X)
#             rfelist.append(log_rfe_feature)
#         return rfelist

def rfeFeature(indep_X, dep_Y, n):
    rfelist = []
    
    log_model = LogisticRegression(solver='lbfgs')
    RF = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    DT = DecisionTreeClassifier(criterion='gini', max_features='sqrt', splitter='best', random_state=0)
    svc_model = SVC(kernel='linear', random_state=0)
    
    rfemodellist = [log_model, svc_model, RF, DT]
    
    for i in rfemodellist:
        print(i)
        # Fix: Pass n_features_to_select explicitly as a keyword argument
        log_rfe = RFE(estimator=i, n_features_to_select=n)
        log_fit = log_rfe.fit(indep_X, dep_Y)
        log_rfe_feature = log_fit.transform(indep_X)
        rfelist.append(log_rfe_feature)
        
    return rfelist
    

def split_scalar(indep_X,dep_Y):
        X_train, X_test, y_train, y_test = train_test_split(indep_X, dep_Y, test_size = 0.25, random_state = 0)
        #X_train, X_test, y_train, y_test = train_test_split(indep_X,dep_Y, test_size = 0.25, random_state = 0)
        
        #Feature Scaling
        #from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
def cm_prediction(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
        
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
    
    # y_test is now accessible inside the function
    cm = confusion_matrix(y_test, y_pred)
    Accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return classifier, Accuracy, report, X_test, y_test, cm

def logistic(X_train, y_train, X_test, y_test):        
    from sklearn.linear_model import LogisticRegression
    
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)
    
    # Fixed: Pass y_test into cm_prediction
    classifier, Accuracy, report, X_test, y_test, cm = cm_prediction(classifier, X_test, y_test)
    
    return classifier, Accuracy, report, X_test, y_test, cm   
    
def svm_linear(X_train,y_train,X_test,y_test):
                
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'linear', random_state = 0)
        classifier.fit(X_train, y_train)
        classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test,y_test)
        return  classifier,Accuracy,report,X_test,y_test,cm
    
def svm_NL(X_train,y_train,X_test,y_test):
                
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'rbf', random_state = 0)
        classifier.fit(X_train, y_train)
        classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test,y_test)
        return  classifier,Accuracy,report,X_test,y_test,cm
   
def Navie(X_train,y_train,X_test,y_test):       
        # Fitting K-NN to the Training set
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test,y_test)
        return  classifier,Accuracy,report,X_test,y_test,cm         
    
    
def knn(X_train,y_train,X_test,y_test):
           
        # Fitting K-NN to the Training set
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(X_train, y_train)
        classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test,y_test)
        return  classifier,Accuracy,report,X_test,y_test,cm
def Decision(X_train,y_train,X_test,y_test):
        
        # Fitting K-NN to the Training set
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)
        classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test,y_test)
        return  classifier,Accuracy,report,X_test,y_test,cm      


def random(X_train,y_train,X_test,y_test):
        
        # Fitting K-NN to the Training set
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)
        classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test,y_test)
        return  classifier,Accuracy,report,X_test,y_test,cm
    

def rfe_classification(acclog, accsvml, accsvmnl, accknn, accnav, accdes, accrf):
    # Dictionary mapping column names directly to their accuracy lists
    data = {
        'Logistic': acclog,
        'SVMl': accsvml,
        'SVMnl': accsvmnl,
        'KNN': accknn,
        'Navie': accnav,
        'Decision': accdes,
        'Random': accrf
    }
    
    # Custom RFE Feature Selection models used as row indices
    rfe_models = ['Logistic', 'SVC', 'Random', 'DecisionTree']
    
    # Create DataFrame with RFE models as row indices
    df = pd.DataFrame(data, index=rfe_models)
    
    return df



# dataset1=pd.read_csv("prep.csv",index_col=None)
# df2=dataset1
# df2 = pd.get_dummies(df2, drop_first=True)

# indep_X=df2.drop('classification_yes', 1)
# dep_Y=df2['classification_yes']


# rfelist=rfeFeature(indep_X,dep_Y,3)       

# acclog=[]
# accsvml=[]
# accsvmnl=[]
# accknn=[]
# accnav=[]
# accdes=[]
# accrf=[]

# for i in rfelist:   
#     X_train, X_test, y_train, y_test=split_scalar(i,dep_Y)   
    
        
#     classifier,Accuracy,report,X_test,y_test,cm=logistic(X_train,y_train,X_test)
#     acclog.append(Accuracy)
    
#     classifier,Accuracy,report,X_test,y_test,cm=svm_linear(X_train,y_train,X_test)  
#     accsvml.append(Accuracy)
    
#     classifier,Accuracy,report,X_test,y_test,cm=svm_NL(X_train,y_train,X_test)  
#     accsvmnl.append(Accuracy)
    
#     classifier,Accuracy,report,X_test,y_test,cm=knn(X_train,y_train,X_test)  
#     accknn.append(Accuracy)
    
#     classifier,Accuracy,report,X_test,y_test,cm=Navie(X_train,y_train,X_test)  
#     accnav.append(Accuracy)
    
#     classifier,Accuracy,report,X_test,y_test,cm=Decision(X_train,y_train,X_test)  
#     accdes.append(Accuracy)
    
#     classifier,Accuracy,report,X_test,y_test,cm=random(X_train,y_train,X_test)  
#     accrf.append(Accuracy)
    
# result=rfe_classification(acclog,accsvml,accsvmnl,accknn,accnav,accdes,accrf)

# result