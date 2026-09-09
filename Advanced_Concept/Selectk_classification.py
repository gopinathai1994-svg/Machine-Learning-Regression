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


def selectkbest(indep_X, dep_Y, n):
    test = SelectKBest(score_func=chi2, k=n)
    fit1 = test.fit(indep_X, dep_Y)
    selectk_features = fit1.transform(indep_X)
    selected_cols = indep_X.columns[fit1.get_support()]

    # kbest (features) முதலில் return செய்ய வேண்டும்
    return selectk_features, selected_cols
    
# def selectkbest(indep_X,dep_Y,n):
#         test = SelectKBest(score_func=chi2, k=n)
#         fit1= test.fit(indep_X,dep_Y)
#         selectk_features = fit1.transform(indep_X)
#         selected_cols = indep_X.columns[fit1.get_support()]
#         return selectk_features,selected_cols
    
def split_scalar(indep_X,dep_Y):
        X_train, X_test, y_train, y_test = train_test_split(indep_X, dep_Y, test_size = 0.25, random_state = 0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)    
        return X_train, X_test, y_train, y_test
    
 
# 1. cm_prediction-ல் y_test சேர்க்கப்பட்டது
def cm_prediction(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )

    cm = confusion_matrix(y_test, y_pred)
    Accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return classifier, Accuracy, report, X_test, y_test, cm


# 2. அனைத்து Algorithm ஃபங்ஷன்களிலும் y_test சேர்க்கப்பட்டது
def logistic(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)
    return cm_prediction(classifier, X_test, y_test)


def svm_linear(X_train, y_train, X_test, y_test):
    from sklearn.svm import SVC

    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(X_train, y_train)
    return cm_prediction(classifier, X_test, y_test)


def svm_NL(X_train, y_train, X_test, y_test):
    from sklearn.svm import SVC

    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(X_train, y_train)
    return cm_prediction(classifier, X_test, y_test)


def Navie(X_train, y_train, X_test, y_test):
    from sklearn.naive_bayes import GaussianNB

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    return cm_prediction(classifier, X_test, y_test)


def knn(X_train, y_train, X_test, y_test):
    from sklearn.neighbors import KNeighborsClassifier

    classifier = KNeighborsClassifier(
        n_neighbors=5, metric='minkowski', p=2
    )
    classifier.fit(X_train, y_train)
    return cm_prediction(classifier, X_test, y_test)


def Decision(X_train, y_train, X_test, y_test):
    from sklearn.tree import DecisionTreeClassifier

    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    return cm_prediction(classifier, X_test, y_test)


def random(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import RandomForestClassifier

    classifier = RandomForestClassifier(
        n_estimators=10, criterion='entropy', random_state=0
    )
    classifier.fit(X_train, y_train)
    return cm_prediction(classifier, X_test, y_test)
    
def selectk_Classification(acclog, accsvml, accsvmnl, accknn, accnav, accdes, accrf):
    data = {
        'Logistic': acclog,
        'SVMl': accsvml,
        'SVMnl': accsvmnl,
        'KNN': accknn,
        'Navie': accnav,
        'Decision': accdes,
        'Random': accrf,
    }

    dataframe = pd.DataFrame(data, index=['ChiSquare'])
    return dataframe
    
# dataset1=pd.read_csv("prep.csv",index_col=None)

# df2=dataset1

# df2 = pd.get_dummies(df2,dtype=int, drop_first=True)

# indep_X=df2.drop('classification_yes', axis=1)
# dep_Y=df2['classification_yes']


# kbest=selectkbest(indep_X,dep_Y,5)       

# acclog=[]
# accsvml=[]
# accsvmnl=[]
# accknn=[]
# accnav=[]
# accdes=[]
# accrf=[]


# X_train, X_test, y_train, y_test=split_scalar(kbest,dep_Y)   
    
        
# classifier,Accuracy,report,X_test,y_test,cm=logistic(X_train,y_train,X_test)
# acclog.append(Accuracy)

# classifier,Accuracy,report,X_test,y_test,cm=svm_linear(X_train,y_train,X_test)  
# accsvml.append(Accuracy)
    
# classifier,Accuracy,report,X_test,y_test,cm=svm_NL(X_train,y_train,X_test)  
# accsvmnl.append(Accuracy)
    
# classifier,Accuracy,report,X_test,y_test,cm=knn(X_train,y_train,X_test)  
# accknn.append(Accuracy)
    
# classifier,Accuracy,report,X_test,y_test,cm=Navie(X_train,y_train,X_test)  
# accnav.append(Accuracy)
    
# classifier,Accuracy,report,X_test,y_test,cm=Decision(X_train,y_train,X_test)  
# accdes.append(Accuracy)
    
# classifier,Accuracy,report,X_test,y_test,cm=random(X_train,y_train,X_test)  
# accrf.append(Accuracy)
    
# result=selectk_Classification(acclog,accsvml,accsvmnl,accknn,accnav,accdes,accrf)

# result