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

    return selectk_features, selected_cols
    
def split_scalar(indep_X,dep_Y):
        X_train, X_test, y_train, y_test = train_test_split(indep_X, dep_Y, test_size = 0.25, random_state = 0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)    
        return X_train, X_test, y_train, y_test
    
def r2_prediction(regressor,X_test,y_test):
     y_pred = regressor.predict(X_test)
     from sklearn.metrics import r2_score
     r2=r2_score(y_test,y_pred)
     return r2
 
def Linear(X_train,y_train,X_test,y_test):       
        # Fitting K-NN to the Training set
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        r2=r2_prediction(regressor,X_test,y_test)
        return  r2   
    
def svm_linear(X_train,y_train,X_test,y_test):
                
        from sklearn.svm import SVR
        regressor = SVR(kernel = 'linear')
        regressor.fit(X_train, y_train)
        r2=r2_prediction(regressor,X_test,y_test)
        return  r2  
    
def svm_NL(X_train,y_train,X_test,y_test):
                
        from sklearn.svm import SVR
        regressor = SVR(kernel = 'rbf')
        regressor.fit(X_train, y_train)
        r2=r2_prediction(regressor,X_test,y_test)
        return  r2  
     

def Decision(X_train,y_train,X_test,y_test):
        
        # Fitting K-NN to the Training setC
        from sklearn.tree import DecisionTreeRegressor
        regressor = DecisionTreeRegressor(random_state = 0)
        regressor.fit(X_train, y_train)
        r2=r2_prediction(regressor,X_test,y_test)
        return  r2  
     

def random(X_train,y_train,X_test,y_test):       
        # Fitting K-NN to the Training set
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
        regressor.fit(X_train, y_train)
        r2=r2_prediction(regressor,X_test,y_test)
        return  r2 
    
    
def selectk_regression(acclin, accsvml, accsvmnl, accdes, accrf): 
    # Wrap each accuracy variable in a list if they are single numbers, 
    # or pass directly if they are already lists.
    data = {
        'Linear': acclin if isinstance(acclin, list) else [acclin],
        'SVMl': accsvml if isinstance(accsvml, list) else [accsvml],
        'SVMnl': accsvmnl if isinstance(accsvmnl, list) else [accsvmnl],
        'Decision': accdes if isinstance(accdes, list) else [accdes],
        'Random': accrf if isinstance(accrf, list) else [accrf]
    }
    
    # Construct DataFrame directly with row label
    dataframe = pd.DataFrame(data, index=['ChiSquare'])
    return dataframe
    
# dataset1=pd.read_csv("prep.csv",index_col=None)

# df2=dataset1

# df2 = pd.get_dummies(df2, drop_first=True)

# indep_X=df2.drop('classification_yes', 1)
# dep_Y=df2['classification_yes']


# kbest=selectkbest(indep_X,dep_Y,5)      

# acclin=[]
# accsvml=[]
# accsvmnl=[]
# accdes=[]
# accrf=[]

# X_train, X_test, y_train, y_test=split_scalar(kbest,dep_Y)  
# for i in kbest:   
#     r2_lin=Linear(X_train,y_train,X_test)
#     acclin.append(r2_lin)
    
#     r2_sl=svm_linear(X_train,y_train,X_test)    
#     accsvml.append(r2_sl)
    
#     r2_NL=svm_NL(X_train,y_train,X_test)
#     accsvmnl.append(r2_NL)
    
#     r2_d=Decision(X_train,y_train,X_test)
#     accdes.append(r2_d)
    
#     r2_r=random(X_train,y_train,X_test)
#     accrf.append(r2_r)
    
    
# result=selectk_regression(acclin,accsvml,accsvmnl,accdes,accrf)



# result