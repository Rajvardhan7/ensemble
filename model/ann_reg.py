# ANN FOR REGRESSION

# Part 0 - DATA PREPROCESSING

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Importing the dataset
dataset = pd.read_excel('../dataset/utube.xlsx')
dataset.drop(dataset[(dataset['Subscribers'] > 1754000.00)&(dataset['Likes'] > 50300)&(dataset['Dislikes']>3030)].index, inplace = True)
X = dataset.iloc[:, [0,1,2]].values
y = dataset.iloc[:, [-1]].values

# Chechking Correlation
corr = dataset.corr()
sns.heatmap(corr, vmax=0.9, square=True)

#Splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
Y_train = sc.fit_transform(Y_train)
Y_test = sc.fit_transform(Y_test)

# PART 1 - RIGHT ANN FITTING ON MODEL'S OUTPUT (ENSEMBLE STACKING)

# importing models
from mlxtend.regressor import StackingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb

# creating models from imported classes
lasso = Lasso(alpha = 0.1, random_state = 1)
el_reg = ElasticNet(alpha=0.0005)
krr =  KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
lr = LinearRegression()
svr_linear = SVR(kernel = 'linear')
svr_rbf = SVR(kernel = 'rbf')
xgboost = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

# Checking performance
regressors=[lasso, el_reg, krr, lr, svr_linear, svr_rbf, xgboost]

from sklearn.model_selection import cross_val_score
def rmse(model,X_train,Y_train):
  rmse= np.sqrt(-cross_val_score(model, X_train, Y_train, 
                                 scoring="neg_mean_squared_error", cv = 6))  
  return(rmse.mean())

mean_error = []  
r2score = []
for i in regressors:
  mean_error.append(rmse(i,X_train,Y_train))
  print('hi')
  i.fit(X_train,Y_train)
  r2score.append(i.score(X_test,Y_test))

# Creating output matrix
lasso.fit(X_train,Y_train)
lst_1 = lasso.predict(X_test).tolist()  

el_reg.fit(X_train,Y_train)
lst_2 = el_reg.predict(X_test).tolist()  

krr.fit(X_train,Y_train)
lst_3 = np.array(pd.DataFrame(krr.predict(X_test)).iloc[:, -1]).tolist() 

lr.fit(X_train,Y_train)
lst_4 = np.array(pd.DataFrame(lr.predict(X_test)).iloc[:, -1]).tolist()  

svr_linear.fit(X_train,Y_train)
lst_5 = svr_linear.predict(X_test).tolist()  

svr_rbf.fit(X_train,Y_train)
lst_6 = svr_rbf.predict(X_test).tolist()  

xgboost.fit(X_train,Y_train)
lst_7 = xgboost.predict(X_test).tolist()  

dict = {'lasso' : lst_1, 'el_reg' : lst_2, 'krr' : lst_3, 'lr' : lst_4,
        'svr_linear' : lst_5, 'svr_rbf' : lst_6, 'xgboost' : lst_7}
df_feature_ouput = pd.DataFrame(dict)


"""# creating meta regressor
from sklearn.neural_network import MLPRegressor
meta_reg = MLPRegressor(hidden_layer_sizes =(15,15),activation='relu',
                        batch_size='auto', solver='adam', alpha=0.1, max_iter = 100)

# Creating stacked regressor
stregr = StackingRegressor(regressors=[lr, lasso, svr_linear, svr_rbf, svr_poly, dec_tree, rand_forest], 
                           meta_regressor= meta_reg)"""

# Part 2 - LEFT ANN creation ON 

# importing Keras Libraries
import tensorflow as tf
import keras

# module to initialize ANN
from keras.models import Sequential
# module to create layers in ANN
from keras.layers import Dense

# Initalizing ANN
regressor = Sequential()

# Adding Input layer and first hidden layer
regressor.add(Dense(10, input_dim = 3, activation = 'relu'))

# Adding 2nd hidden layer
regressor.add(Dense(10, input_dim = 10, activation = 'relu'))

# Adding 3rd hidden layer
regressor.add(Dense(10, input_dim = 10, activation = 'relu'))

# Adding output layer
regressor.add(Dense(1, input_dim = 10, activation = 'linear'))

# Compiling ANN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mse'])

# Fitting the ANN to the dataset
regressor.fit(X_train, Y_train, epochs =50)

# predicting the test set
y_pred = regressor.predict(X_test)
from sklearn.metrics import r2_score
r2_score(Y_test,y_pred)

y_test_re = sc.inverse_transform(Y_test)
y_pred_re = sc.inverse_transform(y_pred)

df_ANN_feature = pd.DataFrame(y_pred)

final_df = pd.concat([df_ANN_feature,df_feature_ouput],axis = 1)

# PART3 : FITTIN FINAL MODULE TO final_df

# data preprocessing
X_fin = final_df.iloc[:,0:8].values
Y_fin = Y_test
# Initalizing ANN
regressor_fin = Sequential()

# Adding Input layer and first hidden layer
regressor_fin.add(Dense(15, input_dim = 8, activation = 'relu'))

# Adding 2nd hidden layer
regressor_fin.add(Dense(15, input_dim = 15, activation = 'relu'))

# Adding 3rd hidden layer
regressor_fin.add(Dense(15, input_dim = 15, activation = 'relu'))

# Adding output layer
regressor_fin.add(Dense(1, input_dim = 15, activation = 'linear'))

# Compiling ANN
regressor_fin.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mse'])

# Fitting the ANN to the dataset
regressor_fin.fit(X_fin, Y_fin, epochs =100)

# predicting the test set
y_pred_fin = regressor_fin.predict(X_fin)
from sklearn.metrics import r2_score
r2_score(Y_fin,y_pred_fin)

y_pred_fin_re = sc.inverse_transform(y_pred_fin)













