#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns', None)
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.simplefilter("ignore")

#r2_score(y_test, y_pred)

df=pd.read_csv (r"C:\Users\Admin\Downloads\day (1).csv")
df.head()


# In[4]:


df.describe()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


###Drop un wanted Columns and Renaming

df['season']=df['season'].map({1:"spring", 2:"summer", 3:'fall', 4:'winter'})
df['yr']=df['yr'].map({0: '2018', 1:'2019'})
df['mnth']=df.mnth.map({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'June',7:'July',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'})
df['weathersit']=df.weathersit.map({1: 'Clear',2:'Mist + Cloudy',3:'Light Snow',4:'Snow + Fog'})
df['weekday']=df.weekday.map({0:'Sun',1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat'})


# In[9]:


df.info()


# In[10]:


df.head()


# In[11]:


############################ STEP 2 :Visulaizing the data ###################################################

# Analysing/visualizing the categorical columns
# to see how predictor variable stands against the target variable

sns.pairplot(df, vars=["temp", "hum",'casual','windspeed','registered','atemp','cnt','instant'])
plt.show()


# In[12]:


df.describe()


# In[13]:


df.head()


# In[14]:


#visualizing the categorical variables of the dataset using barplot 
plt.figure(figsize=(20, 12))
plt.subplot(2, 4, 1)
sns.boxplot(x='season', y='cnt', data=df)
plt.subplot(2, 4, 2)
sns.boxplot(x='mnth', y='cnt', data=df)
plt.subplot(2, 4, 3)
sns.boxplot(x='weekday', y='cnt', data=df)
plt.subplot(2, 4, 4)
sns.boxplot(x='weathersit', y='cnt', data=df)
plt.subplot(2, 4, 5)
sns.boxplot(x='yr', y='cnt', data=df)
plt.subplot(2, 4, 6)
sns.boxplot(x='workingday', y='cnt', data=df)
plt.subplot(2, 4, 7)
sns.boxplot(x='holiday', y='cnt', data=df)
plt.show()


# In[15]:


#visualizing the categorical variables of the dataset using barplot 
plt.figure(figsize=(20, 12))
plt.subplot(2, 4, 1)
sns.barplot(x='season', y='cnt', data=df)
plt.subplot(2, 4, 2)
sns.barplot(x='mnth', y='cnt', data=df)
plt.subplot(2, 4, 3)
sns.barplot(x='weekday', y='cnt', data=df)
plt.subplot(2, 4, 4)
sns.barplot(x='weathersit', y='cnt', data=df)
plt.subplot(2, 4, 5)
sns.barplot(x='yr', y='cnt', data=df)
plt.subplot(2, 4, 6)
sns.barplot(x='workingday', y='cnt', data=df)
plt.subplot(2, 4, 7)
sns.barplot(x='holiday', y='cnt', data=df)
plt.show()

##Some of the observations from the plots above are as follows####

###People are more likely to rent bikes in the fall and the summer season.
###Bike rental rates are the most in June,july,aug,sept and October.
###Saturday,monday,tuesday, Wednesday,Thursday and friday are the days where more bikes are rented
###Most bike rentals take place in the clear weather
###More bikes were rented in 2019
###There is no big discernable difference in bike rental rates depending on whether it's a working day or not
###Bike rental rates are higher on holidays


# In[16]:


plt.figure(figsize=(10, 4))
plt.subplot(1,2,1)
sns.boxplot(df["cnt"])
plt.subplot(1,2,2)
sns.distplot(df["cnt"])
plt.show()


# In[17]:


#Lets see correlation between numerical features using heatmap.

num_df=df[['temp','atemp','hum','windspeed','casual','registered','cnt']]


# In[18]:


num_df.corr()


# In[19]:


plt.figure(figsize=(8,6))
sns.heatmap(num_df.corr(),annot=True)

##temp and atemp have very strong correlation.
##registered,casual have also strong correlation with cnt.


# In[20]:


##Encode the data using get dummies

# dropping the columns which are not helpfull in our analysis.
#atemp is not needed as temp is already being used, dteday and casual are also not required for regression analysis 
df = df.drop(['atemp', 'instant', 'dteday', 'casual', 'registered'], axis=1)

df.head()


# In[21]:


#creating dummy variables
enc_df=pd.get_dummies(df,columns=['season','yr','mnth','weekday','weathersit'],drop_first=True)


# In[22]:


enc_df.head()


# In[23]:


##Train Split 

X=enc_df.drop(['cnt'],axis=1)
Y=enc_df['cnt']


# In[24]:


#splitting the dataset into train and test sets
X_train, X_test,Y_train, Y_test = train_test_split(X,Y,
                                   random_state=104, 
                                   test_size=0.25, 
                                   )


# In[25]:


X_train.head()


# In[26]:


print(X_train.shape)
print(X_test.shape)


# In[27]:


#Scaling

#Scaling the X_train using Minmax scaler.
scaler=MinMaxScaler()
x_train_var=['temp','hum','windspeed']


# In[28]:


#Scaling the X_train using Minmax scaler.
X_train[x_train_var]=scaler.fit_transform(X_train[x_train_var])


# In[29]:


X_train.head()


# In[30]:


#Scaling the X_test using Minmax scaler.
scaler=MinMaxScaler()
x_test_var=['temp','hum','windspeed']
X_test[x_test_var]=scaler.fit_transform(X_test[x_test_var])


# In[31]:


X_test.head()


# In[32]:


##Linear Regression Model Buiding Using Stats model

##import statsmodels.api as sm

#add a constant
X_train_sm = sm.add_constant(X_train)

#create first model
lr = sm.OLS(Y_train, X_train_sm)

#fit
lr_model = lr.fit()

#params
lr_model.params


# In[33]:


#model summary
lr_model.summary()


# In[34]:


##Featue Selection RFE

from sklearn.feature_selection import RFE
lm = LinearRegression()
lm.fit(X_train, Y_train)

#setting feature selection variables to 15
rfe = RFE(lm, n_features_to_select = 15) 

#fitting rfe ofject on our training dataset
rfe = rfe.fit(X_train, Y_train)


# In[35]:


#checking the elements selected and the ones rejected in a list after rfe
list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[36]:


# rfe true columns
rfe_true=X_train.columns[rfe.support_]


# In[37]:


rfe_true


# In[38]:


#creating training set with RFE selected variables
X_train_rfe=X_train[rfe_true]


# In[39]:


##Lets create model with these variable

#adding constant to training variable
X_train_rfe = sm.add_constant(X_train_rfe)

#creating first training model with rfe selected variables
lr = sm.OLS(Y_train, X_train_rfe)

#fit
lr_model_1 = lr.fit()

#params
lr_model_1.params


# In[40]:


lr_model_1.summary()


# In[41]:


vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[42]:


##Drop Holiday

X_train_rfe_new=X_train_rfe.drop(['holiday'],axis=1)


# In[43]:


lr = sm.OLS(Y_train,X_train_rfe_new)

#fit
lr_model_2 = lr.fit()

#summary
lr_model_2.summary()


# In[44]:


X_train_2 = X_train_rfe_new.drop(['const'], axis = 1)


# In[45]:


vif = pd.DataFrame()
X = X_train_2
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[46]:


#hum variable can be dropped due to a high VIF
X_train_new_3 = X_train_2.drop(['hum'], axis = 1)


# In[47]:


#adding constant to training variable
X_train_lr3 = sm.add_constant(X_train_new_3)

#creating first training model with rfe selected variables
lr = sm.OLS(Y_train, X_train_lr3)

#fit
lr_model = lr.fit()

#summary
lr_model.summary()


# In[48]:


#checking the VIF of the model 

#dropping the constant variables from the dataset
X_train_lr3 = X_train_lr3.drop(['const'], axis = 1)


# In[49]:


#calculating the VIF of the model
vif = pd.DataFrame()
X = X_train_new_3
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[50]:


#windspeed variable can be dropped due to high p value and negative correlation with cnt
X_train_new_4 = X_train_lr3.drop(['workingday'], axis = 1)


# In[51]:


#adding constant to training variable
X_train_lr4 = sm.add_constant(X_train_new_4)

#creating first training model with rfe selected variables
lr = sm.OLS(Y_train, X_train_lr4)

#fit
lr_model = lr.fit()

#summary
lr_model.summary()


# In[64]:


#adding constant to training variable
X_train_lr4 = sm.add_constant(X_train_new_4)

#creating first training model with rfe selected variables
lr = sm.OLS(Y_train, X_train_lr4)

#fit
lr_model = lr.fit()

#summary
lr_model.summary()


# In[66]:


#checking the VIF of the model 

#dropping the constant variables from the dataset
X_train_lr5 = X_train_lr4.drop(['const'], axis = 1)


# In[68]:


#calculating the VIF of the model
vif = pd.DataFrame()
X = X_train_new_4
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[69]:


# Final traing data 
X_train_new_4.head()


# In[70]:


##Residul Analysis ''

#adding constant
X_train_lrf=sm.add_constant(X_train_new_4)

lr = sm.OLS(Y_train, X_train_lrf)

#fit
lr_model_final = lr.fit()

#summary
lr_model_final.summary()


# In[71]:


y_train_pred=lr_model_final.predict(X_train_lrf)


# In[72]:


#calculating residual
residual=Y_train-y_train_pred


# In[73]:


sns.distplot(residual)
plt.title('Training error distribution')


# In[75]:


##predicion on test

#Eleminating columns from test data as well.
X_test_new=X_test[X_train_new_4.columns]


# In[76]:


X_test_new.shape


# In[77]:


X_test_lrf=sm.add_constant(X_test_new)


# In[78]:


#predict cnt value on test data
y_test_pred=lr_model_final.predict(X_test_lrf)


# In[79]:


#R2 score on traing data
r2_score_train=r2_score(Y_train,y_train_pred)
print("r2_score on training data is =",r2_score_train)
#R2 score on traing data
r2_score_test=r2_score(Y_test,y_test_pred)
print("r2_score on training data is =",r2_score_test)


# In[80]:


#finding out the mean squared error 

train_mse = (mean_squared_error(Y_train,y_train_pred))
test_mse = (mean_squared_error(Y_test,y_test_pred))
rmse_train=np.sqrt(train_mse)
rmse_test=np.sqrt(test_mse)
print('Route Mean squared error of the train set is', rmse_train)
print('Route Mean squared error of the test set is', rmse_test)


# In[ ]:


##The R-squared value of the train set is 83.52% 

##The R-squared value of the Test set  80.59%

##Which explains that 80% of variance in target variable is explained by input variables.

##We can conclude that the bike demands for the BoomBikes is company is dependent on the temperature,windspeed,year and season whether it is a workingday or not.

##In summer months also show low rental levels, a strong marketing strategy for the first 6 months of the year can assist in driving up the rental numbers.

##Rentals were more in 2019 than 2018 which suggests that over time more people would be exposed to this idea and there has to a strong analysis done to retain the repeat customers.

