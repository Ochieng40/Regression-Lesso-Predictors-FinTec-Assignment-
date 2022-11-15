#!/usr/bin/env python
# coding: utf-8

# # 1.0 Data Preparation

# In[92]:


import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import seaborn as sns


# In[137]:


# Importing the dataset to analyse
df = pd.read_csv("C:\\Users\\DENIS OCHIENG'\\Desktop\\Cherop\\Dataset.csv")


# In[13]:


#Identifying null values
df.isna().sum()


# In[14]:


#Identifying unique values
df.nunique()


# In[24]:


# select only the needed columns
df_clean = df[["trade_date","age","distance_to_MTR","number_of_stores","house_price_per_unit"]]
df_clean.head()


# In[27]:


# check the correlation between the data variables


df_clean.corr()


# The relationship between two variables is generally considered strong 
# when their r value is greater than 0.7 . The correlation r measures the 
#strength of the linear relationship between two quantitative variables.


# # 2.0 Linear Regression

# ## 2.1 Explanation of the Steps
# 
# The analysis looked at the effect of age on the price of houses per unit.
# The first step involved plotting a scatterplot to observe the correlation between the two variables and then you add the line of best fit to observe the slope in the correlation. I have set the line of best fit as orange for
# visibility purposes.
# 

# ## 2.2 Code

# In[202]:


# trend line for the regression
# Here we are plotting our scatterplot with the line of best fit. 


sns.set_style('dark')
sns.lmplot(data = df, x = 'age',
            y = 'house_price_per_unit',
          line_kws={"color": "C1"}).set(title= "Age against House Price Per Unit")


# ## 2.3 RESULTS
# 
# We observe a downward slope in the line of best fit; which means that a unit increase in age results to a unit decrease in house price per unit.
# Hence there is a negative correlation between age and house per unit.
# 

# # 3.0 Lesso Regression

# ## 3.1 Explanation of the Steps
# 
# This model tries to predict how the independent variable age can be used
# to predict the behavior of the dependent variable house per unit price.
# 
# First step involves splitting our dataframe into x and y. X will contain the independent variables and y will contain the dependent variables.
# Then the step is followed by splitting the x and y variables into train and test. Where we import from sklearn.import and 
# then make the x_train and y_train and pass it under x and y and then state random as 0.
# This is followed by checking the shape of the X_train and y_train.
# Thereafter, you import the Lasso reggression and fit it on the X_train and y_train and then make prediction using a new variable in the X_test.
# Finally, you look at the predictions available by calling the y_pred (The output will be the predicted values of the house price per unit).
# 
# Other important steps conducted later is checking the R-Squared value, the slope, the intercept and the mean square root.

# ## 3.2 Code

# In[183]:


# First we plit our dataframe into x and y. X will contain the independent variables
# and y will contain the dependent variables.

X = df[['age','number_of_stores']]
y = df['house_price_per_unit']


# In[184]:


# Then the step is followed by splitting the x and y variables into train and test.
# where you import from sklearn.import and then make the x_train and y_train and pass it under x and y and then state random as 0.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[185]:


# Here we check the shape of our X_train

X_train.shape


# In[186]:


# Here we check the shape of our y_train

y_train.shape


# In[187]:


# Then we import the Lasso reggression

from sklearn.linear_model import Lasso
Lasso = Lasso(alpha = 1.0)


# In[188]:


# Then next you fit it on our X_train and y_train

Lasso.fit(X_train, y_train)


# In[189]:


# Then we make our prediction using a new variable in the X_test

y_pred = Lasso.predict(X_test)


# In[190]:


# Then we look at the predictions available by calling the y_pred
# The output will be the predicted values of the house price per unit.
y_pred


# In[192]:


# Then we will print the slope and intercept

print("Slope: %.2f" % Lasso.coef_[0] )


# In[ ]:


## 2.2 RESULTS


# In[193]:


print("Intercept: %.2f" % Lasso.intercept_)


# In[194]:


# Then we evaluate the type of our model. First is through getting the mean squred error

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)


# In[195]:


# Second we get the rmse which is the squareroot of the mean squared error

np.sqrt(mean_squared_error(y_test, y_pred))


# In[201]:


# Third we get the rsquared

print("Rsquared: %.2f" % Lasso.score(X_test, y_test))


# In[200]:


# Then lastly we plot our prediction using sns seaborne

sns.distplot(y_test-y_pred)


# ## 3.3 Results
# 
# Our r-square value tells us that only 37% of the variability in 
# the target variable is what is explained by the regression model.
# The low r-square value shows that the predicability cannot be highly relied on since the data
# lacks a good fit.
# The low negative slope indicates a weak negative correlation between 
# the independent and the dependent values.
# The Lasso intercept score tells us that the housing price per unit is 32.27 when
# the age and number of store are equal to zero.
