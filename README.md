# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE
```
Developed By: Mohanish.K
Register No: 212222100028
```
```python
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
```
```python
x = load_boston()
df = pd.DataFrame(x.data, columns = x.feature_names)
df["PRICE"] = x.target
X = df.drop("PRICE",1) 
y = df["PRICE"]          
df.head(10)
```
```python
plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
```
```python
cor_target = abs(cor["PRICE"])
relevant_features = cor_target[cor_target>0.5]
relevant_features
```
```python
print(df[["LSTAT","PTRATIO"]].corr())
print(df[["RM","LSTAT"]].corr())
print(df[["RM","PTRATIO"]].corr())
print(df[["PRICE","PTRATIO"]].corr())
```
```python
X_1 = sm.add_constant(X)
model = sm.OLS(y,X_1).fit()
model.pvalues
```
```python
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
```
```python
model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, 7)
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  
#Fitting the data to model
model.fit(X_rfe,y)
print(rfe.support_)
print(rfe.ranking_)
```
```python
nof_list=np.arange(1,13)            
high_score=0
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))
```
```python
cols = list(X.columns)
model = LinearRegression()
rfe = RFE(model, 10)             
X_rfe = rfe.fit_transform(X,y)  
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)
```
```python
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
```
```python
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
```
# OUPUT

```yaml
Feature selection can be done in multiple ways but there are broadly 3 categories of it:
1. Filter Method
2. Wrapper Method
3. Embedded Method
```

DATASET:

![d1](https://user-images.githubusercontent.com/119559844/234180495-7ac5423e-bbcb-4f23-9fbd-e2e0213e8c7c.png)

FILTER METHOD:
```
The filtering here is done using correlation matrix and it is most commonly done using Pearson correlation.
```
HIGHLY CORRELATED FEATURE WITH OUTPUT VARIABLE PRICE:

![d2 ](https://user-images.githubusercontent.com/119559844/234180584-bd55abbe-3f20-4b7b-8748-871fb841063a.png)

CHECKING CORRELATION WITH EACH OTHER:

![d3 ](https://user-images.githubusercontent.com/119559844/234180593-46ca847e-eea2-4d2a-b3fe-2c0afccdbc25.png)

WRAPPER METHOD:

```
Wrapper Method is an iterative and computationally expensive process but it is more accurate than the filter method.

There are different wrapper methods such as Backward Elimination, Forward Selection, Bidirectional Elimination and RFE.
```
BACKWARD ELIMINATION:

![d4](https://user-images.githubusercontent.com/119559844/234180602-2a9effbc-7fdd-492a-bc5b-6ce830880990.png)

![d5](https://user-images.githubusercontent.com/119559844/234180613-3eb9cec5-43e2-43fc-9bd3-0b7a66c4bb20.png)

RECURSIVE FEATURE ELIMINATION(RFE):

![d6](https://user-images.githubusercontent.com/119559844/234180624-16e2d1dc-8371-4766-9f50-932fa8264c50.png)

NUMBER OF FEATURE HAVING HIGH ACCURACY:

![d7](https://user-images.githubusercontent.com/119559844/234180627-2a86e228-9d2f-40de-ae60-3825826dfd8d.png)

FINAL SET OF FEATURE:

![d8](https://user-images.githubusercontent.com/119559844/234180638-beaa20ed-a6a5-4303-b5e0-6e601a53a9ae.png)

EMBEDDED METHOD:

![d9](https://user-images.githubusercontent.com/119559844/234180646-6198293f-0ece-4861-80eb-2d937eed6ea1.png)



RESULT:

Various feature selection techniques have been performed on a given dataset successfully.
