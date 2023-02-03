import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

df = pd.read_csv('boston - Copy (2).csv')
print(df.dtypes)

# this code to ensure that there isnt missing values in the data set
print(df.isnull().sum())

df.describe()


corr=df.corr()
corr.shape
corr.style.background_gradient(cmap = 'brg_r') 


plt.figure(figsize=(15,15))
sns.heatmap(corr, vmin=-1, vmax=1, annot=True, cmap='bwr_r')
plt.title("Correlation_Matrix\n",color="black",fontsize=20)
plt.show()

# partition the data into dependent AND independant  (x and y)
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

print(X.head(5))
print(y.head(5))



MX = MinMaxScaler()
X = MX.fit_transform(X)

#this code for draw first four features with y
x_plot = df.iloc[:,0]
x_plot2 = df.iloc[:,1]
x_plot3 = df.iloc[:,2]
x_plot4 = df.iloc[:,3]

plt.scatter(x_plot, y, color='blue',alpha=0.5)
plt.scatter(x_plot2, y, color='red',alpha=0.5)
plt.scatter(x_plot3, y, color='black',alpha=0.5)
plt.scatter(x_plot4, y, color='red',alpha=0.5)


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)


x_train.shape
x_test.shape


# Linear Regression model code
model = LinearRegression(fit_intercept=True, normalize=True,copy_X=True,n_jobs=-1)
model.fit(x_train,y_train)


model.score(x_train,y_train)
model.score(x_test,y_test)


y_predict = model.predict(x_test)

plt.plot(x_plot, model.predict(X), color='red', linewidth=2)


for i in range(5):
    print(y_predict[i])
    

MSE = mean_squared_error(y_predict,y_test)
print("MSE is : ",MSE)

# DecisionTree algorithm code

DecisionTreeRegressorModel = DecisionTreeRegressor( max_depth=6,random_state=33)
DecisionTreeRegressorModel.fit(x_train, y_train)

print('training score using desision tree',DecisionTreeRegressorModel.score(x_train,y_train))
print('testing score using desision tree',DecisionTreeRegressorModel.score(x_test,y_test))

#here code to choose best number for max depth
for i in range(1,11):
    DTRM = DecisionTreeRegressor( max_depth=i,random_state=33)
    DTRM.fit(x_train, y_train)
    print('training score using desision tree for i =',i,DTRM.score(x_train,y_train))
    print('testing score using desision tree for i =',i,DTRM.score(x_test,y_test))
    
#best number for depth that i found is 6 so i will apply algorithm on it

#predict y_test values using decisiontree model
y_predict_DTR = DecisionTreeRegressorModel.predict(x_test)
#compute MSE for decisiontree model
MSE_DTR = mean_squared_error(y_predict_DTR,y_test)
print("MSE is : ",MSE_DTR)

# random forest algorithm code

RFR = RandomForestRegressor(n_estimators=10, random_state=0)
RFR.fit(x_train,y_train)

print('training score using random forest',RFR.score(x_train,y_train))
print('testing score using random forest',RFR.score(x_test,y_test))

n_estimators_range = range(10, 200, 10)

# Keep track of the scores for each value of n_estimators
scores = []

for n_estimators in n_estimators_range:
    clf = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
    scores.append(np.mean(cross_val_score(clf, x_train, y_train, cv=5)))

# Find the value of n_estimators that gives the highest mean score
best_n_estimators = n_estimators_range[np.argmax(scores)]
print("Best n_estimators:", best_n_estimators)

y_predict_MSE_for_randforest = RFR.predict(x_test)
MSE_for_randforest = mean_squared_error(y_predict_MSE_for_randforest,y_test)
print('mse for random forest is : ',MSE_for_randforest)

