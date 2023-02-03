# Boston House Prices-Advanced Regression Techniques
about dataset:
this dataset exists in kaggle it contains 506 records and it contains the following attributes
Input features in order:
1) CRIM: per capita crime rate by town
2) ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
3) INDUS: proportion of non-retail business acres per town
4) CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
5) NOX: nitric oxides concentration (parts per 10 million) [parts/10M]
6) RM: average number of rooms per dwelling
7) AGE: proportion of owner-occupied units built prior to 1940
8) DIS: weighted distances to five Boston employment centres
9) RAD: index of accessibility to radial highways
10) TAX: full-value property-tax rate per $10,000 [$/10k]
11) PTRATIO: pupil-teacher ratio by town
12) B: The result of the equation B=1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
13) LSTAT: % lower status of the population
Output variable:
1) MEDV: Median value of owner-occupied homes in $1000's [k$]

after applied multiple regression algorithms i have noticed the following results :
1-Linear Regression
training score : 0.73
testing score : 0.72
MSE : 20.7

2-Decision tree
training score : 0.94
testing score : 0.80  
MSE : 15.0

3-Random forest
training score : 0.96
testing score : 0.84  
MSE : 11.8

