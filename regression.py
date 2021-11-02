import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
import array

data = pd.read_csv('C:\\Data set\\salary_data.csv')

#to print the number of rows and columns in the data set
print(data.shape)

#shows the first five lines of the data set
print(data.head())

#shows the last five lines of the data set
print(data.tail())

#shows the random value of the data set
print(data.sample(5))

#checks the data types features
print(data.dtypes)

#describes the data in statically
print(data.describe())

#information about the data set
print(data.info())

#DATA CLEANING
#drops the duplicates
data = data.drop_duplicates()

#checks the null values
print(data.isnull().sum())

target = 'Salary'
#seperate the target for the target feature
y = data[target]
#seperate objects for input features
X = data.drop(target,axis=1)
print(X.shape)
print(y.head())
print(y.shape)

#DATA VISUALIZATION BEFORE THE TRAIN MODEL
plt.scatter(X,y)
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.grid()
plt.show()

#SPLIT DATASET TO TRAIN AND TEST
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)

#APPLY LINEAR REGRESSION ON THE TRAIN DATASET
regr = LinearRegression()
print(regr.fit(X_train,y_train))

#get the parameters
regr.intercept_
print(f'intercept(b) is : {regr.intercept_}')

regr.coef_
print(f'coefficient(m) is : {regr.coef_}')

#apply the model on the test dataset to get the prediction values
pred_y = regr.predict(X_test)
print(pred_y)
print(pred_y.shape)

#to compare the actual output values with the predicted values
data1 = pd.DataFrame({'Actual':y_test,'Predicted':pred_y,'variance':y_test-pred_y})
print(data1)

#prediction
pred = np.array([2.0]).reshape(-1,1)
print(regr.predict(pred))

#VISUALIZATION
#visualizing the training set results
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regr.predict(X_train),color='blue')
plt.title('Salary VS experience(training set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.grid()
plt.show()

#visualizing the test set results
#visualizing the testing set results
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regr.predict(X_train),color='blue')
plt.title('Salary VS Experience(Test_set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.grid()
plt.show()

#Evaluation metrics of regression algorithms
score = r2_score(y_test,pred_y)*100
print(f'Score : {score}')

print(f'Mean Absolute Error : {metrics.mean_absolute_error(y_test,pred_y)}')
print(f'Mean Squared Error : {metrics.mean_squared_error(y_test,pred_y)}')
print(f'Root Mean Squared Error : {np.sqrt(metrics.mean_squared_error(y_test,pred_y))}')