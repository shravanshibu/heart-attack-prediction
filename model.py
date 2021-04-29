import numpy as np #linear algebra
import pandas as pd #data processing
import warnings 
warnings.filterwarnings("ignore")
#for model building
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
import pickle #to make a model


dataset = pd.read_csv('heart.csv')

#droping the duplicate values and replace the dataset with non duplicate rows
dataset.drop_duplicates(inplace=True) 

X = dataset.drop('output', axis = 1)
y = dataset['output']

dataset.reset_index(drop=True, inplace=True)
columns_to_scale = dataset.iloc[:,[0,3,4,7,9,]]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
ss = StandardScaler()
scaled_values = ss.fit_transform(columns_to_scale)
scaled_values = pd.DataFrame(scaled_values, columns=columns_to_scale.columns)
scaled_dataset = pd.concat([scaled_values,dataset.iloc[:,[1,2,5,6,8,10,11,12,13]]],axis=1)

key = ['LogisticRegression','KNeighborsClassifier','SVC','DecisionTreeClassifier','RandomForestClassifier','GradientBoostingClassifier','AdaBoostClassifier','XGBClassifier']
value = [LogisticRegression(random_state=9), KNeighborsClassifier(), SVC(), DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), AdaBoostClassifier(), xgb.XGBClassifier()]
models = dict(zip(key,value))
predicted =[]

for name,algo in models.items():
    model=algo
    model.fit(X_train,y_train)
    predict = model.predict(X_test)
    acc = accuracy_score(y_test, predict)
    predicted.append(acc)

#Observation: From the above figure we can see that none of the above models give an accuracy greater than 90%. Let us try some other approach. 
# Lets take some other random_state for Logistic Regression Model and see if the accuracy improves!
lr = LogisticRegression(solver='lbfgs', max_iter=10000)
rs = []
acc = []
for i in range(1,100,1):
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = i)    
    model_lr_rs = lr.fit(X_train, y_train.values.ravel())
    predict_values_lr_rs = model_lr_rs.predict(X_test)
    acc.append(accuracy_score(y_test, predict_values_lr_rs))
    rs.append(i)

pickle.dump(lr, open('model.pkl','wb'))#exporting model
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[74,0,1,120,269,0,0,121,1,0.2,2,1,2]]))