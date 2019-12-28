'''These data are the results of a chemical analysis of wines grown in the same region in Italy
but derived from three different cultivars. The analysis determined the quantities of 13
constituents found in each of the three types of wines.
Wine data set contains 13 features

Design machine learning application which follows below steps as
Step 1:
Get Data
Load data from WinePredictor.csv file into python application.
Step 2:
Clean, Prepare and Manipulate data
As we want to use the above data into machine learning application we have prepare
that in the format which is accepted by the algorithms.
Step 3:
Train Data
Now we want to train our data for that we have to select the Machine learning algorithm.
For that we select K Nearest Neighbour algorithm.
use fit method for training purpose.
For training use 70% dataset and for testing purpose use 30% dataset.
Step 4:
Test Data
After successful training now we can test our trained data by passing some value of
wether and temperature.
As we are using KNN algorithm use value of K as 3.
After providing the values check the result and display on screen.
Result may be Yes or No.

Step 5:
Calculate Accuracy
Write one function as CheckAccuracy() which calculate the accuracy of our algorithm.
For calculating the accuracy divide the dataset into two equal parts as Training data and
Testing data.
Calculate Accuracy by changing value of K.
Before designing the application first consider all features of data set.

'''


import pandas as pd;
import numpy as np;
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_path='WinePredictor.csv';
data=pd.read_csv(data_path)

data.columns=[c.replace(' ','_')for c in data.columns]


feature_name=['Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','Color intensity','OD280/OD315 of diluted wines','Proline']

print("Name offeature ",feature_name);



Alcohol=data.Alcohol
malic=data.Malic_acid
Ash=data.Ash
AAsh=data.Alcalinity_of_ash
Magnesium=data.Magnesium
Tphenols=data.Total_phenols
Flavanoids=data.Flavanoids
Nphenols=data.Nonflavanoid_phenols
Pc=data.Proanthocyanins
colori=data.Color_intensity
Hue=data.Hue
Proline=data.Proline
Class=data.Class



feature=list(zip(Alcohol,malic,Ash,AAsh,Magnesium,Tphenols,Flavanoids,Nphenols,Pc,colori,Hue,Proline,Class))

data_train,data_test,target_train,target_test=train_test_split(data,Class,test_size=0.5);

model=KNeighborsClassifier(n_neighbors=3);

model.fit(data_train,target_train)

predicted=model.predict(data_test);

Accuracy=accuracy_score(target_test,predicted)

print("Accuracy of cassification alogritham with k Neighbor classifier is ",Accuracy*100,"%");



