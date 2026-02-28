#-------------------------------------------------------------------------
# AUTHOR: Andrew Mazmanian
# FILENAME: naive_bayes.py
# SPECIFICATION: Naive Bayes Classifier to predict if it is going to rain or not tomorrow based on the given training and test data.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 4.5 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM
#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd

dbTraining = []
dbTest = []

#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
X = []
for row in dbTraining:
    outlook = 1 if row[1] == 'Sunny' else 2 if row[1] == 'Overcast' else 3
    temp = 1 if row[2] == 'Hot' else 2 if row[2] == 'Mild' else 3
    humidity = 1 if row[3] == 'High' else 2
    windy = 1 if row[4] == 'Weak' else 2
    X.append([outlook, temp, humidity, windy])

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
Y = []
for row in dbTraining:
    Y.append(1 if row[5] == 'Yes' else 2)

#Fitting the naive bayes to the data using smoothing
clf = GaussianNB()
clf.fit(X, Y)

#Reading the test data using Pandas
df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())

#Printing the header os the solution
print(f"{"Day":<10} {"Outlook":<10} {"Temperature":<15} {"Humidity":<10} {"Windy":<10} {"PlayTennis":<10}{"Confidence":<10}")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
for row in dbTest:
    outlook = 1 if row[1] == 'Sunny' else 2 if row[1] == 'Overcast' else 3
    temp = 1 if row[2] == 'Hot' else 2 if row[2] == 'Mild' else 3
    humidity = 1 if row[3] == 'High' else 2
    windy = 1 if row[4] == 'Weak' else 2
    
    proba = clf.predict_proba([[outlook, temp, humidity, windy]])[0]
    confidence = max(proba)
    
    if confidence >= 0.75:
        if proba[0] > proba[1]:
            prediction = 'Yes'
        else:  
            prediction = 'No'
        
        print(f"{row[0]:<10} {row[1]:<10} {row[2]:<15} {row[3]:<10} {row[4]:<10} {prediction:<10}{confidence:.2%}")