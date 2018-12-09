#I WORKED WITH HANNAH AND MARISSA ON THIS ASSIGNMENT. WE ALL MET UP AND WORKED THROUGH THE PROBLEMS 

import pandas as pd
import seaborn as sns
import numpy as np 

# import ML support libraries
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score 
from sklearn.cluster import KMeans 

def loadData(datafile):
    with open(datafile, 'r', encoding = 'latin1') as csvfile:
        data = pd.read_csv(csvfile)
        
    #inspect the data
    print(data.columns.values)
    
    return data

# MONDAY PROBLEM - worked with Hannah and Marissa

def runKNN(dataset, prediction, ignore, neighbors):
    
    # set up our dataset
    X = dataset.drop(columns = [prediction, ignore])
    Y = dataset[prediction].values
    
    # split the data into training and testing set
    # test size = what percent of the data do you want to test on
    # random_state = 1 = split them randomly
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.4, random_state = 1, stratify = Y)
    
    # run k-NN algorithm
    # n_neighbors = k-value
    knn = KNeighborsClassifier(n_neighbors = neighbors)
    
    # train the model
    knn.fit(X_train, Y_train)
    
    # test the model
    score = knn.score(X_test, Y_test)
    Y_prediction = knn.predict(X_test)
    print("Predicts " + prediction + " with " + str(score) + " accuracy ")
    print("Chance is: " + str(1.0 / len(dataset.groupby(prediction))))
    print("F1 score: " + str(f1_score(Y_test, Y_prediction, average = 'macro')))
    
    return knn

#question 2 - with the accuracy score and F1 being .45 it means that the algorithm is not very accurate when predicting the players positions. 



def classifyPlayer(targetRow, data, model, prediction, ignore):
    X = targetRow.drop(columns = [prediction, ignore])
    
    
    #Determine the five closets neightbors to our target row 
    neighbors = model.kneighbors(X, n_neighbors=5, return_distance=False)
    
    # print out the neightbors data 
    
    for neighbor in neighbors[0]:
        print(data.iloc[neighbor])




def runkNNCrossfold(dataset, prediction, ignore, neighbors):
    fold = 0 #making a counter 
    accuracies = [] #making a list 
    #using the KFold built in model from sklearn 
    kf = KFold(n_splits=neighbors)

    X = dataset.drop(columns=[prediction, ignore]) # this is where we are setting up X and Y 
    Y = dataset[prediction].values

    for train,test in kf.split(X): #making a loop for each k fold split 
        fold += 1 # counts which fold it is working on 
        knn = KNeighborsClassifier(n_neighbors=5) # uses the KNeighbors classifier for the 3 k inputs 
        knn.fit(X[train[0]:train[-1]], Y[train[0]:train[-1]]) # trains the classifier on each of the folds but removes the last one for testing 

        pred = knn.predict(X[test[0]:test[-1]]) # makes a test prediction 
        accuracy = accuracy_score(pred, Y[test[0]:test[-1]])
        accuracies.append(accuracy)
        print("Fold " + str(fold) + ":" + str(accuracy))

    return np.mean(accuracies)



nbaData = loadData("nba_2013_clean.csv")

# WEDNESDAY PROBLEMS
knnModel = runKNN(nbaData, "pos", "player", 3)


for k in [5,7,10]:
    print("Folds: " + str(k))
    runkNNCrossfold(nbaData,"pos", "player", k)

