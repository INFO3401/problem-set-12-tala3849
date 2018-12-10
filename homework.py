#PROBLEMS 1 - 3 I WORKED WITH HANNAH AND MARISSA ON. WE ALL MET UP AND WORKED THROUGH THE PROBLEMS 


 
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

#question 3 - with the accuracy score and F1 being .45 it means that the algorithm is not very accurate when predicting the players positions. 



def classifyPlayer(targetRow, data, model, prediction, ignore):
    X = targetRow.drop(columns = [prediction, ignore])
    
    
    #Determine the five closets neightbors to our target row 
    neighbors = model.kneighbors(X, n_neighbors=5, return_distance=False)
    
    # print out the neightbors data 
    
    for neighbor in neighbors[0]:
        print(data.iloc[neighbor])


#**********PLEASE READ!!!********
# QUESTIONS 1 - 3 AND QUESTION 6 ARE MARISSA HANNAH AND I'S CODE THAT WE FIGURED OUT ON OUR OWN BY WORKING TOGETHER. QUESTIONS 4, 5, AND 7 ARE JACOBS CODE - HE MET WITH HANNAH MARISSA AND I TO GO THROUGH AND EXPLAIN WHAT HE DID AND WHY. WE THEN WENT THROUGH AND COMMENTED THROUGH TO SHOW THAT WE UNDERSTAND WHAT HE DID. WE JUST WANTED TO LET YOU KNOW THAT THE NEXT FOUR QUESTIONS ARE JACOBS CODE BUT OUR COMMENTS. WE DIDNT WANT YOU TO THINK WE WERE CHEATING, BUT WE ALSO WANTED TO SHOW YOU THAT WE TOOK THE STEPS TO LEARN TO HOW TO DO IT. WE THOUGHT IT WOULD BE MORE BENEFICIAL TO SHOW THAT WE HAD SOMEONE EXPLAIN TO US HOW TO DO IT RATHER THAN  NOT PUT ANYTHING AT ALL BUT IT IS UP TO YOU IF YOU WANT TO GRADE THIS OR NOT - WE JUST WANTED TO SHOW OUR EFFORTS. 


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
        accuracies.append(accuracy) #appends the accuracy into the list made above 
        print("Fold " + str(fold) + ":" + str(accuracy)) # prints a statement of the put 

    return np.mean(accuracies)


# Problem 5:
def determineK(dataset, prediction, ignore, k_vals):
    best_k = 0 # this will create an integer value 
    best_accuracy = 0 # creates an integer value 
    # the values will be replaced in the for loop

    for k in k_vals: # for loop to loop through each k (5, 7, 10) in the input
        current_k = runkNNCrossfold(dataset, prediction, ignore, k) # runs the kNN cross validation function that we created in problem 4 for each k value in k_vals
        if current_k > best_accuracy: # asks if the current k value is better than the best accuracy that is stored
            best_k = k # if that k is better than the stored best accuracy then set the variable k to the k value being looped through
            best_accuracy = current_k # stores the current k value as best accuracy

    print("Best k, accuracy = " + str(best_k) + ", " + str(best_accuracy)) # prints the best k and current k value accuracy as the output in the terminal

# Problem 6 - worked on with Taylor and Marissa (without Jacob's help)
def runKMeans(dataset, ignore, neighbors):
    # set up dataset
    X = dataset.drop(columns = ignore)
    
    # run k-means algorithm
    kmeans = KMeans(n_clusters = neighbors)
    
    # train the model
    kmeans.fit(X)
    
    # add the predictions to the dataframe
    dataset['cluster'] = pd.Series(kmeans.predict(X), index = dataset.index)
    
    return kmeans

# Problem 7:
#Adapted from: https://datascience.stackexchange.com/a/41125
def findClusterK(dataset, ignore):
    
    mean_distances = {} # creates an empty dictionary
    X = dataset.drop(columns=ignore) # sets up the dataset
    
    for n in np.arange(4,12):
        model = runKMeans(dataset, ignore, n) #run the model from problem 6
        mean_distances[n] = np.mean([np.min(x) for x in model.transform(X)]) # use .transform() to get the distances of the points from all clusters. Then use list comprehension to get the min of those distances for each point to get the distance from the cluster the point belongs to. Take the mean of that list to get average distance.

    print("Best k by average distance: " + str(min(mean_distances, key=mean_distances.get))) #  this prints the best k based on the average distance to the other points, then use .get to return the value in the mean_distances key
    

    
nbaData = loadData("nba_2013_clean.csv")

knnModel = runKNN(nbaData, "pos", "player", 3)


for k in [5,7,10]: #this is looping through the different ways that they trained it 
    print("Folds: " + str(k))
    runkNNCrossfold(nbaData,"pos", "player", k)
    
    
    

for k in [5,7,10]:
    print("Folds: " + str(k))
    runkNNCrossfold(nbaData,"pos", "player", k)
    
determineK(nbaData,"pos", "player", [5,7,10])

kmeansModel = runKMeans(nbaData, ['pos', 'player'], 5)

findClusterK(nbaData, ['pos', 'player'])





