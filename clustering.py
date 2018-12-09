import pandas as pd 
import seaborn as sns 


#import ML support libraries 

from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.cluster import KMeans 

def loadData(datafile):
    with open(datafile, 'r', encoding = 'latin1') as csvfile:
        data = pd.read_csv(csvfile)
        
        
        #inspect the data 
        print (data.columns.values)
        
        
        return data 
    
def runKNN(dataset, prediction, ignore, neighbors):
    #set up dataset 
    
    X = dataset.drop(columns=[prediction, ignore])
    Y = dataset[prediction].values 
    
    
    # split the data into a training and testing set 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, 
    random_state=1, stratify=Y)
    
    
    #run k-NN algorithm 
    
    knn = KNeighborsClassifier(n_neighbors= neighbors)
    
    # Train the model 
    knn.fit(X_train, Y_train)
    
    #test the model 
    
    score = knn.score(X_test, Y_test)
    print('Predicts' + prediction + "with" + str(score) + "accuracy")
    print("Chance is: " + str(1.0/len(dataset.groupby(prediction))))
    
    return knn

def classifyPlayer(targetRow, data, model, prediction, ignore):
    X = targetRow.drop(columns = [prediction, ignore])
    
    
    #Determine the five closets neightbors to our target row 
    neighbors = model.kneighbors(X, n_neighbors=5, return_distance=False)
    
    # print out the neightbors data 
    
    for neighbor in neighbors[0]:
        print(data.iloc[neighbor])
        
        
def runKMeans(dataset, ignore):

    #set up dataset 
    
    X = dataset.drop(columns=ignore)
    
    
    #Run k-means algorthm 
    
    kmeans = KMeans(n_clusters=5)
    
    
    #train the model 
    kmeans.fit(X)
    
    
    #Add the predictions to the dataframe 
    
    dataset['cluster'] = pd.Series(kmeans.predict(X), index=dataset.index)

    #print a scatterplot matrix 
    scatterMatrix = sns.pairplot(dataset.drop(columns=ignore), hue='cluster', palette = 'Set2')
    
    
    scatterMatrix.savefig("kmeansClusters.png")
    
    
    return kmeans 
    
        
    
#test your code 

nbaData = loadData("nba_2013_clean.csv")
knnModel = runKNN(nbaData, "pos", "player", 7)
classifyPlayer(nbaData.loc[nbaData['player']=='LeBron James'], nbaData, knnModel, 'pos', 'player')

kmeansModel = runKMeans(nbaData, ['pos', 'player'])

