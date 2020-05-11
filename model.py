from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Reading the data ,header=None as there are no headers,.to_numpy() converts the pandas dataFrame to a Numpy 
# for later user
angleTrain = pd.read_csv("Angle_Training.txt",header=None,sep='\t').to_numpy()
spikeTrain = pd.read_csv("Training_SpikeTrains.txt",header=None,sep='\t').to_numpy()
angleTest = pd.read_csv("Angle_Testing.txt",header=None,sep='\t').to_numpy()
spikeTest = pd.read_csv("Testing_SpikeTrains.txt",header=None,sep='\t').to_numpy()
# Converting the angles in both training and testing data to 4 classes
# for example  if angle = 200 then it belongs to class 3 which is between 180 to 270 degrees
angleTrain=np.where(angleTrain<90,0,angleTrain)
angleTrain=np.where((angleTrain>=90) & (angleTrain<180),1,angleTrain)
angleTrain=np.where((angleTrain>=180) & (angleTrain<270),2,angleTrain)
angleTrain=np.where((angleTrain>=270) & (angleTrain<=360),3,angleTrain)
angleTest = np.where(angleTest < 90, 0, angleTest)
angleTest = np.where((angleTest >= 90) & (angleTest < 180), 1, angleTest)
angleTest = np.where((angleTest >= 180) & (angleTest < 270), 2, angleTest)
angleTest=np.where((angleTest>=270) & (angleTest<=360),3,angleTest)
# Initialize our scores array that we will append the scores for each K value later
scores=[]
for i in range (1,300): #Loop through values of K from 1 to 300 and append scores
    print(i)
    classifier=KNeighborsClassifier(n_neighbors=i)
    print(spikeTrain.shape)
    print(angleTrain.shape)
    classifier.fit(spikeTrain.T,angleTrain.T.reshape(8000,))
    scores.append(classifier.score(spikeTest.T,angleTest.T))
plt.plot(scores)
plt.show()
# print(classifier.predict(spikeTrain.T))


