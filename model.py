from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
angleTrain = pd.read_csv("Angle_Training.txt",header=None,sep='\t').to_numpy()
spikeTrain = pd.read_csv("Training_SpikeTrains.txt",header=None,sep='\t').to_numpy()
angleTest = pd.read_csv("Angle_Testing.txt",header=None,sep='\t').to_numpy()
spikeTest = pd.read_csv("Testing_SpikeTrains.txt",header=None,sep='\t').to_numpy()
angleTrain=np.where(angleTrain<90,0,angleTrain)
angleTrain=np.where((angleTrain>=90) & (angleTrain<180),1,angleTrain)
angleTrain=np.where((angleTrain>=180) & (angleTrain<270),2,angleTrain)
angleTrain=np.where((angleTrain>=270) & (angleTrain<=360),3,angleTrain)
angleTest = np.where(angleTest < 90, 0, angleTest)
angleTest = np.where((angleTest >= 90) & (angleTest < 180), 1, angleTest)
angleTest = np.where((angleTest >= 180) & (angleTest < 270), 2, angleTest)
angleTest=np.where((angleTest>=270) & (angleTest<=360),3,angleTest)
scores=[]
plt.plot(scores)
plt.show()
for i in range (1,300):
    print(i)
    classifier=KNeighborsClassifier(n_neighbors=i)
    print(spikeTrain.shape)
    print(angleTrain.shape)
    classifier.fit(spikeTrain.T,angleTrain.T.reshape(8000,))
    scores.append(classifier.score(spikeTest.T,angleTest.T))
plt.plot(scores)
plt.show()
# print(classifier.predict(spikeTrain.T))


