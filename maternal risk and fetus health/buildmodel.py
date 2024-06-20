
import pandas as pd
import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

dataset=pd.read_csv('pima-indians-diabetes.csv',names=['prgnum','glupl','bp','skinth','insulin','bodymass','diabped','age','outcome'],header=None)
X = dataset.iloc[:,[0,1,2,3,4,5,6,7]]
Y =dataset.iloc[:,[8]]


Y=np.ravel(Y)
(trainData, testData, trainLabels, testLabels) = train_test_split(X,Y,
	 test_size=0.25, random_state=100)

classifier = KNeighborsClassifier(n_neighbors=9)
print("[INFO] training model...")
classifier.fit(trainData, trainLabels)


print("[INFO] evaluating...")
predictions = classifier.predict(testData)
print(classification_report(testLabels, predictions))

import pickle

modelfile=open('knndiabetic.pkl','wb')
pickle.dump(classifier,modelfile)
modelfile.close()