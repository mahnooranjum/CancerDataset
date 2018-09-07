
#===========================================================================
#	Author : MAHNOOR ANJUM
#	Description : Cancer Dataset Solution
#	Using multiple algorithms and plotting their results
#	Obtaining Classification reports
#
#	References:
#	SuperDataScience
#	Official Documentation
#
#	Data Source:
#	https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
#===========================================================================
# Importing the libraries
import numpy as np#Numpy for data importing, array management
import matplotlib.pyplot as plt#for data plotting 
import pandas as pd#for data manipulation, data mining, data munging

# Importing the dataset
dataset = pd.read_csv('data2.csv')#importing the label encoded dataset
X = dataset.iloc[:, 3:33].values#selecting the independent variables MATRIX
y = dataset.iloc[:, 2].values#selecting/slicing a dependent variable VECTOR


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split#Importing library module for splitting dataset into
#test and train subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)

# Feature Scaling
from sklearn.preprocessing import StandardScaler#for feature scaling
sc = StandardScaler()#making an instance of a class
X_train = sc.fit_transform(X_train)#fitting on the train set
X_test = sc.transform(X_test)#transforming on the test set 

#PRINCIPLE COMPONENT ANALYSIS

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio2

#===================================================================
#ARTIFICIAL NEURAL NETWORK
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense


# Initialising the ANN
classifierann = Sequential()

# Adding the input layer and the first hidden layer CHANGE INPUT_DIM TO 2 WITH PCA
classifierann.add(Dense(output_dim = 9, init = 'uniform', activation = 'relu', input_dim = 30))

# Adding the second hidden layer
#classifier.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu'))
# Adding the output layer
classifierann.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifierann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifierann.fit(X_train, y_train, batch_size = 20, nb_epoch = 100)

# Predicting the Test set results
y_pred_ann = classifierann.predict(X_test)
y_pred_ann = y_pred_ann.round()
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmann = confusion_matrix(y_test, y_pred_ann)
#===============================================================================

#NAIVE BAYES


from sklearn.naive_bayes import GaussianNB
classifiernb = GaussianNB()
classifiernb.fit(X_train, y_train)

y_pred_nb = classifiernb.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmnb = confusion_matrix(y_test, y_pred_nb)


from sklearn.metrics import accuracy_score
asnb = accuracy_score(y_test, y_pred_nb)
#===============================================================================

#KNN

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifierknn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifierknn.fit(X_train, y_train)

# Predicting the Test set results
y_predknn = classifierknn.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmknn = confusion_matrix(y_test, y_predknn)
# Making the predictions and evaluating the model
#===============================================================================
#DECISION TREES
# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifierdt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifierdt.fit(X_train, y_train)

# Predicting the Test set results
y_preddt = classifierdt.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmdt = confusion_matrix(y_test, y_preddt)

#===============================================================================
#KERNEL GAUSSIAN

from sklearn.svm import SVC
classifierksvm = SVC(kernel = 'rbf', random_state = 0)
classifierksvm.fit(X_train, y_train)

# Predicting the Test set results
y_predksvm = classifierksvm.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predksvm)

classifier = classifierann


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.55, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('ANN(Train set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.55, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()



#MODEL EVALUATION
from sklearn import metrics
# calculate the fpr and tpr for all thresholds of the classification
y_pred = classifier.predict(X_test)
#y_pred = y_prob[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1.02])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()







