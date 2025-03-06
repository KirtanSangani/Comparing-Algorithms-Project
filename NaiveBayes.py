import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import sklearn.model_selection as model_selection
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("Edited Car Evaluation - Sheet1.csv")

X = data.drop('Target', axis=1)
y = data['Target']

X, y = make_classification(n_samples = 1000, n_features = 2, n_informative = 2, n_clusters_per_class = 1, n_redundant = 0, n_classes = 4, random_state=42)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, random_state=42)

model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

gaussian_accuracy = accuracy_score(y_test, y_pred)
print("Gaussian Accuracy: ", gaussian_accuracy)

x_min, x_max = X_train[:,0].min()-1,X_train[:,0].max()+1
y_min, y_max = X_train[:,1].min()-1,X_train[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max,0.1),
                     np.arange(y_min, y_max,0.1))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA']))
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=ListedColormap(['r', 'g', 'b', 'y']), edgecolors='k')
plt.xlabel('Capacity')
plt.ylabel('Safety')
plt.title('Gaussian Naive Bayes Classifier Decision Regions')
plt.show()

model = BernoulliNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Bernoulli Accuracy: ", accuracy)

x_min, x_max = X_train[:,0].min()-1,X_train[:,0].max()+1
y_min, y_max = X_train[:,1].min()-1,X_train[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max,0.1),
                     np.arange(y_min, y_max,0.1))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA']))
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=ListedColormap(['r', 'g', 'b', 'y']), edgecolors='k')
plt.xlabel('Capacity')
plt.ylabel('Safety')
plt.title('Bernoulli Naive Bayes Classifier Decision Regions')
plt.show()
