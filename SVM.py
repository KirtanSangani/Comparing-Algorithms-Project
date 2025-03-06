import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
import sklearn.model_selection as model_selection
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint
from sklearn import svm
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("Edited Car Evaluation - Sheet1.csv")
data1 = data[:len(data)//6]
X = data1.drop('Target', axis = 1)
y = data1['Target']

X, y = make_classification(n_samples = 1000, n_features = 2, n_informative = 2, n_redundant = 0, n_classes = 4, n_clusters_per_class=1,random_state=42)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 101)

rbf = svm.SVC(kernel='rbf', gamma = 0.5, C=0.1).fit(X_train,y_train)
poly = svm.SVC(kernel='poly', degree = 3, C=1).fit(X_train,y_train)

poly_pred = poly.predict(X_test)
rbf_pred = rbf.predict(X_test)

rbf_accuracy = accuracy_score(y_test, rbf_pred)
rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
print(rbf_f1)
print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))

# Plot decision boundary for the RBF kernel
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

Z_rbf = rbf.predict(np.c_[xx.ravel(), yy.ravel()])
Z_rbf = Z_rbf.reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, Z_rbf, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, edgecolors='k')
plt.xlabel('Capacity ')
plt.ylabel('Safety ')
plt.title('SVM (RBF Kernel)')
plt.legend()
plt.show()

poly_accuracy = accuracy_score(y_test, poly_pred)
poly_f1 = f1_score(y_test, poly_pred, average='weighted')
print('Accuracy (Polynomial Kernel): ',"%.2f" % (poly_accuracy*100))
print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))

# Plot decision boundary for the polynomial kernel
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z_poly = poly.predict(np.c_[xx.ravel(), yy.ravel()])
Z_poly = Z_poly.reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, Z_poly, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, edgecolors='k')
plt.xlabel('Capacity')
plt.ylabel('Safety')
plt.title('SVM (Polynomial Kernel)')
plt.legend()
plt.show()