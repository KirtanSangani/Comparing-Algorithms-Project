import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.metrics import accuracy_score
import graphviz
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("Car Evaluation.csv")
data.head()
data.info()

X = data.drop('Target', axis = 1)
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)

labels = ['0','1','2','3']

y_train.index = [labels[i-1] for i in y_train]
print(y_train.index.value_counts())

X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size = 0.25)

clf = DecisionTreeClassifier(max_depth=7)
cross_val_score(clf,X_train,y_train,cv=7)

clf.fit(X_train,y_train)
print("Pre hyperparameter accuracy: ", metrics.accuracy_score(y_valid,clf.predict(X_valid)))

clf.feature_importances_

plt.figure(figsize=(15,10))
plot_tree(clf, filled = True, feature_names = X_train.columns)
plt.show()
features = list(X)
print(features)
def sortSecond(val):
    return val[1]
values = clf.feature_importances_
features = list(X)
importances = [(features[i], values[i]) for i in range(len(features))]
importances.sort(reverse = True, key=sortSecond)
print(importances)

print('All features:', X_train.memory_usage(index=True).sum()/1000000)
print('Top 6 features:', X_train[[col[0] for col in importances[:6]]].memory_usage(index=True).sum()/1000000)

X_train = X_train[[col[0] for col in importances[:6]]]
X_valid = X_valid[[col[0] for col in importances[:6]]]

cut_clf = DecisionTreeClassifier()
cut_clf.fit(X_train, y_train)
print(accuracy_score(y_valid, cut_clf.predict(X_valid)))

print(sorted_idx)

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth= 4,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, random_state=None,
                       splitter='best')
params_dist= {
    'criterion':['gini','entropy'],
    'max_depth': randint(1,9),
    'max_leaf_nodes': randint(10, 2000),
    'min_samples_leaf': randint(20, 50),
    'min_samples_split': randint(40, 100),
    }
clf_tuned = DecisionTreeClassifier(random_state=1)
random_search = RandomizedSearchCV(clf_tuned, params_dist, cv=7)
random_search.fit(X_train, y_train)
random_search.best_estimator_
best_tuned_clf = random_search.best_estimator_
print("Post hyperparameter accuracy", accuracy_score(y_valid, best_tuned_clf.predict(X_valid)))

print(metrics.classification_report(y_test, cut_clf.predict(X_test[[col[0] for col in importances[:6]]])))
print(metrics.classification_report(y_test, best_tuned_clf.predict(X_test[[col[0] for col in importances[:6]]])))


f, ax = plt.subplots(figsize=(30, 24))
ax=sns.barplot(x=feature_scores, y=X_train.columns)
ax.set_title("Visualize feature scores of the features")
ax.set_yticklabels(feature_scores.index)
ax.set_xlabel("Feature importance score")
ax.set_ylabel("Features")
plt.show()

plt.figure(figsize=(15,10))
plot_tree(clf, filled = True)
plt.show()