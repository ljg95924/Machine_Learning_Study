# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 15:03:07 2018

@author: ljg
"""
import mglearn
import matplotlib.pyplot as plt

X,y=mglearn.datasets.make_forge()
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.legend(['Class 0','Class 1'],loc=4)
plt.xlabel('First feature')
plt.ylabel('Second feature')
print('X.shape: {}'.format(X.shape))

X,y=mglearn.datasets.make_wave(n_samples=40)
plt.plot(X,y,'o')
plt.ylim(-3,3)
plt.xlabel('feature')
plt.ylabel('Target')

from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
print('cancer.keys():\n{}'.format(cancer.keys()))

print('Shape of cancer data: {}'.format(cancer.data.shape))

print('Sample counts per class:\n{}'.format({n: v for n,v in zip(cancer.target_names,np.bincount(cancer.target))}))
#cancer.target_names
#np.bincount(cancer.target)
#cancer.target
print('Feature names:\n{}'.format(cancer.feature_names))

from sklearn.datasets import load_boston
boston=load_boston()
print('Data shape: {}'.format(boston.data.shape))

X,y = mglearn.datasets.load_extended_boston()
print('X.shape: {}'.format(X.shape))

mglearn.plots.plot_knn_classification(n_neighbors=1)

mglearn.plots.plot_knn_classification(n_neighbors=3)

from sklearn.model_selection import train_test_split
X,y =mglearn.datasets.make_forge()
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=3)

clf.fit(X_train,y_train)

print('Test set predictions:{}'.format(clf.predict(X_test)))

print('Test set accuracy: {:.2f}'.format(clf.score(X_test,y_test)))

fig,axes=plt.subplots(1,3,figsize=(10,3))

for n_neighbors,ax in zip([1,3,9],axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
    mglearn.plots.plot_2d_separator(clf,X,fill=True,eps=0.5,ax=ax,alpha=.4)
    mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)
    ax.set_title('{} neighbors(s)'.format(n_neighbors))
    ax.set_xlabel('feature 0')
    ax.set_ylabel('feature 1')
axes[0].legend(loc=3)

#In[18]:
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(
        cancer.data,cancer.target,stratify=cancer.target,random_state=66)

training_accuracy=[]
test_accuracy=[]
neighbors_settings=range(1,11)

for n_neighbors in neighbors_settings:
    clf=KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train,y_train)
    training_accuracy.append(clf.score(X_train,y_train))
    test_accuracy.append(clf.score(X_test,y_test))
    
plt.plot(neighbors_settings,training_accuracy,label='training accurary')
plt.plot(neighbors_settings,test_accuracy,label='test accuracy')
plt.ylabel('Accuracy')
plt.xlabel('n_neighbors')
plt.legend()

#In[19]:
mglearn.plots.plot_knn_regression(n_neighbors=1)

#In[20]
mglearn.plots.plot_knn_regression(n_neighbors=3)

#In[21]
from sklearn.neighbors import KNeighborsRegressor

X,y=mglearn.datasets.make_wave(n_samples=40)

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
reg=KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train,y_train)

#In[22]
print('Test set predictions:\n{}'.format(reg.predict(X_test)))

#In[23]
print('Test set R^2: {:.2f}'.format(reg.score(X_test,y_test)))

#In[24]
fig,axes=plt.subplots(1,3,figsize=(15,4))

line=np.linspace(-3,3,1000).reshape(-1,1)
for n_neighbors,ax in zip([1,3,9],axes):
    reg=KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train,y_train)
    ax.plot(line,reg.predict(line))
    ax.plot(X_train,y_train,'^',c=mglearn.cm2(0),markersize=8)
    ax.plot(X_test,y_test,'v',c=mglearn.cm2(1),markersize=8)
    
    ax.set_title(
            '{} neighbors(s)\n train score: {:.2f} test score: {:.2f}'.format(
                    n_neighbors, reg.score(X_train,y_train),
                    reg.score(X_test,y_test)))
    ax.set_xlabel('Feature')
    ax.set_ylabel('Target')
axes[0].legend(['Model predictions','Training data/target',
    'Test data/target'],loc='best')
    
#In[51]:
logreg = LogisticRegression().fit(X_train, y_train)
#In[54]:
X = np.array([[0, 1, 0, 1],
 [1, 0, 1, 1],
 [0, 0, 0, 1],
 [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])
#In[55]:
counts = {}
for label in np.unique(y):
 # iterate over each class
 # count (sum) entries of 1 per feature
 counts[label] = X[y == label].sum(axis=0)
print("Feature counts:\n{}".format(counts))
#In[56]:
mglearn.plots.plot_animal_tree()    
