# ai_perfect_lecture_os_sw_ML_packages
---
This repo contains two machine learning algorithms
1. polynomial regression
2. Classification with Scikit Learn Library

## I will cover Problem 2 because midterm project is over :D

### Classification with Scikit Learn Library
This will cover classification problem by using Scikit Learn Library

     import sklearn.datasets
     import sklearn.linear_model
     import sklearn.tree
     import sklearn.ensemble
     import sklearn.model_selection
     import sklearn.metrics
     import numpy as np

     import matplotlib.pyplot as plt 
     %matplotlib inline

 This code is to import Scikit Learn Library to solve classification problem
 
     from sklearn import svm

This is my additional package (support vector machine) to classify images

The reason why I used this algorithm is svm can be universally applied at any datasets (even at images!)

In addition, It can be used at high-dimensional imagne dateset. So I chose this algorithm :)

     olivetti_faces = sklearn.datasets.fetch_olivetti_faces(random_state=0,)
     print(olivetti_faces['DESCR'])

     example_indices = [0, 10, 62, 70]
     for idx in example_indices:
         plt.title(olivetti_faces['target'][idx])
         plt.imshow(olivetti_faces['images'][idx])
         plt.gray()
         plt.show()
    
     X = olivetti_faces['data']
     y = olivetti_faces['target']
     X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=0)
     
This part is a process to load data points

This dataset contains a set of face images`_ taken between April 1992 and April 1994 at AT&T Laboratories Cambridge.

**Data Set Characteristics:**

     =================   =====================
     Classes                                40
     Samples total                         400
     Dimensionality                       4096
     Features            real, between 0 and 1
     =================   =====================
 ---

     clf = svm.SVC(kernel='poly', random_state = 0)
     
I created a classification object by using svm algorithm

I set the kernel parameter poly because the image has high dimensionality(4096) 

I set random_state is 0 because to avoid some randomness that causes different test results even if the hyperparameters are not changed

     clf.fit(X_train, y_train)
     
This is code expression that fit the object to training dataset
    
     y_pred = clf.predict(X_test)

This code is to predict the label of test data point

     print('Accuracy: %.2f' % sklearn.metrics.accuracy_score(y_test, y_pred))

This is the last part.. that print accuracy! ( 0.97 )
