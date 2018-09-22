#libs
import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

#classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb

#for voting
from sklearn.ensemble import VotingClassifier

#importing data
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:,1:56]
y = np.array(dataset.iloc[:,55:56])

#all categorical variables
le = LabelEncoder()
X = X.apply(le.fit_transform)

enc = OneHotEncoder()
enc.fit(X)
X = enc.transform(X).toarray()

#Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

#splitting into test/train sets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)

#training data
clf1 = DecisionTreeClassifier()
clf2 = AdaBoostClassifier(n_estimators=500)
clf3 = xgb.XGBClassifier(n_estimators=500, nthread=-1)
clf4 = GradientBoostingClassifier(n_estimators=500)
clf5 = RandomForestClassifier(n_estimators=10)
clf6 = ExtraTreesClassifier()
clf7 = SVC()
clf8 = GaussianNB()
clf9 = KNeighborsClassifier(n_neighbors=2)


#Fitting the models
clf1 = clf1.fit(X_train,y_train.ravel())
clf2 = clf2.fit(X_train,y_train.ravel())
clf3 = clf3.fit(X_train,y_train.ravel())
clf4 = clf4.fit(X_train,y_train.ravel())
clf5 = clf5.fit(X_train,y_train.ravel())
clf6 = clf6.fit(X_train,y_train.ravel())
clf7 = clf7.fit(X_train,y_train.ravel())
clf8 = clf8.fit(X_train,y_train.ravel())
clf9 = clf9.fit(X_train,y_train.ravel())


#predicting
y_pred1 = clf1.predict(X_test)
print('The accuracy of DecisionTree is ' + str(accuracy_score(y_test,y_pred1)))

y_pred2 = clf2.predict(X_test)
print('The accuracy of AdaBoost is ' + str(accuracy_score(y_test,y_pred2)))

y_pred3 = clf3.predict(X_test)
print('The accuracy of XGBoost is ' + str(accuracy_score(y_test,y_pred3)))


y_pred4 = clf4.predict(X_test)
print('The accuracy of GradientBoosting is ' + str(accuracy_score(y_test,y_pred4)))

y_pred5 = clf5.predict(X_test)
print('The accuracy of RandomForest is ' + str(accuracy_score(y_test,y_pred5)))

y_pred6 = clf6.predict(X_test)
print('The accuracy of ExtraTrees is ' + str(accuracy_score(y_test,y_pred6)))

y_pred7 = clf7.predict(X_test)
print('The accuracy of Support Vector Machine is ' + str(accuracy_score(y_test,y_pred7)))

y_pred8 = clf8.predict(X_test)
print('The accuracy of Naive Bayes is ' + str(accuracy_score(y_test,y_pred8)))

y_pred9 = clf9.predict(X_test)
print('The accuracy of KNeighbors is ' + str(accuracy_score(y_test,y_pred9)))

confusion1 = confusion_matrix(y_test,y_pred1)
confusion2 = confusion_matrix(y_test,y_pred2)
confusion3 = confusion_matrix(y_test,y_pred3)
confusion4 = confusion_matrix(y_test,y_pred4)
confusion5 = confusion_matrix(y_test,y_pred5)
confusion6 = confusion_matrix(y_test,y_pred6)
confusion7 = confusion_matrix(y_test,y_pred7)
confusion8 = confusion_matrix(y_test,y_pred8)
confusion9 = confusion_matrix(y_test,y_pred9)


vclf = VotingClassifier(estimators=[('decision',clf1),('nb',clf2),('xgb',clf3),('kn',clf4),('f',clf5),('qq',clf6),('v',clf7),('qqqq',clf8),('sds',clf9)],weights=[1,1,1,1,1,1,1,1,1])
vclf = vclf.fit(X_train,y_train.ravel())

#predicting
y_pred = vclf.predict(X_test)

#confusion matrix
confusion = confusion_matrix(y_test,y_pred)

#test data
data = pd.read_csv('test.csv')
perid = data.id
x_test = data.iloc[:,1:16]
x_test = x_test.apply(le.fit_transform)
enc = OneHotEncoder()
enc.fit(x_test)
x_test = enc.transform(x_test).toarray()
#Feature scaling
x_test = scaler.fit_transform(x_test)

predictions = vclf.predict(x_test)

#submission file creation
submission = pd.DataFrame({'id':perid.tolist(),'P':predictions.tolist()})
columnsTitles=["id","P"]
submission=submission.reindex(columns=columnsTitles)
submission.to_csv('tp.csv',index=False)