
import csv
import pandas

#getting training data
train=[]
train_labels=[]
train=pandas.read_csv('train_samples.csv',header=None)
train_labels=pandas.read_csv('train_labels.csv',header=None)

#splitting data into 3 folds 
from sklearn.model_selection import train_test_split
#1=>0.4,0.6
x1,x2,y1,y2=train_test_split(train,train_labels,test_size=0.6)
#0.6=>0.3,0.3
x3,x4,y3,y4=train_test_split(x2,y2,test_size=0.5)
#x_1=0.4,x_2=0.3,x_3=0.3
x_1=x1 #first fold
y_1=y1 #first fold labels
x_2=x3 #second fold 
y_2=y3 #second fold labels
x_3=x4 #third fold
y_3=y4 #third fold labels

#normalize data
from sklearn.preprocessing import StandardScaler
Std=StandardScaler().fit(train)
x_1=Std.transform(x_1)
x_2=Std.transform(x_2)
x_3=Std.transform(x_3)

#%%
#classifier-SVM
#trying to train the classifier with the first and second fold
from sklearn import svm
classifier1=svm.SVC(kernel='rbf',C=10.5,gamma='scale')
classifier1=classifier1.fit(x_1,y_1)
classifier1=classifier1.fit(x_2,y_2)

#prediction fold 3
prediction1_fold_3=classifier1.predict(x_3)

#accuracy score
from sklearn.metrics import accuracy_score
print('accuracy_score_fold_3=')
acc1_fold3=accuracy_score(prediction1_fold_3,y_3)
print(acc1_fold3)

#confusion matrix for both folds tested
from sklearn.metrics import confusion_matrix
print('confusion_matrix_3:')
print(confusion_matrix(y_3,prediction1_fold_3))

#%%
#classifier-SVM
#trying to train the classifier with the second and third fold
from sklearn import svm
classifier2=svm.SVC(kernel='rbf',C=10.5,gamma='scale')
classifier2=classifier2.fit(x_2,y_2)
classifier2=classifier2.fit(x_3,y_3)

#predicting fold 1
prediction2_fold_1=classifier2.predict(x_1)


#accuracy score
from sklearn.metrics import accuracy_score
#fold2
print('accuracy_score_fold_1=')
acc2_fold1=accuracy_score(prediction2_fold_1,y_1)
print(acc2_fold1)


#confusion matrix fold 1
from sklearn.metrics import confusion_matrix
print('confusion_matrix_1:')
print(confusion_matrix(y_1,prediction2_fold_1))


#%%
#classifier-SVM
#trying to train the classifier with the first and third fold
from sklearn import svm
classifier3=svm.SVC(kernel='rbf',C=10.5,gamma='scale')
classifier3=classifier3.fit(x_3,y_3)
classifier3=classifier3.fit(x_1,y_1)

#prediction fold 2
prediction3_fold_2=classifier3.predict(x_2)

#accuracy score
from sklearn.metrics import accuracy_score
#fold2
print('accuracy_score_fold_2=')
acc3_fold2=accuracy_score(prediction3_fold_2,y_2)
print(acc3_fold2)

#confusion matrix for both folds tested
from sklearn.metrics import confusion_matrix
print('confusion_matrix_2:')
print(confusion_matrix(y_2,prediction3_fold_2))

#%%
#fitting all the data from train_samples
train=Std.transform(train)
from sklearn import svm
final_classifier=svm.SVC(kernel='rbf',C=10.5,gamma='scale')
final_classifier=final_classifier.fit(train, train_labels)

#reading all data from test_samples
test=pandas.read_csv('test_samples.csv',header=None)
test=Std.transform(test)
prediction=final_classifier.predict(test)

#writing final csv for submitting
with open('result.csv','w',newline="") as res:
     field=['Id','Prediction']
     csv_writer=csv.DictWriter(res, fieldnames = field)
     csv_writer.writeheader()
     i=0
     for j in range(0,5000):
           csv_writer.writerow({'Id':i+1,'Prediction':int(prediction[i])})
           i=i+1
