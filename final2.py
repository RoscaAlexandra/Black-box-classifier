import numpy as np
import csv
import pandas

#getting training data
train=[]
train_labels=[]
train=pandas.read_csv('train_samples.csv',header=None)
train_labels=pandas.read_csv('train_labels.csv',header=None)
train=np.array(train)
train_labels=np.array(train_labels)  
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
#classifier-tensorflow keras
#trying to train the classifier with the first and second fold
import tensorflow as tf
classifier1= tf.keras.models.Sequential()
classifier1.add(tf.keras.layers.Flatten())
# 3 hidden layer
classifier1.add(tf.keras.layers.Dense(90,activation=tf.nn.relu))
classifier1.add(tf.keras.layers.Dense(40,activation=tf.nn.relu))
classifier1.add(tf.keras.layers.Dense(15,activation=tf.nn.relu))
classifier1.add(tf.keras.layers.Dense(8,activation=tf.nn.softmax))
classifier1.compile(optimizer='adam', #learning rate 0.001
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
classifier1.fit(x_1,y_1,epochs=5)
classifier1.fit(x_2,y_2,epochs=5)

#prediction fold 3
prediction1_fold_3=classifier1.predict(x_3)
final_prediction=[]
for p in prediction1_fold_3:
    final_prediction.append(np.argmax(p))
prediction1_fold_3=final_prediction
#accuracy score
from sklearn.metrics import accuracy_score
#fold3
print('accuracy_score_fold_3=')
acc1_fold3=accuracy_score(prediction1_fold_3,y_3)
print(acc1_fold3)

#confusion matrix 
from sklearn.metrics import confusion_matrix

print('confusion_matrix_3:')
print(confusion_matrix(y_3,prediction1_fold_3))

#%%
#classifier-tensorflow keras
#trying to train the classifier with the second and third fold
import tensorflow as tf
classifier2= tf.keras.models.Sequential()
classifier2.add(tf.keras.layers.Flatten())
# 3 hidden layer
classifier2.add(tf.keras.layers.Dense(90,activation=tf.nn.relu))
classifier2.add(tf.keras.layers.Dense(40,activation=tf.nn.relu))
classifier2.add(tf.keras.layers.Dense(15,activation=tf.nn.relu))
classifier2.add(tf.keras.layers.Dense(8,activation=tf.nn.softmax))
classifier2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
classifier2.fit(x_2,y_2,epochs=5)
classifier2.fit(x_3,y_3,epochs=5)

#predicting fold 1
prediction2_fold_1=classifier2.predict(x_1)
final_prediction=[]
for p in prediction2_fold_1:
    final_prediction.append(np.argmax(p))
prediction2_fold_1=final_prediction

#accuracy score
from sklearn.metrics import accuracy_score
#fold1
print('accuracy_score_fold_1=')
acc2_fold1=accuracy_score(prediction2_fold_1,y_1)
print(acc2_fold1)

#confusion matrix fold 1
from sklearn.metrics import confusion_matrix
print('confusion_matrix_1:')
print(confusion_matrix(y_1,prediction2_fold_1))


#%%
#classifier-tensorflow keras
#trying to train the classifier with the first and third fold
import tensorflow as tf
classifier3= tf.keras.models.Sequential()
classifier3.add(tf.keras.layers.Flatten())
# 3 hidden layer
classifier3.add(tf.keras.layers.Dense(90,activation=tf.nn.relu))
classifier3.add(tf.keras.layers.Dense(40,activation=tf.nn.relu))
classifier3.add(tf.keras.layers.Dense(15,activation=tf.nn.relu))
classifier3.add(tf.keras.layers.Dense(8,activation=tf.nn.softmax))
classifier3.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
classifier3.fit(x_3,y_3,epochs=5)
classifier3.fit(x_1,y_1,epochs=5)

#prediction fold 2
prediction3_fold_2=classifier3.predict(x_2)
final_prediction=[]
for p in prediction3_fold_2:
    final_prediction.append(np.argmax(p))
prediction3_fold_2=final_prediction

#accuracy score
from sklearn.metrics import accuracy_score

#fold2
print('accuracy_score_fold_2=')
acc3_fold2=accuracy_score(prediction3_fold_2,y_2)
print(acc3_fold2)

#confusion matrix fold 2
from sklearn.metrics import confusion_matrix
print('confusion_matrix_2:')
print(confusion_matrix(y_2,prediction3_fold_2))

#%%
#fitting all the data from train_samples
train=Std.transform(train)
import tensorflow as tf
final_classifier= tf.keras.models.Sequential()
final_classifier.add(tf.keras.layers.Flatten())
# 3 hidden layer
final_classifier.add(tf.keras.layers.Dense(90,activation=tf.nn.relu))
final_classifier.add(tf.keras.layers.Dense(40,activation=tf.nn.relu))
final_classifier.add(tf.keras.layers.Dense(15,activation=tf.nn.relu))
final_classifier.add(tf.keras.layers.Dense(8,activation=tf.nn.softmax))
final_classifier.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
final_classifier.fit(train,train_labels,epochs=5)

#reading all data from test_samples
test=pandas.read_csv('test_samples.csv',header=None)
test=Std.transform(test)
prediction=final_classifier.predict(test)
final_prediction=[]
for p in prediction:
    final_prediction.append(np.argmax(p))
prediction=final_prediction

#writing final csv for submitting
with open('result.csv','w',newline="") as res:
     field=['Id','Prediction']
     csv_writer=csv.DictWriter(res, fieldnames = field)
     csv_writer.writeheader()
     i=0
     for j in range(0,5000):
           csv_writer.writerow({'Id':i+1,'Prediction':int(prediction[i])})
           i=i+1
