#There are commented print instruction in each step in case you can to check the changes as they happen
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from google.colab import drive

drive.mount("/content/gdrive")  
!pwd
%cd "/content/gdrive/My Drive/8VO SEMESTRE/Sistemas inteligentes/PROJECT"
!ls
df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
print(df.head())

df['NObeyesdad'].value_counts().plot.bar()
plt.xlabel("Class")
plt.ylabel("People count")
plt.title("""Levels of obesity
Original data frame""")

plt.figure(figsize=(20,6))

plt.subplot(131)
plt.scatter(df['Age'],df['Weight'],  color='blue')
plt.xlabel("Age")
plt.ylabel("Weight")

plt.subplot(132)
plt.scatter(df['Height'],df['Weight'],  color='Orange')
plt.xlabel("Height")
plt.ylabel("Weight")

df_y = df[['NObeyesdad']]
print(df_y.head())

df_bool=df[['Gender','family_history_with_overweight','FAVC','SMOKE','SCC']]
#print(df_bool.head())

df_xstr=df[['CAEC','CALC','MTRANS']]
#print(df_xstr.head())

df_xnum=df[['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE']]
#print(df_xnum.head())

#Label encoder
#print(df_xstr)
le = preprocessing.LabelEncoder()
df_xEn = df_xstr.apply(le.fit_transform)
#print(df_xEn.head())
#type(df_xEn)

#OneHotEncoder
#print(df_bool)
enc = preprocessing.OneHotEncoder()
enc.fit(df_bool)
df_xOHE = enc.transform(df_bool).toarray()
df_xOHE.shape
df_xOHE

array_x=np.concatenate((df_xnum,df_xEn,df_xOHE),axis=1)
#print(array_x)


df_x = pd.DataFrame(array_x)
df_x.columns=['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE','CAEC','CALC','MTRANS','Female','Male','Overweight history_No','Overweight history_Yes','FAVC_No','FAVC_Yes','SMOKE_No','SMOKE_Yes','SCC_No','SCC_Yes']
print(df_x)

from sklearn.model_selection import train_test_split
dfx_train,dfx_test,dfy_train,dfy_test=train_test_split(df_x,df_y,test_size=0.1)
print("""Train data set
""")
print(dfx_train.head())
print(dfy_train.head())
print("""
Test data set
""")
print(dfx_test.head())
print(dfy_test.head())

#Tree classifier training - hyper parameter max depth
tree_clf = DecisionTreeClassifier(max_depth = 10)
tree_clf.fit(dfx_train,dfy_train)

print("tree classifier configuration")
print (tree_clf)

#Predictoin results vs original data
print("")
print("Prediction for test data")
Test_predict = pd.DataFrame(tree_clf.predict(dfx_test))
print(Test_predict)
print("")
print("Real class for test data")
print(dfy_test)

print("Accuracy:",accuracy_score(dfy_test, Test_predict))

#Plots for results 
plt.figure(figsize=(15, 3))
plt.subplot(131)
Test_predict.value_counts().plot.bar(color='Orange')
plt.xlabel("Class")
plt.ylabel("People count")
plt.title("""Prediction results 
Count of people per class """)
plt.subplot(132)
dfy_test.value_counts().plot.bar()
plt.xlabel("Class")
plt.ylabel("People count")
plt.title("""Original Test Data 
Count of people per class """)

Val1=Test_predict.value_counts()
Val2=dfy_test.value_counts()
print("""
Prediction Results""")
print(Val1)
print("""
Original Test Data""")
print(Val2)

#Interface to enter new instances
Array_Ans=[None]*11
Array_Ans[0] = int (input('Age:'))
Array_Ans[1] = float (input('Height in meters:'))
Array_Ans[2] = int (input('Weight in kg:'))

print("Never=3,Sometimes=2,Alwas=1")
Array_Ans[3] = int (input('Do you usually eat vegetables in ypur meals?'))

print("One to two=3,  Three=2  , More than three=1")
Array_Ans[4] = int (input('How many main meals do you have daily?'))

print("Less than a liter=3,  One or two liters=2 , More than two liters=1")
Array_Ans[5] = int (input('How much water do you drink daily?'))

print("0 to 1 day= 3,  1 or to days=2,  2 or 4 days=1,  More than 4: 0")
Array_Ans[6] = int (input('How often do you excercise?'))

print("0 to 2 hours=2,  3 to 5 hours=1,  More than 5: 0")
Array_Ans[7] = int (input('Hours spent using electronic devices?'))

print("no=3,Sometimes=2,Frequently=1,Always=0")
Array_Ans[8] = int(input('Do you eat between meals?'))

print("Never=3,Sometimes=2,Frequently=1,Always=0")
Array_Ans[9] = int (input('How often do you dink alcohol?'))

print("Automobile=0,Motorbike=1,Bike=2,Public_Transportation=3,Walking=4")
Array_Ans[10] = int (input('From the above, choose your transportation method:'))
print(Array_Ans)

Ans_str=[[None]*5]
Ans_str[0][0] = str (input('Gender(female,male):'))
Ans_str[0][1] = str (input('Are there any cases of obesity in your family?(yes/no):'))
Ans_str[0][2] = str (input('Do you eat high caloric food frequently?(yes/no):'))
Ans_str[0][3] = str (input('Do you smoke?(yes/no):'))
Ans_str[0][4] = str (input('Do you monitor your calories?(yes/no):'))
print(Ans_str)
#Necesito hacer one hot encoding para en array de respuestas
Array_OHE=[None]*10
for i in range (5):
  if i ==0:
    if Ans_str[0][i]== 'female':
      Array_OHE[0]=1
      Array_OHE[1]=0
    else:
      Array_OHE[0]=0
      Array_OHE[1]=1
  else:
    if  Ans_str[0][i]== 'no':
      Array_OHE[i*2]=1
      Array_OHE[(i*2)+1]=0
    else:
      Array_OHE[i*2]=0
      Array_OHE[(i*2)+1]=1
      
print(Array_OHE)

UserAns=np.concatenate((Array_Ans,Array_OHE))
print(UserAns)


#user's prediction

print("Prediction for new user")
probs=pd.DataFrame(tree_clf.predict_proba([UserAns]))
probs = tree_clf.predict_proba([UserAns])
print("probability of class for user is",probs)

pred =  tree_clf.predict([UserAns])
print("""
The user's weight was classified into the next class: 
""",pred)




