import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score , precision_score ,f1_score, recall_score
from sklearn.metrics import confusion_matrix
 
df=pd.read_csv(r"\WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(df.head(5))
print(df.isnull().sum())
print(df.info())
print(df.shape)
print(df.duplicated().sum())
print(df["Churn"].value_counts())
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
df["TotalCharges"]=df["TotalCharges"].fillna(0)

x=df.drop(["Churn","customerID"], axis=1)
y=df["Churn"]
y= y.map({"Yes":1, "No":0})

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, random_state=42, stratify=y)
x_train=pd.get_dummies(x_train) 
x_test=pd.get_dummies(x_test)

scaler=StandardScaler() 
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print('accuracy_lr:' ,accuracy_score(y_test,y_pred)*100)
print('precision:' , precision_score(y_test, y_pred)*100)
print('f1:' , f1_score(y_test,y_pred)*100)
print('recall:', recall_score(y_test,y_pred)*100)
print("cm :" , confusion_matrix(y_test, y_pred))

tree_model=DecisionTreeClassifier(random_state=42)
tree_model.fit(x_train,y_train)
y_pred_tree=tree_model.predict(x_test)
print('decision_tree_accuracy:' ,accuracy_score(y_test,y_pred_tree)*100)
print('precision:' ,precision_score(y_test,y_pred_tree)*100)
print('f1:' , f1_score(y_test,y_pred_tree)*100)
print('recall:' ,recall_score(y_test,y_pred_tree)*100)

rf_model=RandomForestClassifier(random_state=42) 
rf_model.fit(x_train,y_train)
y_pred_rf=rf_model.predict(x_test)
print('rf_accuracy:', accuracy_score(y_test, y_pred_rf)*100)
print('precision:' ,precision_score(y_test,y_pred_rf)*100)
print('f1:' ,f1_score(y_test,y_pred_rf)*100)
print('recall:' , recall_score(y_test,y_pred_rf)*100)
final_outcome = pd.DataFrame({'model':['logistic_regression','decision_tree','random_forest'],
                              'accuracy':[80.69,72.10,78.63],
                              'precision':[65.83,47.54,62.54],
                              'f1':[60.91,48.35,54.73],
                              'recall':[56.68,49.19,48.66]})
print(final_outcome)

#viusal chart of all metrics through pandas
final_outcome = pd.DataFrame({'model':['logistic_regression','decision_tree','random_forest'],
                              'accuracy':[80.69,72.10,78.63],
                              'precision':[65.83,47.54,62.54],
                              'f1':[60.91,48.35,54.73],
                              'recall':[56.68,49.19,48.66]},
                              index=['logistic regression','decision tree','random forest'])

final_outcome.plot(kind='bar',rot=0)
plt.title('model performance comparison')
plt.ylabel('score(%)')
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

#visual chart of accuracy_score through matplotlib
models=['logistic_regression','decision_tree','random_forest']
accuracy=[80.69,72.10,78.63]
precision=[65.83,47.54,62.54]
f1=[60.91,48.35,54.73]
recall=[56.68,49.19,48.66]
plt.figure()
plt.bar( models,accuracy)
plt.title('model accuracy camparison')
plt.ylabel('accuracy (%)')
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

