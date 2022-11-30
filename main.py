#Import Libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

# Using GridDB     (Loading Dataset)
# import griddb_python as grid
# Using Pandas
heart_dataset = pd.read_csv('heart.csv')

# Exploratory Data Analysis
heart_dataset.shape
heart_dataset.head()
heart_dataset.dtypes
heart_dataset.isna().sum()

categorical_cols= heart_dataset.select_dtypes(include=['object'])

categorical_cols.columns

for cols in categorical_cols.columns:
    print(cols,'-', len(categorical_cols[cols].unique()),'Labels')
    
train, test = train_test_split(heart_dataset,test_size=0.3,random_state= 1234)

labels = [x for x in train.ChestPainType.value_counts().index]
values = train.ChestPainType.value_counts()

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.update_layout(
    title_text="Distribution of data by Chest Pain Type (in %)")
fig.update_traces()
fig.show()

fig=px.histogram(heart_dataset, 
                 x="HeartDisease",
                 color="Sex",
                 hover_data=heart_dataset.columns,
                 title="Distribution of Heart Diseases by Gender",
                 barmode="group")
fig.show()

#Handling categorical variables
train['Sex'] = np.where(train['Sex'] == "M", 0, 1)
train['ExerciseAngina'] = np.where(train['ExerciseAngina'] == "N", 0, 1)
test['Sex'] = np.where(test['Sex'] == "M", 0, 1)
test['ExerciseAngina'] = np.where(test['ExerciseAngina'] == "N", 0, 1)

train.head()

train=pd.get_dummies(train)
test=pd.get_dummies(test)

train.head()

test.head()

train.shape

test.shape

x_train=train.drop(['HeartDisease'],1)
x_test=test.drop(['HeartDisease'],1)

y_train=train['HeartDisease']
y_test=test['HeartDisease']


print(x_train.shape)
print(x_test.shape)

# Machine Learning Model
lr = LogisticRegression(max_iter=10000)
model1=lr.fit(x_train, y_train)

print("Train accuracy:",model1.score(x_train, y_train))

print("Test accuracy:",model1.score(x_test,y_test))

lrpred = lr.predict(x_test)

print(classification_report(lrpred,y_test))

displr = plot_confusion_matrix(lr, x_test, y_test,cmap=plt.cm.OrRd , values_format='d')