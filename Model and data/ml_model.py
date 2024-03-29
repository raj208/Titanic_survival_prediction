import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
train = pd.read_csv('titanic_train.csv')

# Preprocessing data (cleaning data)
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 29
        else:
            return 25
    else:
        return Age

train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
train.drop('Cabin', axis=1, inplace=True)

# Convert categorical variables into dummy variables
sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)

# Drop unnecessary columns
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)

# Concatenate dummy variables
train = pd.concat([train, sex, embark], axis=1)

# Logistic Regression Model
x_train, x_test, y_train, y_test = train_test_split(train.drop('Survived', axis=1), train['Survived'], test_size=0.30, random_state=101)
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)
predictions = logmodel.predict(x_test)
sc = MinMaxScaler(feature_range=(0, 1))

# Evaluation
# print(confusion_matrix(y_test, predictions))
# print(classification_report(y_test, predictions))
pickle.dump(logmodel, open("ml_model.sav", "wb"))
pickle.dump(sc, open("scaler.sav", "wb"))