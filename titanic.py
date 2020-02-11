# accuracy = .74162
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.chdir('/media/niketan/cartush/project/kagal-competetion/titanic')

dataset = pd.read_csv("titanic_train.csv")

# removing nan(Age) values from dataset
dataset['Age'].fillna(dataset['Age'].median(), inplace=True)

# data exploration and visualisation
dataset.describe()
dataset.head()

dataset['Died'] = 1 - dataset['Survived']
dataset.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7),
                                                             stacked=True, color=['g', 'r']);
# female survived more than man there is a corelation between Survived and sex

plt.bar(dataset['Survived'], height=dataset['Survived'].aggregate('sum'), width=0.2);

fig = plt.figure(figsize=(19, 6))
sns.violinplot(x='Sex', y='Age',
               hue='Survived', data=dataset,
               split=True,
               palette={0: "r", 1: "g"}
               );

plt.hist2d(x='Sex', y='Age', data=dataset);  # im not good at it right now


# insites of data

# data preprocessing and preparing for training

def status(feature):
    print('Processing', feature, ':ok-doki')


def combined_data():
    # combing test train for preprocessing
    train = pd.read_csv("titanic_train.csv")
    test = pd.read_csv('titanic_test.csv')

    survived = train.Survived
    train.drop(columns=['Survived'], inplace=True)

    join = train.append(test)
    join.reset_index(inplace=True)
    join.drop(columns=['index', 'PassengerId'], inplace=True)

    return join, survived


joined, y = combined_data()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

print(joined.iloc[:891].Age.isnull().sum())
print(joined.iloc[891:].Age.isnull().sum())


def process_age():
    global joined
    joined['Age'].fillna(joined['Age'].median(), inplace=True)
    # removing blank spaces with median
    status('Age')
    return joined


joined = process_age()

print(joined.iloc[:891].SibSp.isnull().sum())


def process_sibsp():
    status('Sibsp')


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

print(joined['Embarked'].isnull().sum())


def process_embarked():
    global joined
    joined['Embarked'].fillna('S', inplace=True)
    embarked_dumm = pd.get_dummies(joined.Embarked, prefix='Embarked')
    joined = pd.concat([joined, embarked_dumm], axis=1)
    joined.drop('Embarked', axis=1, inplace=True)
    status("Embarked")
    return joined


joined = process_embarked()
print(joined['Sex'].isnull().sum())
from sklearn.preprocessing import LabelEncoder
def process_sex():
    global joined
    le = LabelEncoder()
    joined.Sex = le.fit_transform(joined.Sex)
    status("Sex")
    return joined

joined = process_sex()

joined.Parch.isnull().sum()
def process_family():
    global joined
    # introducing a new feature : the size of families (including the passenger)
    joined['FamilySize'] = joined['Parch'] + joined['SibSp'] + 1

    # introducing other features based on the family size
    joined['Singleton'] = joined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    joined['SmallFamily'] = joined['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    joined['LargeFamily'] = joined['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

    status('family')
    return joined


joined = process_family()
print(joined.shape)

def final_data():
    global joined
    joined.drop(columns=['Name', 'Ticket', 'Cabin', 'Fare'], inplace=True)
    status('final_data')
    return joined


joined = final_data()
print(joined.shape)
joined.isnull().any()
# choosing the model and traing the model

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
lr = LinearRegression()
logr = LogisticRegression()
sc = StandardScaler()
rfc = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=0)


## creating train test
def train_test():
    global joined
    global y
    train = joined.iloc[:891]
    test = joined.iloc[891:]
    yin = y
    return train, test, yin


train, test, y = train_test()

# model1
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(train, y, test_size=0.1, random_state=0)
print(xtrain.shape)
print(xtest.shape)

logr = logr.fit(xtrain, ytrain)
ypred = logr.predict(xtest)
ypred = ypred.reshape(90,1)

####model2
xtrain = sc.fit_transform(xtrain)
xtest = sc.fit_transform(xtest)
rfc.fit(xtrain, ytrain)
ypred2 = rfc.predict(xtest)
cm =confusion_matrix(ytest, ypred2)

train = sc.fit_transform(train)
test = sc.fit_transform(test)
rfc1 = RandomForestClassifier(n_estimators=50, criterion="entropy", random_state=0)
rfc1.fit(train, y)
final_prediction = rfc1.predict(test)
print(final_prediction.shape)
final_prediction = final_prediction.reshape(418, 1)

from sklearn.model_selection import cross_val_score
score = cross_val_score(rfc, train, y, cv=5, scoring='accuracy')

# comparing different models on accuracy confussion metrix


# creating submission csv
submission_df = pd.DataFrame()
a = pd.read_csv("titanic_test.csv")
submission_df['PassengerId'] = a['PassengerId']
submission_df['Survived'] = final_prediction
submission_df[['PassengerId','Survived']].to_csv('./final_sub.csv', index=False)
######## accuracy .74162
