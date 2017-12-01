import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

names=pd.read_csv('/Users/naduong1001/Desktop/name.csv')
names = names.as_matrix()[:,1:]

TRAIN_SPLIT = 0.7

def features(name):
    name = name.lower()
    return {
        'first-letter': name[:6],
        'firstTwo-letters': name[:7],
        'firstThree-letters': name[:8],
        'last-letter': name[-5:],
        'lastTwo-letters': name[-4:],
        'lastThree-letters': name[-3:],
    }



features = np.vectorize(features)
Name = features(names[:, 0])
Gender = names[:, 2]


Name_Train, Name_Test, Gender_Train, Gender_Test = train_test_split(Name, Gender, test_size = 0.3)

vectorizer = DictVectorizer()
vectorizer.fit(Name_Train)

clf = DecisionTreeClassifier()
clf.fit(vectorizer.transform(Name_Train), Gender_Train)

Gender_pred = clf.predict(vectorizer.transform(Name_Test))



print (clf.predict(vectorizer.transform(features(["Nguyễn Ánh Dương", "Vũ Tiến Đạt", "Ngô Văn Vĩ", "Phạm Ngọc Hà", "Hoàng Mai Hương"]))))
from sklearn.metrics import accuracy_score
print('Accuracy = ',accuracy_score(Gender_Test, Gender_pred))

from sklearn.model_selection import cross_val_score
crossValScore = cross_val_score(clf, vectorizer.transform(Name), Gender, cv = 10)

print('Distribution of cross-validation scores: %.2f (+/-) %.2f' %(crossValScore.mean(), crossValScore.std()*2))
