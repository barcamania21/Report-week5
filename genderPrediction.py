import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier

names=pd.read_csv('/Users/naduong1001/Desktop/names_dataset.csv')
"""Đọc dữ liệu từ dataset, trả về một DataFrame"""
names = names.as_matrix()[:,1:]
"""Đưa dữ liệu từ DataFrame trên vào một ma trận chỉ chứa hai cột Name và ender"""

TRAIN_SPLIT = 0.8
"""Lấy 80% dữ liệu làm training set và 20% còn lại là test set"""

def features(name):
    name = name.lower()
    return {
        'first-letter': name[0],
        'firstTwo-letters': name[:2],
        'firstThree-letters': name[:3],
        'last-letter': name[-1],
        'lastTwo-letters': name[-2:],
        'lastThree-letters': name[-3:],
    }

"""
Xây dựng hàm features để trích xuất những ký tự đầu và cuối của tên. Những cái tên kết thúc 
(hoặc bắt đầu) bởi một số chữ cái nhất định sẽ đặc trưng cho giới tính của người đó. 
VD: Những người có tên kết thúc bằng "na" thường là phụ nữ.
"""

features = np.vectorize(features)
"""Thư viện numpy cung cấp một phương thức giúp vectorize các function một cách dễ dàng, 
để kết quả trả về là list (array)
"""

Name = features(names[:, 0])
Gender = names[:, 1]

"""
Đẩy toàn bộ mảng dữ liệu chứa tên vào Name, chứa giới tính vào Gender
"""

Name, Gender = shuffle(Name, Gender)
Name_Train, Name_Test = Name[:int(TRAIN_SPLIT * len(Name))], Name[int(TRAIN_SPLIT * len(Name)):]
Gender_Train, Gender_Test = Gender[:int(TRAIN_SPLIT * len(Gender))], Gender[int(TRAIN_SPLIT * len(Gender)):]

"""
Chia dữ liệu thành 2 phần, training set và test set.
"""

vectorizer = DictVectorizer()
"""DictVectorizer map các giá trị thành phần thành các ma trận có các cột là các thuộc tính, 
hàng là các giá trị riêng rẽ chỉ thứ tự
VD: cột đầu tiên (feature_names_[0]) là 'first-letter=a', cột kế tiếp (feature_names_[1] là 'first-letter=b',...
"""
vectorizer.fit(Name_Train)
"""
Đưa dữ liệu từ Name_Train làm tiêu chuẩn đánh giá. 
"""
transformed = vectorizer.transform(features(["Duong", "Vi"]))
print(vectorizer.feature_names_)
print (transformed)
"""
VD:
Một ma trận thưa (sparse matrix) được xuất ra, nhìn vào ma trận ta thấy:
Với tên "Duong", hàng 0. 'first_letter=d', nên transformed.toarray[0, 3] = 1.0,...
Tương tự với "Vi", hàng 1. 'first_letter=v', nên transformed.toarray[0, 21] = 1.0 ,...
"""
clf = DecisionTreeClassifier()
clf.fit(vectorizer.transform(Name_Train), Gender_Train)
"""
Sử dụng mô hình dự báo DecisionTreeClassifer (cây quyết định phân loại) với tiêu chuẩn đánh giá là training set.
"""
print (clf.predict(vectorizer.transform(features(["Roger", "Lionel", "Cristiano", "Andres", "Sergio", "Taylor", "Adele"]))))
"""
Thực hiện dự đoán kết quả từ những thông số biến đầu vào
Thử nghiệm với một số cái tên nổi tiếng :d
"""

"""
Đánh giá độ chính xác của mô hình
"""
Gender_pred = clf.predict(vectorizer.transform(Name_Test))

""" 
Accuracy: độ đo = số dự đoán đúng / tổng số dự đoán
"""

accuracy = accuracy_score(Gender_Test, Gender_pred)
print('Accuracy: ', accuracy)

"""
Cross-validation: 
Chia dữ liệu thành k tập con cùng kích thước (ở đây là 5). 
Tại mỗi vòng lặp sử dụng một tập con là test set, các tập con còn lại là các training set.
"""

crossValScore = cross_val_score(clf, vectorizer.transform(Name), Gender, cv = 5)

print('Mảng giá trị qua mỗi vòng lặp: ',crossValScore)
print('Phân bố: %f (+/- %f)' %(crossValScore.mean(), crossValScore.std() * 2))
