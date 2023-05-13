from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer()

x = data.data
y = data.target 

# split data 80% for training and 20% for testing 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# use the linear kernel to add another feature to the data and use this featre to seperate the data
# 'C' refers to the soft margin
clf = SVC(kernel='linear', C=5)

# train the model using training data
clf.fit(x_train, y_train)

# run model against testing data
print(f'SVC: {clf.score(x_test, y_test)}')



# Implementing a KNN classifier to compare the results
clf2 = KNeighborsClassifier(n_neighbors=5)
clf2.fit(x_train, y_train)
print(f'KNN: {clf2.score(x_test, y_test)}')


# Implementing a decision tree classifier
clf3 = DecisionTreeClassifier()
clf3.fit(x_train, y_train)
print(f'DTC: {clf3.score(x_test, y_test)}')


# Implementing a RandomForestClassifier
clf4 = RandomForestClassifier()
clf4.fit(x_train, y_train)
print(f'RFC: {clf4.score(x_test, y_test)}')