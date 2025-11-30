# importing dependencies
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



# load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
dataset = pd.read_csv(url, names=names)

#dimensions of the data
print (dataset.shape)

#peek at the data
print(dataset.head(20))

#statistical summary
print(dataset.describe())

#class distribution
print(dataset.groupby('class').size())

#univariant plots - box and whisker plots
dataset.plot(kind='box', subplots=True,layout=(2,2),sharex=False,sharey=False)
plt.show()

#histogram of the variable
dataset.hist()
plt.show()

#multivariate plots
scatter_matrix(dataset)
plt.show()

#creating a validation dataset
#splitting dataset
array = dataset.values
X=array[:,0:4]
Y=array[:, 4]
X_train, X_validation,Y_train, Y_validation=train_test_split(X,Y,test_size=0.2,random_state=1)

#Logistic Regression
#linear Discriminant Analysis
#K-Nearest neighbours
#Classification and Regression Trees
#Gaussian Naive Bayes
#Support Vector Machines 

#building models
models=[]
models.append(('LR', LogisticRegression(solver='lbfgs', max_iter=1000)))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

#evaluate the created models
results =[]
names=[]
for name,model in models:
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s:%f(%f)'% (name,cv_results.mean(), cv_results.std()))

#compare our models
plt.boxplot(results, tick_labels=names)
plt.title("Algorithm Comparison")
plt.show()

#make predictions on svm
model = SVC(gamma='auto')
model.fit(X_train,Y_train)
predictions = model.predict(X_validation)

#evaluate our predictions
print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))


