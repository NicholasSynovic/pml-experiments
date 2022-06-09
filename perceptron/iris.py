from numpy import ndarray
from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils._bunch import Bunch

iris: Bunch = datasets.load_iris()

sepalLengthIndex: int = 0
sepalWidthIndex: int = 1
petalLengthIndex: int = 2
petalWidthIndex: int = 3

xData: ndarray = iris.data[:, [petalLengthIndex, petalWidthIndex]]
yData: ndarray = iris.target

xTrain: ndarray
xTest: ndarray
yTrain: ndarray
yTest: ndarray
xTrain, xTest, yTrain, yTest = train_test_split(
    xData, yData, test_size=0.3, random_state=1, stratify=yData
)

sc: StandardScaler = StandardScaler()
sc.fit(X=xTrain)
transformedXTrain: ndarray = sc.transform(X=xTrain)
transformedXTest: ndarray = sc.transform(X=xTest)

perceptron: Perceptron = Perceptron(eta0=0.1, random_state=1)
perceptron.fit(X=transformedXTrain, y=yTrain)

prediction: ndarray = perceptron.predict(X=transformedXTest)


print(f"Missed examples: {(yTest != prediction).sum()}")
print(f"Accuracy : {accuracy_score(yTest, prediction) * 100}%")
