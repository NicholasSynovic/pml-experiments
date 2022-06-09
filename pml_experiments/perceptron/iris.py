from sklearn import datasets
from pandas import DataFrame
from sklearn.utils._bunch import Bunch
from pandas import Series
from numpy import ndarray, stack
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

def loadIrisDataset()   ->  DataFrame:
    iris: Bunch = datasets.load_iris(as_frame=True)
    data: DataFrame = iris.data
    target: Series = iris.target
    data["target"] = target
    return data

def dropTarget(data: DataFrame, target: int) ->  DataFrame:
    return data[data["target"] != target].reset_index(drop=True)

def splitData(data: DataFrame, attribute1_Index: int, attribut2_Index: int, testSize: float = 0.3)    ->  tuple[ndarray, ndarray, ndarray, ndarray]:
    a1: ndarray = data[data.columns[attribute1_Index]].to_numpy()
    a2: ndarray = data[data.columns[attribut2_Index]].to_numpy()

    x: ndarray = stack((a1, a2), axis=1)
    y: ndarray = data["target"].to_numpy()

    xTrain: ndarray
    xTest: ndarray
    yTrain: ndarray
    yTest: ndarray
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=testSize, random_state=1, stratify=y)

    return (xTrain, xTest, yTrain, yTest)

def transformData(xTrain: ndarray, xTest:ndarray)   ->  tuple[ndarray, ndarray]:
    sc: StandardScaler = StandardScaler()
    sc.fit(X=xTrain)

    transformedTrain: ndarray = sc.transform(X=xTrain)
    transformedTest: ndarray = sc.transform(X=xTest)

    return (transformedTrain, transformedTest)

def main()  ->  None:
    df: DataFrame = loadIrisDataset()
    df = dropTarget(data = df, target = 0)
    dataSets: tuple = splitData(data=df, attribute1_Index=0, attribut2_Index=1)
    transformedData: tuple = transformData(xTrain=dataSets[0], xTest=dataSets[1])
    ppn: Perceptron = Perceptron(eta0=0.1, random_state=1)
    ppn.fit(X=transformedData[0], y=dataSets[2])
    ppn: ndarray = ppn.predict(X=transformedData[1])

    print(f"Accuracy: {accuracy_score(y_true=dataSets[3], y_pred=ppn) * 100}%")

if __name__ == "__main__":
    main()
