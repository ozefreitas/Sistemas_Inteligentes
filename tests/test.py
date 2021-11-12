import numpy as np


# dataset processing

from src.si.data.dataset import Dataset, summary
dataseteste = Dataset.from_data("C:/Users/Ze/Desktop/Mestrado/3ºSemestre/si/datasets/breast-bin.data")
print(dataseteste.Y)
print(type(dataseteste.X))
# teste
print(dataseteste.X - 1)

print(summary(dataseteste))


# dataset standardization

from src.si.util.scale import StandardScaler
scaler = StandardScaler()
scaler.fit(dataseteste)
scaler.transform(dataseteste)
np.sqrt(scaler.var)
datascaled = scaler.fit_transform(dataseteste)
summary(datascaled)

# dataset feature selection

