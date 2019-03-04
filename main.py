from sklearn.datasets import load_boston, load_iris, load_diabetes, load_breast_cancer, load_wine
import numpy as np
import matplotlib.pyplot as plt

boston = load_boston()
boston_data = boston.data
boston_target = boston.target

iris = load_iris()
iris_data = iris.data
iris_target = iris.target

diabetes = load_diabetes()
diabetes_data = diabetes.data
diabetes_target = diabetes.target

breast_cancer = load_breast_cancer()

breast_cancer_data = breast_cancer.data
breast_cancer_target = breast_cancer.target

wine = load_wine()

wine_data = wine.data
wine_target = wine.target

print(boston.data.shape)
print(iris.data.shape)
print(diabetes.data.shape)
print(breast_cancer.data.shape)
print(wine.data.shape)

