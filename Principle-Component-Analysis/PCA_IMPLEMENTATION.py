import pandas as pd
import numpy as np


# from FUNCTIONS import *


class PcaImplementation:
    def __init__(self):
        self.dataset = {"X": [1, -1, 4],
                        "Y": [2, 1, 3],
                        "Z": [1, 3, -1]}

        self.df = pd.DataFrame(self.dataset)
        self.length_key = 0
        self.eigenvalues, self.eigenvectors, self.eigen_vector_transpose = [], [], []
        self.display_of_inputs()
        self.mean()
        self.covariance()
        self.eigen_values()
        self.eigen_vector()

    def display_of_inputs(self):

        print("The values for given dimensions are : \n")
        for dimension in self.dataset:
            print(dimension, " = ", self.dataset[dimension], "\n")

    def mean(self):

        for dimension in self.dataset:
            self.length_key = len(self.dataset[dimension])
            # print(self.length_key)
            print(f"\nthe mean of {dimension} dimension is : ", self.df[dimension].sum() / self.length_key)

    def covariance(self):
        print("\nThe Covariance Matrix is : \n\n", self.df.cov())

    def eigen_values(self):
        m = (self.df.cov())
        self.eigenvalues, self.eigenvectors = np.linalg.eig(m)
        print("\n the Eigen value for the obtained covariance matrix is : ", self.eigenvalues)

    def eigen_vector(self):
        # print("Eigenvectors of the said matrix", self.eigenvectors)

        max_eigen_val = max(self.eigenvalues)
        result = self.eigenvalues.tolist()

        place_of_max_eigen_val = result.index(max_eigen_val)
        # print(place_of_max_eigen_val)

        print(f"\nso the eigen vector corresponding to {max_eigen_val} is \n")
        self.eigen_vector_transpose = list(map(list, zip(*self.eigenvectors)))
        print(self.eigen_vector_transpose)

        print(f"\nso the pca of the corresponding dataset is : \n")
        print(self.eigen_vector_transpose[place_of_max_eigen_val])


if __name__ == '__main__':
    pca_implementation_obj = PcaImplementation()
