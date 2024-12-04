import numpy as np
import sympy as sp

# Define variables
x1, x2, x3, x4, y = sp.symbols('x1 x2 x3 x4 y')

class Matrix:
    def __init__(self, n):
        self.size = 4
        self.values = np.array([
            [-3.0504, -44.6371, -14.7046, 56.9748],
            [-1.4677, -89.0323, -25.8790, 110.2661],
            [-3.4798, -10.1452, -6.7681, 15.2601],
            [-1.2984, -66.4516, -19.4315, 82.3508]
        ])
        self.determinant = None
        self.eigenvalues = np.zeros(4)
        self.eigenvectors = np.zeros((4, 4))
        self.initial_conditions = np.array([[2.0000, -1.5000, 1.0000, -2.5000]])
        self.coefficients = np.zeros(4)
        self.inverse_eig = None
        self.inverse = None
        self.index_max_eigval = None
        self.matr_with_x = []

    def is_determinant_negative(self):
        self.determinant = np.linalg.det(self.values)
        return self.determinant < 0

    def calculate_eigenvalues_and_eigenvectors(self):
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.values)

    def calculate_inverse(self):
        self.inverse_eig = np.linalg.inv(self.eigenvectors)

    def calculate_coefficients(self):
        self.calculate_inverse()
        if self.inverse_eig is not None:
            self.coefficients = self.inverse_eig @ self.initial_conditions.T

    def calculate_inverse_from_eigenvectors(self):
        diagonal_matrix = np.diag(self.coefficients.flatten())
        self.inverse = np.linalg.inv(self.eigenvectors @ diagonal_matrix)

    def find_index_of_max_eigenvector(self):
        self.index_max_eigval = np.argmax(abs(self.eigenvalues))

    def expr_max_exp(self):
        self.find_index_of_max_eigenvector()
        x = None
        if (self.index_max_eigval == 0):
            x = x1
        elif (self.index_max_eigval == 1):
            x = x2
        elif (self.index_max_eigval == 2):
            x = x3
        else:
            x = x4
        xi = sp.solve(self.inverse[self.index_max_eigval][0]*x1+self.inverse[self.index_max_eigval][1]*x2+self.inverse[self.index_max_eigval][2]*x3+self.inverse[self.index_max_eigval][3]*x4, x1)[0]
        for i in range(self.size):
            temp_expr = sum(self.values[i][j] * sp.symbols(f'x{j+1}') for j in range(self.size))
            self.matr_with_x.append(temp_expr)
        print(self.matr_with_x)
        substitutedequations = [eq.subs(x1, xi) for eq in self.matr_with_x]
        simplifiedequations = [sp.simplify(eq) for eq in substitutedequations]
        print("\nObtained equations:")
        for eq in simplifiedequations:
            sp.pprint(eq)
            print()

    def calc(self):
        self.calculate_eigenvalues_and_eigenvectors()
        self.calculate_coefficients()
        self.calculate_inverse_from_eigenvectors()
        self.expr_max_exp()

a = Matrix(4)
a.calc()
print(a.values)
print(a.determinant)
print(a.eigenvalues)
print(a.eigenvectors)
print(a.initial_conditions)
print(a.inverse)
print("max_index", a.index_max_eigval)
print(a.matr_with_x)