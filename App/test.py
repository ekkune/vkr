import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math as m
class Matrix:
    def __init__(self, n, values, initial_conditions, err):
        self.size = n
        self.err = err
        self.values = values
        self.determinant = None
        self.eigenvalues = np.zeros(n)
        self.eigenvectors = np.zeros((n, n))
        self.initial_conditions = initial_conditions
        self.coefficients = np.zeros(n)
        self.inverse_eig = None
        self.inverse = None
        self.index_max_eigval = None
        self.matr_with_x = []
        self.matr_resh = np.zeros((n-1, n-1))
        self.gran = []

    def calc_gran(self):
        for i in range (self.size):
            self.gran.append(( 1 / self.eigenvalues[i] ) * m.log(self.err))

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
        x_resh = sp.symbols(f'x{self.index_max_eigval + 1}')
        xi = sp.solve(sum(self.inverse[self.index_max_eigval][i] * sp.symbols(f'x{i + 1}') for i in range(self.size)), x_resh)[0]
        for i in range(self.size):
            temp_expr = sum(self.values[i][j] * sp.symbols(f'x{j + 1}') for j in range(self.size))
            self.matr_with_x.append(temp_expr)
        substitutedequations = [eq.subs(x_resh, xi) for eq in self.matr_with_x]
        simplifiedequations = [sp.simplify(eq) for eq in substitutedequations]
        print("\nПолученные выражения:")
        for i, eq in enumerate(simplifiedequations[1:]):  # Исключаем первую строку
            coefficients = [float(eq.coeff(sp.symbols(f'x{j+1}'))) for j in range(1, self.size)]  # Исключаем первый столбец
            if i < self.matr_resh.shape[0]:
                self.matr_resh[i, :len(coefficients)] = coefficients



    def construct_equations_matrix(self):
        # Построение матрицы уравнений с экспонентами
        equations_matrix = []
        for i in range(self.size):
            equation = sum(self.coefficients[j] * self.eigenvectors[j, i] * sp.exp(self.eigenvalues[j]) for j in range(self.size))
            equations_matrix.append(equation)
        return equations_matrix

    def calc(self):
        self.calculate_eigenvalues_and_eigenvectors()
        self.calculate_coefficients()
        self.calculate_inverse_from_eigenvectors()
        self.expr_max_exp()
        self.calc_gran()
        equations_matrix = self.construct_equations_matrix()


def plot_transient_response(с, n):
    n = n
    t = np.linspace(n, 100, 400)
    responses = np.zeros((с.size, t.size))

    for i in range(с.size):
        homogeneous_part = sum(
            с.coefficients[j] * с.eigenvectors[j, i] * np.exp(с.eigenvalues[j] * t) for j in range(с.size)
        )

        responses[i] = homogeneous_part


    plt.figure(figsize=(12, 8))
    for i in range(с.size):
        plt.plot(t, responses[i], label=f'x{i+1}')

    plt.title('Transient Response')
    plt.xlabel('Time')
    plt.ylabel('Response')
    plt.legend()
    plt.grid()
    plt.show()
    
def plot_transient_response2(с,d, n, m):
    t = np.linspace(n, m, 400)
    responses = np.zeros((с.size, t.size))
    responses2 = np.zeros((d.size, t.size))
    for i in range(с.size):
        homogeneous_part = sum(
            с.coefficients[j] * с.eigenvectors[j, i] * np.exp(с.eigenvalues[j] * t) for j in range(с.size)
        )

        responses[i] = homogeneous_part
    for i in range(d.size):
        homogeneous_part = sum(
            d.coefficients[j] * d.eigenvectors[j, i] * np.exp(d.eigenvalues[j] * t) for j in range(d.size)
        )

        responses2[i] = homogeneous_part


    plt.figure(figsize=(12, 8))
    for i in range(с.size):
        plt.plot(t, responses[i], label=f'x{i+1}')
    for i in range(d.size):
        plt.plot(t, responses2[i], label=f'x_пониж{i+2}')

    plt.title('Transient Response')
    plt.xlabel('Time')
    plt.ylabel('Response')
    plt.legend()
    plt.grid()
    plt.show()

# Пример использования
values = np.array([
    [-3.0504, -44.6371, -14.7046, 56.9748],
    [-1.4677, -89.0323, -25.8790, 110.2661],
    [-3.4798, -10.1452, -6.7681, 15.2601],
    [-1.2984, -66.4516, -19.4315, 82.3508]
])
initial_conditions = np.array([[2.0000, -1.5000, 1.0000, -2.5000]])
err = 0.01
a = Matrix(4, values, initial_conditions, err)
a.calc()
print(a.gran)
# Создание новой матрицы без первой строки и первого столбца
new_values = a.matr_resh
b = Matrix(3, new_values, initial_conditions[:, 1:], err)
b.calc()
plot_transient_response(a, 0)
plot_transient_response(b, 0)
plot_transient_response2(a, b, min(a.gran), max(a.gran))