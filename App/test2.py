import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
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
        for i in range(self.size):
            gran_value = (1 / self.eigenvalues[i]) * m.log(self.err)
            print(gran_value)
            self.gran.append(gran_value)

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
        for i, eq in enumerate(simplifiedequations[1:]):
            coefficients = [float(eq.coeff(sp.symbols(f'x{j+1}'))) for j in range(1, self.size)]
            if i < self.matr_resh.shape[0]:
                self.matr_resh[i, :len(coefficients)] = coefficients

    def construct_equations_matrix(self):
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

def system_of_equations(t, y, values):
    x = y
    dx_dt = values @ x
    return dx_dt

def plot_transient_response(matrix, n):
    t_span = (n, 5)
    t_eval = np.linspace(t_span[0], t_span[1], 400)
    solution = solve_ivp(system_of_equations, t_span, matrix.initial_conditions.flatten(), t_eval=t_eval, args=(matrix.values,))

    plt.figure(figsize=(12, 8))
    for i in range(matrix.size):
        plt.plot(solution.t, solution.y[i], label=f'x{i+1}')
    plt.title('Transient Response')
    plt.xlabel('Time')
    plt.ylabel('Response')
    plt.legend()
    plt.grid()
    plt.show()

def plot_transient_response2(temp_sol, temp_sol2, size_matr_1, size_matr_2):

    plt.figure(figsize=(12, 8))
    for i in range(size_matr_1):
        plt.plot(temp_sol.t, temp_sol.y[i], label=f'x{i+1}')
    for i in range(size_matr_2):
        plt.plot(temp_sol2.t, temp_sol2.y[i], label=f'x_пониж{i+2}')
    plt.title('Transient Response')
    plt.xlabel('Time')
    plt.ylabel('Response')
    plt.legend()
    plt.grid()
    plt.show()

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_error_response(solution1_new, solution2_new, size_matr_2, window_size=10):
    errors = solution1_new.y[1:size_matr_2+1, :] - solution2_new.y
    
    plt.figure(figsize=(12, 8))
    for i in range(size_matr_2):
        smoothed_errors = moving_average(errors[i], window_size)
        smoothed_t = moving_average(solution1_new.t, window_size)
        plt.plot(smoothed_t, smoothed_errors, label=f'Smoothed Error in x{i+2}')
    
    plt.title('Smoothed Error Response')
    plt.xlabel('Time')
    plt.ylabel('Smoothed Error')
    plt.legend()
    plt.grid()
    plt.show()

# Пример использования
values = np.array([[-49.1038 , 46.4140 , 52.4756 , 31.2930 , -0.2952],
  [-22.3111 , 53.5784 , 41.5627 , 45.1743, -22.1660],
 [79.7428, -84.5988, -92.3521, -59.1265 ,  5.5125],
  [-10.2607 , -0.8217 ,  7.6380 , -4.5224 ,  8.1703],
  [-58.6736 , 85.6086 , 77.5850 , 65.4118, -21.6001]])
initial_conditions = np.array([[5, 2, -1.5, 1, -2.5]])
err = 0.01
a = Matrix(5, values, initial_conditions, err)
a.calc()

# Создание новой матрицы без первой строки и первого столбца
new_values = a.matr_resh
b = Matrix(4, new_values, initial_conditions[:, 1:], err)
b.calc()

t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 400)
solution1 = solve_ivp(system_of_equations, t_span, a.initial_conditions.flatten(), t_eval=t_eval, args=(a.values,))
solution2 = solve_ivp(system_of_equations, t_span, b.initial_conditions.flatten(), t_eval=t_eval, args=(b.values,))

initial_conditions_new = None
print("Solution 1:")
min_gran = min(a.gran)
print(min_gran)
for i, t in enumerate(solution1.t):
    if abs(t - min_gran) <= 0.02:
        initial_conditions_new = np.array([solution1.y[:, i]])
        print(f"t = {t:.2f}, y = {solution1.y[:, i]}")
        break  # Optionally, break if there's only one match

a_new = Matrix(5, values, initial_conditions_new, err)
a_new.calc()


# Создание новой матрицы без первой строки и первого столбца
new_values1 = a_new.matr_resh
b_new = Matrix(4, new_values1, initial_conditions_new[:, 1:], err)
b_new.calc()

t_span = (min(a.gran), max(a.gran))
t_eval = np.linspace(t_span[0], t_span[1], 400)
solution1_new = solve_ivp(system_of_equations, t_span, a_new.initial_conditions.flatten(), t_eval=t_eval, args=(a_new.values,))
solution2_new = solve_ivp(system_of_equations, t_span, b_new.initial_conditions.flatten(), t_eval=t_eval, args=(b_new.values,))

plot_transient_response2(solution1_new, solution2_new, a.size, b.size)
plot_error_response(solution1_new, solution2_new, b.size)

