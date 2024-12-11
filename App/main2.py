import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp
import math as m
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QSpinBox, QLineEdit, QPushButton, QGridLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Графическое приложение на PyQt6")

        self.main_layout = QHBoxLayout()

        # Правая часть для графиков
        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout()
        self.right_widget.setLayout(self.right_layout)

        self.fig1, self.ax1 = plt.subplots()
        self.canvas1 = FigureCanvas(self.fig1)
        self.right_layout.addWidget(self.canvas1)

        self.fig2, self.ax2 = plt.subplots()
        self.canvas2 = FigureCanvas(self.fig2)
        self.right_layout.addWidget(self.canvas2)

        # Левая часть для ввода данных
        self.left_widget = QWidget()
        self.left_layout = QVBoxLayout()
        self.left_widget.setLayout(self.left_layout)

        self.n_label = QLabel("Размер матрицы (n):")
        self.n_spinbox = QSpinBox()
        self.n_spinbox.setMinimum(3)
        self.n_spinbox.setMaximum(10)
        self.n_spinbox.valueChanged.connect(self.create_matrix_input)

        self.matrix_label = QLabel("Матрица (n x n):")
        self.matrix_grid = QGridLayout()

        self.initial_label = QLabel("Начальные условия (1 x n):")
        self.initial_grid = QGridLayout()

        self.error_label = QLabel("Ошибка:")
        self.error_input = QLineEdit()

        self.plot_button = QPushButton("Построить графики")
        self.plot_button.clicked.connect(self.plot_graphs)

        self.left_layout.addWidget(self.n_label)
        self.left_layout.addWidget(self.n_spinbox)
        self.left_layout.addWidget(self.matrix_label)
        self.left_layout.addLayout(self.matrix_grid)
        self.left_layout.addWidget(self.initial_label)
        self.left_layout.addLayout(self.initial_grid)
        self.left_layout.addWidget(self.error_label)
        self.left_layout.addWidget(self.error_input)
        self.left_layout.addWidget(self.plot_button)

        self.main_layout.addWidget(self.left_widget)
        self.main_layout.addWidget(self.right_widget)

        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)

    def create_matrix_input(self):
        n = self.n_spinbox.value()
        # Clear previous matrix inputs
        for i in reversed(range(self.matrix_grid.count())):
            self.matrix_grid.itemAt(i).widget().setParent(None)
        
        # Create matrix input fields
        self.matrix_inputs = []
        for i in range(n):
            row_inputs = []
            for j in range(n):
                input_field = QLineEdit()
                input_field.setFixedSize(40, 25)  # Уменьшенные размеры окон ввода
                self.matrix_grid.addWidget(input_field, i, j)
                row_inputs.append(input_field)
            self.matrix_inputs.append(row_inputs)

        # Clear previous initial condition inputs
        for i in reversed(range(self.initial_grid.count())):
            self.initial_grid.itemAt(i).widget().setParent(None)
        
        # Create initial condition input fields
        self.initial_inputs = []
        for i in range(n):
            input_field = QLineEdit()
            input_field.setFixedSize(40, 25)  # Уменьшенные размеры окон ввода
            self.initial_grid.addWidget(input_field, 0, i)
