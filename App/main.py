# Matrix Exponent App

import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton, QMainWindow, QSpinBox, QHBoxLayout
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCharts import QChart, QChartView, QLineSeries
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
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

def plot_transient_response2(temp_sol, temp_sol2, size_matr_1, size_matr_2, chart_view):
    chart = QChart()
    for i in range(size_matr_1):
        series = QLineSeries()
        for j in range(len(temp_sol.t)):
            series.append(temp_sol.t[j], temp_sol.y[i][j])
        series.setName(f'x{i+1}')
        chart.addSeries(series)
    for i in range(size_matr_2):
        series = QLineSeries()
        for j in range(len(temp_sol2.t)):
            series.append(temp_sol2.t[j], temp_sol2.y[i][j])
        series.setName(f'x_пониж{i+2}')
        chart.addSeries(series)
    chart.createDefaultAxes()
    chart.setTitle('Transient Response')
    chart_view.setChart(chart)
    chart_view.setRubberBand(QChartView.RubberBand.RectangleRubberBand)

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_error_response(solution1_new, solution2_new, size_matr_2, chart_view, window_size=10):
    errors = solution1_new.y[1:size_matr_2+1, :] - solution2_new.y
    chart = QChart()
    for i in range(size_matr_2):
        smoothed_errors = moving_average(errors[i], window_size)
        smoothed_t = moving_average(solution1_new.t, window_size)
        series = QLineSeries()
        for j in range(len(smoothed_t)):
            series.append(smoothed_t[j], smoothed_errors[j])
        series.setName(f'Smoothed Error in x{i+2}')
        chart.addSeries(series)
    chart.createDefaultAxes()
    chart.setTitle('Smoothed Error Response')
    chart_view.setChart(chart)
    chart_view.setRubberBand(QChartView.RubberBand.RectangleRubberBand)

class MatrixApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Matrix Exponent App")
        self.setGeometry(100, 100, 1200, 600)
        self.set_dark_theme()
        self.flag = True
        self.flag2 = True
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QHBoxLayout()
        
        self.input_layout = QVBoxLayout()
        self.grid_layout = QGridLayout()
        
        self.size_input = QSpinBox()
        self.size_input.setRange(2, 10)
        self.size_input.setValue(3)
       
        self.input_layout.addWidget(QLabel("Введите размер матрицы (n):"))
        self.input_layout.addWidget(self.size_input)
        self.entries = []
        self.initial_conditions_entries = []
        self.error_entries = []
        self.values_matrix = np.zeros((int(self.size_input.text()), int(self.size_input.text())))
        self.values_matrix_initial_cond = np.zeros((1, int(self.size_input.text())))
        self.error_value = None

        self.size_input.valueChanged.connect(self.create_matrix_inputs)
        self.plot_button = QPushButton("Понизить порядок")
        self.chart_view1 = QChartView()
        self.chart_view2 = QChartView()
        
        self.chart_layout = QVBoxLayout()
        self.plot_button.clicked.connect(self.plot_exponents)
        
        self.input_layout.addLayout(self.grid_layout)
        self.input_layout.addWidget(self.plot_button)

        self.chart_layout.addWidget(self.chart_view1)
        self.chart_layout.addWidget(self.chart_view2)
        
        self.layout.addLayout(self.input_layout)
        self.layout.addLayout(self.chart_layout)
        
        self.central_widget.setLayout(self.layout)
        

    def set_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        self.setPalette(palette)

    def create_matrix_inputs(self):
        for i in reversed(range(len(self.entries))):
            for j in range(len(self.entries[i])):
                self.grid_layout.itemAtPosition(i, j).widget().deleteLater()
            self.entries.pop()
        
        for i in range(self.size_input.value()):
            row_entries = []
            for j in range(self.size_input.value()):
                entry = QLineEdit()
                entry.setPlaceholderText(f"Элемент ({i+1},{j+1})")
                entry.setText("0")
                row_entries.append(entry)
                self.grid_layout.addWidget(entry, i, j)
            self.entries.append(row_entries)
        
        
        self.create_initial_conditions_inputs()
        
        self.create_error_input()
    
    def create_initial_conditions_inputs(self):
        if (self.flag):
            self.initial_conditions_label = QLabel("Введите начальные условия:")
            self.input_layout.addWidget(self.initial_conditions_label)
            self.flag = False
        else:
            self.initial_conditions_label.deleteLater()
            self.initial_conditions_label = QLabel("Введите начальные условия:")
            self.input_layout.addWidget(self.initial_conditions_label)
        for entry in self.initial_conditions_entries:
            entry.deleteLater()
        self.initial_conditions_entries.clear()
        
        n = self.size_input.value()
        self.initial_conditions_grid = QGridLayout()
        for j in range(n):
            entry = QLineEdit()
            entry.setPlaceholderText(f"Начальное условие {j+1}")
            entry.setText("0")
            self.initial_conditions_entries.append(entry)
            self.initial_conditions_grid.addWidget(entry, 0, j)
        self.input_layout.addLayout(self.initial_conditions_grid)
        
    def read_matrix_data(self):
        n = self.size_input.value()
        self.values_matrix = np.zeros((n, n))
        self.values_matrix_initial_cond = np.array([np.zeros(n)])
        self.error_value = 0.0
        
        for i in range(n):
            for j in range(n):
                try:
                    self.values_matrix[i][j] = float(self.entries[i][j].text())
                except ValueError:
                    self.values_matrix[i][j] = 0.0
                    
        initial_conditions = []
        for j in range(n):
            try:
                initial_conditions.append(float(self.initial_conditions_entries[j].text()))
            except ValueError:
                initial_conditions.append(0.0)
        self.values_matrix_initial_cond = np.array([initial_conditions])
                
        try:
            self.error_value = float(self.error_entry.text())
        except ValueError:
            self.error_value = 0.0

    def create_error_input(self):
        if (self.flag2):
            self.error_label = QLabel("Введите ошибку :")
            self.input_layout.addWidget(self.error_label)
            self.flag2 = False
        else:
            self.error_label.deleteLater()
            self.error_label = QLabel("Введите ошибку :")
            self.input_layout.addWidget(self.error_label)
        if self.error_entries:
            self.error_entries[0].deleteLater()
        self.error_entries.clear()
        self.error_entry = QLineEdit()
        self.error_entry.setPlaceholderText("Значение ошибки")
        self.error_entry.setText("0")
        self.error_entries.append(self.error_entry)
        self.input_layout.addWidget(self.error_entry)

    def plot_exponents(self):
        self.read_matrix_data()
        matrix_1 = Matrix(int(self.size_input.text()), self.values_matrix, self.values_matrix_initial_cond, self.error_value)
        matrix_1.calc()
        new_values = matrix_1.matr_resh
        matrix_2 = Matrix(int(self.size_input.text())-1, new_values, self.values_matrix_initial_cond[:, 1:], self.error_value)
        matrix_2.calc()
        print(matrix_1.values)
        t_span = (0, 10)
        t_eval = np.linspace(t_span[0], t_span[1], 400)
        solution1 = solve_ivp(system_of_equations, t_span, matrix_1.initial_conditions.flatten(), t_eval=t_eval, args=(matrix_1.values,))
        solution2 = solve_ivp(system_of_equations, t_span, matrix_2.initial_conditions.flatten(), t_eval=t_eval, args=(matrix_2.values,))

        initial_conditions_new = None
        print("Solution 1:")
        min_gran = min(matrix_1.gran)
        print(min_gran)
        for i, t in enumerate(solution1.t):
            if abs(t - min_gran) <= 0.02:
                initial_conditions_new = np.array([solution1.y[:, i]])
                print(f"t = {t:.2f}, y = {solution1.y[:, i]}")
                break
        
        a_new = Matrix(int(self.size_input.text()), self.values_matrix, initial_conditions_new, self.error_value)
        a_new.calc()

        new_values1 = a_new.matr_resh
        b_new = Matrix(int(self.size_input.text())-1, new_values1, initial_conditions_new[:, 1:], self.error_value)
        b_new.calc()
        t_span = (min(matrix_1.gran), max(matrix_1.gran))
        t_eval = np.linspace(t_span[0], t_span[1], 400)
        solution1_new = solve_ivp(system_of_equations, t_span, a_new.initial_conditions.flatten(), t_eval=t_eval, args=(a_new.values,))
        solution2_new = solve_ivp(system_of_equations, t_span, b_new.initial_conditions.flatten(), t_eval=t_eval, args=(b_new.values,))
        
        plot_transient_response2(solution1_new, solution2_new, int(self.size_input.text()), int(self.size_input.text())-1, self.chart_view1)
        plot_error_response(solution1_new, solution2_new, int(self.size_input.text())-1, self.chart_view2)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MatrixApp()
    window.show()
    sys.exit(app.exec())