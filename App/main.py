

import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton, QMainWindow, QSpinBox
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCharts import QChart, QChartView, QLineSeries
import sympy as sp

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


class MatrixApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Matrix Exponent App")
        self.setGeometry(100, 100, 800, 600)
        self.set_dark_theme()
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QVBoxLayout()
        self.grid_layout = QGridLayout()
        
        self.size_input = QSpinBox()
        self.size_input.setRange(2, 10)
        self.size_input.setValue(3)
        self.size_input.valueChanged.connect(self.create_matrix_inputs)
        
        self.layout.addWidget(QLabel("Enter matrix size (n):"))
        self.layout.addWidget(self.size_input)
        
        self.entries = []
        self.initial_conditions_entries = []
        self.create_matrix_inputs()
        
        self.plot_button = QPushButton("Plot Exponents")
        self.plot_button.clicked.connect(self.plot_exponents)
        
        self.layout.addLayout(self.grid_layout)
        self.layout.addWidget(self.plot_button)
        
        self.chart_view = QChartView()
        self.layout.addWidget(self.chart_view)

        self.initial_conditions_label = QLabel("Enter initial conditions (1*n):")
        self.layout.addWidget(self.initial_conditions_label)
        self.create_initial_conditions_inputs()
        self.central_widget.setLayout(self.layout)
        self.matrix = Matrix(self.size_input.value())
        self.matrix1 = Matrix(self.size_input.value() - 1)

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
        
        self.matrix = Matrix(self.size_input.value())
        for i in range(self.size_input.value()):
            row_entries = []
            for j in range(self.size_input.value()):
                entry = QLineEdit()
                entry.setPlaceholderText(f"Element ({i+1},{j+1})")
                entry.setText("0")
                row_entries.append(entry)
                self.grid_layout.addWidget(entry, i, j)
            self.entries.append(row_entries)
        
        self.create_initial_conditions_inputs()
    

    def create_initial_conditions_inputs(self):
        for entry in self.initial_conditions_entries:
            entry.deleteLater()
        self.initial_conditions_entries.clear()
        
        n = self.size_input.value()
        self.initial_conditions_grid = QGridLayout()
        for j in range(n):
            entry = QLineEdit()
            entry.setPlaceholderText(f"Initial Condition {j+1}")
            entry.setText("0")
            self.initial_conditions_entries.append(entry)
            self.initial_conditions_grid.addWidget(entry, 0, j)
        self.layout.addLayout(self.initial_conditions_grid)

    def calculate(self):
        n = self.size_input.value()
        for i in range(n):
            for j in range(n):
                try:
                    self.matrix.values[i][j] = float(self.entries[i][j].text())
                except ValueError:
                    pass
        for j in range(n):
                try:
                    self.matrix.initial_conditions[j] = float(self.initial_conditions_entries[j].text())
                except ValueError:
                    pass
        self.matrix.calc()
        #self.matrix1.values = self.matrix.
    
        
    def plot_exponents(self):
        self.calculate()
        chart = QChart()
        n = self.size_input.value()
        if not self.matrix.is_determinant_negative():
            for i in range(n):
                series = QLineSeries()
                for t in np.linspace(0, 10, 100):
                    exp_value = np.exp(self.matrix.eigenvalues[i] * t) * self.matrix.initial_conditions[i]
                    series.append(t, exp_value)
                for t in np.linspace(0, 10, 100):
                    exp_value = np.exp(self.matrix.eigenvalues[i] * t) * self.matrix.initial_conditions[i]
                    series.append(t, exp_value)
                chart.addSeries(series)
        
        chart.createDefaultAxes()
        chart.setTitle("Transient Processes of Matrix Elements")
        self.chart_view.setChart(chart)
        self.layout.addWidget(self.initial_conditions_label)
        self.layout.addLayout(self.initial_conditions_grid)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MatrixApp()
    window.show()
    sys.exit(app.exec())