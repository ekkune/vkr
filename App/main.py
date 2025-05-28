import matplotlib
matplotlib.use('QtAgg')  # Ensure QtAgg backend is used
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QMainWindow, QSpinBox, QHBoxLayout, QDialog, QFileDialog,
    QGroupBox, QTextEdit, QTableWidgetItem, QTableWidget, QRadioButton, 
    QMessageBox, QComboBox, QScrollArea, QCheckBox, QSizePolicy, QSplitter
)
from PyQt6.QtGui import QPalette, QColor, QPen, QDoubleValidator, QFontMetrics, QPainter
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PyQt6.QtCore import Qt, QLocale, QPointF
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import sys
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

class NoCommaLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QLineEdit {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 3px;
                font-family: monospace;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 1px solid #1f77b4;
            }
        """)

    def keyPressEvent(self, event):
        if event.text() == ",":
            QMessageBox.warning(self, "Ошибка ввода", "Использование запятой не допускается. Используйте точку для десятичных чисел.")
            return
        super().keyPressEvent(event)

# Настройка Matplotlib для использования MathText
plt.rc('text', usetex=False)
plt.rc('font', family='Arial', size=12)

class TimeBoundaryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выбор правой границы временного интервала наблюдения процессов")
        self.layout = QVBoxLayout()
        
        self.time_label = QLabel("Введите правую границу временного интервала наблюдения процессов T(с):")
        self.time_input = QLineEdit("10.0")
        validator = QDoubleValidator(bottom=0.0)
        validator.setLocale(QLocale("C"))
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.time_input.setValidator(validator)
        
        self.ok_button = QPushButton("Подтвердить")
        self.ok_button.clicked.connect(self.accept)
        
        self.layout.addWidget(self.time_label)
        self.layout.addWidget(self.time_input)
        self.layout.addWidget(self.ok_button)
        self.setLayout(self.layout)
    
    def get_time(self):
        try:
            return float(self.time_input.text().replace(",", "."))
        except ValueError:
            return 10.0

from PyQt6.QtGui import QPainter
from PyQt6.QtCharts import QChart, QValueAxis
from PyQt6.QtCore import Qt, QPointF

class ChartWindow(QDialog):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setGeometry(100, 100, 800, 600)
        self.chart_view = QChartView()
        self.chart_view.setRubberBand(QChartView.RubberBand.RectangleRubberBand)
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.chart_view)
        self.setLayout(self.layout)
        
        # Tooltip label
        self.tooltip = QLabel(self.chart_view)
        self.tooltip.setStyleSheet("""
            background-color: rgba(255, 255, 255, 200);
            border: 1px solid black;
            padding: 2px;
            font-size: 12px;
        """)
        self.tooltip.hide()
        
        # Enable mouse tracking
        self.chart_view.setMouseTracking(True)
        self.chart_view.mouseMoveEvent = self.mouse_move_event
        self.series_data = []
        
    def mouse_move_event(self, event):
        pos = event.pos()
        chart = self.chart_view.chart()
        mapped_pos = self.chart_view.mapToScene(pos).toPoint()
        
        closest_dist = float('inf')
        closest_point = None
        closest_series_name = ""
        closest_value = None
        
        for series in chart.series():
            for i in range(series.count()):
                point = series.at(i)
                scene_point = chart.mapToPosition(point)
                dist = ((scene_point.x() - mapped_pos.x()) ** 2 + 
                       (scene_point.y() - mapped_pos.y()) ** 2) ** 0.5
                
                if dist < closest_dist and dist < 20:
                    closest_dist = dist
                    closest_point = point
                    closest_series_name = series.name()
                    closest_value = point
        
        if closest_point is not None:
            self.tooltip.setText(f"{closest_series_name}\nX: {closest_value.x():.3f}\nY: {closest_value.y():.3f}")
            self.tooltip.move(pos.x() + 10, pos.y() - 30)
            self.tooltip.show()
        else:
            self.tooltip.hide()
    
    def setup_axes(self, chart, t_start, t_end, y_values=None):
        axis_x = QValueAxis()
        axis_x.setTitleText("Время (с)")
        axis_x.setRange(t_start, t_end)
        axis_x.setLabelFormat("%.2f")
        
        axis_y = QValueAxis()
        axis_y.setTitleText("х(t)")
        
        if y_values:
            y_min = min(y_values)
            y_max = max(y_values)
            margin = (y_max - y_min) * 0.1
            axis_y.setRange(y_min - margin, y_max + margin)
        
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        
        for series in chart.series():
            series.attachAxis(axis_x)
            series.attachAxis(axis_y)

class TransientResponseWindow(ChartWindow):
    def plot(self, solution1, solution2, choice, size1, size2, t_start, t_end):
        chart = QChart()
        
        y_values = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        if solution1 is not None and choice in [0, 2]:
            mask1 = (solution1.t >= t_start) & (solution1.t <= t_end)
            for i in range(size1):
                series = QLineSeries()
                series.setName(f'x{i+1}')
                series.setColor(QColor(colors[i % len(colors)]))
                y = solution1.y[i][mask1]
                x = solution1.t[mask1]
                if len(x) == 0 or len(y) == 0:
                    raise ValueError("Нет данных для построения графика исходной модели")
                y_values.extend(y)
                for j in range(len(x)):
                    series.append(x[j], y[j])
                chart.addSeries(series)
                self.series_data.append((series, x, y))
        
        if solution2 is not None and choice in [1, 2]:
            mask2 = (solution2.t >= t_start) & (solution2.t <= t_end)
            for i in range(size2):
                series = QLineSeries()
                series.setName(f'x_упр{i+1}')
                series.setColor(QColor(colors[(i+1) % len(colors)]))
                series.setPen(QPen(Qt.PenStyle.DashLine))
                y = solution2.y[i][mask2]
                x = solution2.t[mask2]
                if len(x) == 0 or len(y) == 0:
                    raise ValueError("Нет данных для построения графика пониженной модели")
                y_values.extend(y)
                for j in range(len(x)):
                    series.append(x[j], y[j])
                chart.addSeries(series)
                self.series_data.append((series, x, y))
        
        if not chart.series():
            raise ValueError("Нет данных для отображения на графике")
        
        self.setup_axes(chart, t_start, t_end, y_values)
        self.chart_view.setChart(chart)
    
class ErrorResponseWindow(ChartWindow):
    def plot(self, solution1, solution2, full_size, excluded_index):
        from scipy.interpolate import interp1d
        chart = QChart()
        chart.setTitle("Абсолютные ошибки")

        original_indices = [i for i in range(full_size) if i != excluded_index]
        t1 = solution1.t
        t2 = solution2.t

        for reduced_idx, orig_idx in enumerate(original_indices):
            error_series = QLineSeries()
            error_series.setName(f'Ошибка x{orig_idx + 1}')
            
            interp_func = interp1d(t2, solution2.y[reduced_idx], kind='linear', fill_value="extrapolate")
            interp_values = interp_func(t1)
            errors = solution1.y[orig_idx] - interp_values

            for j in range(len(t1)):
                error_series.append(t1[j], errors[j])
            
            chart.addSeries(error_series)
            self.series_data.append((error_series, t1, errors))

        axis_x = QValueAxis()
        axis_x.setTitleText("Время (с)")
        axis_x.setLabelFormat("%.2f")

        axis_y = QValueAxis()
        axis_y.setTitleText("Ошибка")
        axis_y.setLabelFormat("%.6f")

        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)

        for series in chart.series():
            series.attachAxis(axis_x)
            series.attachAxis(axis_y)

        self.chart_view.setChart(chart)

class QualityMetricsDialog(QDialog):
    def __init__(self, errors, selected_criteria, full_size, excluded_index, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Оценки качества")
        self.setGeometry(200, 200, 500, 400)
        
        layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        
        error_text = "<b>Результаты оценки качества:</b><br><br>"
        
        # Define the table headers: one column for each variable x1, x2, ..., xn
        headers = ["Переменная"] + [f"δ_{i+1}" for i in range(full_size)]
        
        # Start the table
        error_text += "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse; font-family: Arial; font-size: 12px;'>"
        
        # Header row
        error_text += "<tr><th>" + "</th><th>".join(headers) + "</th></tr>"
        
        # Variable row (x1, x2, ..., xn)
        var_row = [""] + [f"x{i+1}" for i in range(full_size)]
        error_text += "<tr><td>" + "</td><td>".join(var_row) + "</td></tr>"
        
        # Map the errors to the correct columns based on original indices
        original_indices = [idx for idx in range(full_size) if idx != excluded_index]
        error_mapping = {original_indices[i]: i for i in range(len(original_indices))}  # Maps original index to error index
        
        # Row for Type I error
        if 0 in selected_criteria:
            error_row = ["Оценка типа I"]
            for i in range(full_size):
                if i == excluded_index:
                    error_row.append("-")
                else:
                    error_idx = error_mapping[i]
                    error_row.append(f"{errors[0][error_idx]:.6f}")
            error_text += "<tr><td>" + "</td><td>".join(error_row) + "</td></tr>"
        else:
            error_row = ["Оценка типа I"] + ["-" for _ in range(full_size)]
            error_text += "<tr><td>" + "</td><td>".join(error_row) + "</td></tr>"
        
        # Row for Type II error
        if 1 in selected_criteria:
            error_row = ["Оценка типа II"]
            for i in range(full_size):
                if i == excluded_index:
                    error_row.append("-")
                else:
                    error_idx = error_mapping[i]
                    error_row.append(f"{errors[1][error_idx]:.6f}")
            error_text += "<tr><td>" + "</td><td>".join(error_row) + "</td></tr>"
        else:
            error_row = ["Оценка типа II"] + ["-" for _ in range(full_size)]
            error_text += "<tr><td>" + "</td><td>".join(error_row) + "</td></tr>"
        
        # Row for Type III error
        if 2 in selected_criteria:
            error_row = ["Оценка типа III"]
            for i in range(full_size):
                if i == excluded_index:
                    error_row.append("-")
                else:
                    error_idx = error_mapping[i]
                    error_row.append(f"{errors[2][error_idx]:.6f}")
            error_text += "<tr><td>" + "</td><td>".join(error_row) + "</td></tr>"
        else:
            error_row = ["Оценка типа III"] + ["-" for _ in range(full_size)]
            error_text += "<tr><td>" + "</td><td>".join(error_row) + "</td></tr>"
        
        # Close the table
        error_text += "</table>"
        
        self.text_edit.setHtml(error_text)
        layout.addWidget(self.text_edit)
        self.setLayout(layout)

class MatrixDisplayDialog(QDialog):
    def __init__(self, matrix, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Матрица пониженного порядка")
        self.matrix = matrix
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        self.table = QTableWidget()
        rows = len(self.matrix)
        cols = len(self.matrix[0]) if rows > 0 else 0
        
        self.table.setRowCount(rows)
        self.table.setColumnCount(cols)
        
        for i in range(rows):
            for j in range(cols):
                item = QTableWidgetItem(f"{self.matrix[i][j]:.6f}".rstrip('0').rstrip('.'))
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(i, j, item)
        
        self.table.horizontalHeader().setDefaultSectionSize(120)
        self.table.verticalHeader().setDefaultSectionSize(40)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        screen = QApplication.primaryScreen().availableGeometry()
        max_width = int(screen.width() * 0.8)
        max_height = int(screen.height() * 0.8)
        
        width = min(cols * 130 + 20, max_width)
        height = min(rows * 45 + 50, max_height)
        
        self.setMinimumSize(300, 200)
        self.resize(width, height)
        
        layout.addWidget(self.table)
        self.setLayout(layout)

class GraphChoiceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выбор графика")
        self.choice = 0
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        self.original_radio = QRadioButton("Решения исходной модели")
        self.reduced_radio = QRadioButton("Решения упрощенной модели")
        self.both_radio = QRadioButton("Решения исходной и упрощенной моделей")
        self.original_radio.setChecked(True)
        
        layout.addWidget(self.original_radio)
        layout.addWidget(self.reduced_radio)
        layout.addWidget(self.both_radio)
        
        button_layout = QHBoxLayout()
        ok_button = QPushButton("ОК")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Отмена")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def get_choice(self):
        if self.original_radio.isChecked():
            return 0
        elif self.reduced_radio.isChecked():
            return 1
        else:
            return 2

class CriteriaSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выбор критериев оценки")
        self.setGeometry(200, 200, 700, 400)  # Increased window size from 600x300 to 700x400
        
        layout = QVBoxLayout()
        
        error1_layout = QHBoxLayout()
        self.error1_cb = QCheckBox("Ошибка типа I")
        self.error1_cb.setChecked(True)
        fig1 = Figure(figsize=(4.5, 1.2), tight_layout=True)  # Increased figsize from 3.5x0.8 to 4.5x1.2
        canvas1 = FigureCanvasQTAgg(fig1)
        ax1 = fig1.add_subplot(111)
        ax1.text(0.1, 0.5, r'$\Delta_1 = \sqrt{\frac{1}{N-1} \frac{\sum_{j=1}^N (x_j(t_j) - \bar{x}_i(t_j))^2}{\sum_{j=1}^N x_i^2(t_j)}}$',
                 fontsize=13, verticalalignment='center')  # Increased font size from 10 to 12
        ax1.axis('off')
        error1_layout.addWidget(self.error1_cb)
        error1_layout.addWidget(canvas1)
        error1_layout.addStretch()
        
        error2_layout = QHBoxLayout()
        self.error2_cb = QCheckBox("Ошибка типа II")
        self.error2_cb.setChecked(True)
        fig2 = Figure(figsize=(4.5, 1.2), tight_layout=True)  # Increased figsize from 3.5x0.8 to 4.5x1.2
        canvas2 = FigureCanvasQTAgg(fig2)
        ax2 = fig2.add_subplot(111)
        ax2.text(0.1, 0.5, r'$\Delta_2 = \frac{\frac{1}{N} \sum_{j=1}^N |x_j(t_j) - \bar{x}_i(t_j)|}{\max |x_i(t_j)|}$',
                 fontsize=13, verticalalignment='center')  # Increased font size from 10 to 12
        ax2.axis('off')
        error2_layout.addWidget(self.error2_cb)
        error2_layout.addWidget(canvas2)
        error2_layout.addStretch()
        
        error3_layout = QHBoxLayout()
        self.error3_cb = QCheckBox("Ошибка типа III")
        self.error3_cb.setChecked(True)
        fig3 = Figure(figsize=(4.5, 1.2), tight_layout=True)  # Increased figsize from 3.5x0.8 to 4.5x1.2
        canvas3 = FigureCanvasQTAgg(fig3)
        ax3 = fig3.add_subplot(111)
        ax3.text(0.1, 0.5, r'$\Delta_3 = \frac{\frac{1}{N} \sum_{j=1}^N |x_j(t_j) - \bar{x}_i(t_j)|}{\frac{1}{N} \sum_{j=1}^N |x_i(t_j)|}$',
                 fontsize=13, verticalalignment='center')  # Increased font size from 10 to 12
        ax3.axis('off')
        error3_layout.addWidget(self.error3_cb)
        error3_layout.addWidget(canvas3)
        error3_layout.addStretch()
        
        self.ok_button = QPushButton("ОК")
        self.ok_button.clicked.connect(self.validate_and_accept)
        
        layout.addLayout(error1_layout)
        layout.addLayout(error2_layout)
        layout.addLayout(error3_layout)
        layout.addStretch()
        layout.addWidget(self.ok_button)
        self.setLayout(layout)
    
    def validate_and_accept(self):
        if not (self.error1_cb.isChecked() or self.error2_cb.isChecked() or self.error3_cb.isChecked()):
            QMessageBox.warning(self, "Ошибка", "Выберите хотя бы один критерий оценки.")
            return
        self.accept()
    
    def get_selected_criteria(self):
        criteria = []
        if self.error1_cb.isChecked():
            criteria.append(0)
        if self.error2_cb.isChecked():
            criteria.append(1)
        if self.error3_cb.isChecked():
            criteria.append(2)
        return criteria
    
class HolonomicConstraintDialog(QDialog):
    def __init__(self, golonom, matrix_size, max_eig_index, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Голономная связь и выбор переменной")
        self.setGeometry(200, 200, 600, 200)
        self.golonom = golonom
        self.matrix_size = matrix_size
        self.max_eig_index = max_eig_index
        self.selected_index = 0
        
        layout = QVBoxLayout()
        
        holonomic_layout = QHBoxLayout()
        holonomic_label = QLabel(f"Приближенная голономная связь:")
        holonomic_label.setStyleSheet("font-size: 14px; padding: 5px;")
        
        formula_label = QLabel(f"0 = {str(sp.simplify(self.golonom))}")
        formula_label.setStyleSheet("""
            font-family: monospace;
            font-size: 12px;
            padding: 5px;
            background-color: #f6f6f6;
            border: 1px solid #cccccc;
            border-radius: 3px;
        """)
        formula_label.setWordWrap(True)
        formula_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        formula_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        
        holonomic_layout.addWidget(holonomic_label)
        holonomic_layout.addWidget(formula_label)
        
        var_layout = QHBoxLayout()
        var_label = QLabel("Исключаемая переменная:")
        self.var_combo = QComboBox()
        self.var_combo.addItems([f"x{i+1}" for i in range(self.matrix_size)])
        self.var_combo.currentIndexChanged.connect(self.update_selected_index)
        var_layout.addWidget(var_label)
        var_layout.addWidget(self.var_combo)
        var_layout.addStretch()
        
        self.ok_button = QPushButton("ОК")
        self.ok_button.clicked.connect(self.accept)
        
        layout.addLayout(holonomic_layout)
        layout.addLayout(var_layout)
        layout.addStretch()
        layout.addWidget(self.ok_button)
        self.setLayout(layout)
    
    def update_selected_index(self):
        self.selected_index = self.var_combo.currentIndex()
    
    def get_selected_index(self):
        return self.selected_index

class Matrix:
    def __init__(self, n, values, initial_conditions, err, delta_large, iskl_ind=-1):
        self.iskl_ind = iskl_ind
        self.size = n
        self.err = err  # δ⁰ (small delta)
        self.delta_large = delta_large  # Δ⁰ (large delta)
        self.values = values
        self.initial_conditions = initial_conditions.reshape(1, -1) if initial_conditions.ndim == 1 else initial_conditions
        self.eigenvalues = np.zeros(n)
        self.eigenvectors = np.zeros((n, n))
        self.coefficients = np.zeros(n)
        self.gran = []  # Will store min_gran
        self.max_gran = []  # New list for max_gran
        self.matr_with_x = []
        self.matr_resh = np.zeros((n-1, n-1))
        self.inverse = None
        self.index_max_eigval = None
        self.golonom = None

    def safe_exp(self, x):
        return np.exp(x) if x < 700 else np.exp(700)

    def calc_gran(self):
        self.gran = []  # For min_gran using δ⁰
        self.max_gran = []  # For max_gran using Δ⁰
        for i in range(self.size):
            log_term_min = np.log(self.err)  # Using δ⁰
            log_term_max = np.log(self.delta_large)  # Using Δ⁰
            gamma_min = abs((1 / abs(self.eigenvalues[i])) * log_term_min)
            gamma_max = abs((1 / abs(self.eigenvalues[i])) * log_term_max)
            self.gran.append(max(0, gamma_min))
            self.max_gran.append(max(0, gamma_max))

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
        try:
            self.inverse = np.linalg.inv(self.eigenvectors @ diagonal_matrix)
        except np.linalg.LinAlgError:
            self.inverse = None

    def expr_max_exp(self):
        if self.iskl_ind != -1:
            self.index_max_eigval = np.argmax(np.abs(self.eigenvalues))
            eigenvector = self.inverse[self.index_max_eigval, :]
            self.golonom = sum(float(eigenvector[i]) * sp.symbols(f'x{i+1}') 
                            for i in range(self.size))
            excluded_var = self.iskl_ind
            x_resh = sp.symbols(f'x{excluded_var + 1}')
            
            try:
                xi = sp.solve(self.golonom, x_resh)[0]
            except (sp.SympifyError, IndexError):
                raise ValueError(f"Не удалось выразить x{excluded_var + 1} через остальные переменные")
            
            self.matr_with_x = []
            for i in range(self.size):
                temp_expr = sum(self.values[i][j] * sp.symbols(f'x{j + 1}') 
                            for j in range(self.size))
                self.matr_with_x.append(temp_expr)
            
            substituted_equations = [eq.subs(x_resh, xi) for eq in self.matr_with_x]
            simplified_equations = [sp.simplify(eq) for eq in substituted_equations]
            
            current_row = 0
            for i, eq in enumerate(simplified_equations):
                if i == excluded_var:
                    continue
                coefficients = []
                for j in range(self.size):
                    if j == excluded_var:
                        continue
                    coeff = eq.coeff(sp.symbols(f'x{j+1}'))
                    coefficients.append(float(coeff) if coeff != 0 else 0.0)
                if current_row < self.matr_resh.shape[0]:
                    self.matr_resh[current_row, :len(coefficients)] = coefficients
                    current_row += 1
        else:
            self.index_max_eigval = np.argmax(np.abs(self.eigenvalues))
            eigenvector = self.inverse[self.index_max_eigval, :]
            self.golonom = sum(float(eigenvector[i]) * sp.symbols(f'x{i+1}') 
                            for i in range(self.size))

    def calc(self):
        self.calculate_eigenvalues_and_eigenvectors()
        self.calculate_coefficients()
        self.calculate_inverse_from_eigenvectors()
        self.expr_max_exp()
        self.calc_gran()

    def calculate_errors(self, reduced_model, excluded_index, selected_criteria):
        self.calc_gran()
        t_start = max(0, min(self.gran))
        t_end = max(self.gran)

        t = np.linspace(t_start, t_end, 100)
        
        error1_list, error2_list, error3_list = [], [], []

        original_indices = [i for i in range(self.size) if i != excluded_index]
        index_mapping = {orig: idx for idx, orig in enumerate(sorted(original_indices))}

        for orig_idx in original_indices:
            reduced_idx = index_mapping[orig_idx]

            g = np.zeros(len(t))
            k = np.zeros(len(t))

            for i, ti in enumerate(t):
                sum_g = sum(
                    self.coefficients[j] *
                    self.eigenvectors[orig_idx][j] *
                    self.safe_exp(self.eigenvalues[j] * ti)
                    for j in range(self.size)
                )
                g[i] = sum_g

                sum_k = sum(
                    reduced_model.coefficients[j] *
                    reduced_model.eigenvectors[reduced_idx][j] *
                    self.safe_exp(reduced_model.eigenvalues[j] * ti)
                    for j in range(reduced_model.size)
                )
                k[i] = sum_k

            diff = g - k
            n = len(t)

            error1 = np.sqrt((1 / (n - 1)) * (np.sum(diff ** 2) / np.sum(g ** 2))) if np.sum(g ** 2) != 0 else 0.0
            error2 = (np.sum(np.abs(diff)) / n) / np.max(np.abs(g)) if np.max(np.abs(g)) != 0 else 0.0
            error3 = (np.sum(np.abs(diff)) / n) / np.mean(np.abs(g)) if np.mean(np.abs(g)) != 0 else 0.0

            error1_list.append(error1)
            error2_list.append(error2)
            error3_list.append(error3)

        return error1_list, error2_list, error3_list

class MatrixApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Декомпозиция моделей линейных САУ")
        self.setGeometry(100, 100, 800, 600)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.solution_original = None
        self.solution_reduced = None
        self.errors = None
        self.current_size = 0
        self.current_bounds = (0, 0)
        self.entries = []
        self.delta_input = None
        self.selected_criteria = [0, 1, 2]
        self.reduced_matrix_group = None
        self.reduced_matrix_layout = None
        self.reduced_entries = []
        self.holonomic_label = None
        self.interval_label = None
        self.original_matrix = None
        self.initial_conditions = None
        self.eigenvalues = None
        self.golonom = None
        self.placeholder_label = None
        
        self.init_ui()
        
    def init_ui(self):
        # Create main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel for input widgets (matrix, params, criteria, decomp)
        left_panel_container = QWidget()
        left_panel = QVBoxLayout()
        
        # Matrix input group
        matrix_group = QGroupBox("Задание исходной модели")
        matrix_group.setAlignment(Qt.AlignmentFlag.AlignCenter)
        matrix_layout = QVBoxLayout()
        
        radio_layout = QHBoxLayout()
        left_radio_column = QVBoxLayout()
        right_radio_column = QVBoxLayout()
        
        self.input_manual_radio = QRadioButton("Ввести элементы системной матрицы А")
        self.input_file_radio = QRadioButton("Загрузить системную матрицу А из файла")
        self.input_manual_radio.setChecked(True)
        
        self.input_file_radio.toggled.connect(self.toggle_input_mode)
        
        left_radio_column.addWidget(self.input_manual_radio)
        right_radio_column.addWidget(self.input_file_radio)
        radio_layout.addLayout(left_radio_column)
        radio_layout.addLayout(right_radio_column)
        
        
        self.size_input = QSpinBox()
        self.size_input.setRange(2, 1000)
        self.size_input.setValue(3)
        self.size_input.valueChanged.connect(self.update_matrix_inputs)
        
        self.matrix_input_layout = QHBoxLayout()
        matrix_label = QLabel("Матрица А:")
        matrix_label.setFixedWidth(80)
        self.matrix_grid = QGridLayout()
        self.matrix_input_layout.addWidget(matrix_label)
        self.matrix_input_layout.addLayout(self.matrix_grid)
        
        initial_conditions_layout = QVBoxLayout()
        initial_label = QLabel("Введите начальные условия:")
        self.initial_layout = QGridLayout()
        initial_conditions_layout.addWidget(initial_label)
        initial_conditions_layout.addLayout(self.initial_layout)
        
        self.original_model_button = QPushButton("Вывести графики процессов исходной модели")
        self.original_model_button.clicked.connect(self.show_original_model_graphs)
        
        matrix_layout.addLayout(radio_layout)
        matrix_layout.addWidget(QLabel("Порядок модели (n):"))
        matrix_layout.addWidget(self.size_input)
        matrix_layout.addLayout(self.matrix_input_layout)
        matrix_layout.addLayout(initial_conditions_layout)
        matrix_layout.addWidget(self.original_model_button)
        
        matrix_group.setLayout(matrix_layout)
        
        # Parameters group
        params_group = QGroupBox("Задание допустимой погрешности упрощенной модели и точности установления процессов")
        params_group.setAlignment(Qt.AlignmentFlag.AlignCenter)
        params_layout = QVBoxLayout()
        
        error_input_layout = QHBoxLayout()
        
        delta_lower_label = QLabel("δ<sup>0</sup>:")
        delta_lower_label.setTextFormat(Qt.TextFormat.RichText)
        delta_lower_label.setStyleSheet("""
            QLabel {
                font-family: Arial;
                font-size: 14px;
                padding: 5px;
            }
        """)
        
        self.error_input = NoCommaLineEdit()
        validator = QDoubleValidator()
        validator.setLocale(QLocale("C"))
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.error_input.setValidator(validator)
        self.error_input.setFixedWidth(100)
        self.error_input.setText("0.1")
        
        error_input_layout.addWidget(delta_lower_label)
        error_input_layout.addWidget(self.error_input)
        
        error_input_layout.addSpacing(20)
        
        delta_upper_label = QLabel("Δ<sup>0</sup>:")
        delta_upper_label.setTextFormat(Qt.TextFormat.RichText)
        delta_upper_label.setStyleSheet("""
            QLabel {
                font-family: Arial;
                font-size: 14px;
                padding: 5px;
            }
        """)
        
        self.delta_input = NoCommaLineEdit()
        self.delta_input.setValidator(validator)
        self.delta_input.setFixedWidth(100)
        self.delta_input.setText("0.1")
        
        error_input_layout.addWidget(delta_upper_label)
        error_input_layout.addWidget(self.delta_input)
        error_input_layout.addStretch()
        
        params_layout.addLayout(error_input_layout)
        params_group.setLayout(params_layout)
        
        # Criteria group
        criteria_group = QGroupBox("Выбор критерия оценивания близости процессов исходной и упрощенной модели.")
        criteria_group.setAlignment(Qt.AlignmentFlag.AlignCenter)
        criteria_layout = QVBoxLayout()
        
        self.criteria_button = QPushButton("Выбрать критерий")
        self.criteria_button.clicked.connect(self.show_criteria_selection)
        self.criteria_button.setStyleSheet("""
            QPushButton {
                padding: 8px;
                font-size: 14px;
                border: 1px solid #cccccc;
                border-radius: 5px;
                background-color: #f6f6f6;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """)
        
        criteria_layout.addWidget(self.criteria_button)
        criteria_layout.addStretch()
        criteria_group.setLayout(criteria_layout)
        
        # Decomposition group
        decomp_group = QGroupBox("Декомпозиция исходной модели")
        decomp_group.setAlignment(Qt.AlignmentFlag.AlignCenter)
        decomp_layout = QVBoxLayout()
        
        self.calc_button = QPushButton("Осуществить декомпозицию модели")
        self.calc_button.clicked.connect(self.perform_calculations)
        self.calc_button.setStyleSheet("""
            QPushButton {
                padding: 8px;
                font-size: 14px;
                border: 1px solid #cccccc;
                border-radius: 5px;
                background-color: #f6f6f6;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """)
        decomp_layout.addWidget(self.calc_button)
        decomp_layout.addStretch()
        decomp_group.setLayout(decomp_layout)
        
        # Add input widgets to left panel
        left_panel.addWidget(matrix_group)
        left_panel.addWidget(params_group)
        left_panel.addWidget(criteria_group)
        left_panel.addWidget(decomp_group)
        left_panel.addStretch()
        left_panel_container.setLayout(left_panel)
        
        # Wrap left panel in a scroll area
        left_scroll_area = QScrollArea()
        left_scroll_area.setWidgetResizable(True)
        left_scroll_area.setWidget(left_panel_container)
        left_scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: 1px solid #cccccc;
                background: #f6f6f6;
                width: 12px;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:vertical {
                background: #888888;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # Right panel for "Результаты декомпозиции"
        right_panel_container = QWidget()
        right_panel = QVBoxLayout()
        
        # Add reduced matrix group to the right panel
        self.reduced_matrix_group = QGroupBox("Результаты декомпозиции")
        self.reduced_matrix_group.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.reduced_matrix_layout = QVBoxLayout()
        
        self.placeholder_label = QLabel()
        self.placeholder_label.setStyleSheet("""
            font-family: Arial;
            font-size: 14px;
            padding: 10px;
            color: #666666;
        """)
        self.reduced_matrix_layout.addWidget(self.placeholder_label)
        
        self.reduced_matrix_group.setLayout(self.reduced_matrix_layout)
        right_panel.addWidget(self.reduced_matrix_group)
        right_panel.addStretch()  # Add stretch to push content to the top
        right_panel_container.setLayout(right_panel)
        
        # Wrap right panel in a scroll area
        right_scroll_area = QScrollArea()
        right_scroll_area.setWidgetResizable(True)
        right_scroll_area.setWidget(right_panel_container)
        right_scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: 1px solid #cccccc;
                background: #f6f6f6;
                width: 12px;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:vertical {
                background: #888888;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # Add scroll areas to splitter (left panel first, then right)
        left_scroll_area.setMinimumWidth(400)
        right_scroll_area.setMinimumWidth(300)
        
        splitter.addWidget(left_scroll_area)
        splitter.addWidget(right_scroll_area)
        splitter.setSizes([600, 400])  # Adjusted to give more space to left panel
        
        # Set main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)
        self.central_widget.setLayout(main_layout)
        
        # Initialize right panel for dynamic content
        self.right_panel = right_panel
        
        self.update_matrix_inputs()
        self.toggle_input_mode()

    
    def show_criteria_selection(self):
        dialog = CriteriaSelectionDialog(self)
        if dialog.exec():
            self.selected_criteria = dialog.get_selected_criteria()

    def update_matrix_inputs(self):
        n = self.size_input.value()
        
        for i in reversed(range(self.matrix_grid.count())):
            self.matrix_grid.itemAt(i).widget().deleteLater()
        self.entries.clear()
        
        for i in reversed(range(self.initial_layout.count())):
            self.initial_layout.itemAt(i).widget().deleteLater()
        
        validator = QDoubleValidator()
        validator.setLocale(QLocale("C"))
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        
        self.entries = []
        for i in range(n):
            row = []
            for j in range(n):
                edit = NoCommaLineEdit()
                edit.setFixedWidth(60)
                edit.setValidator(validator)
                edit.setText("0")
                self.matrix_grid.addWidget(edit, i, j)
                row.append(edit)
            self.entries.append(row)
        
        for i in range(n):
            label = QLabel(f"x{i+1}(0):")
            edit = NoCommaLineEdit()
            edit.setValidator(validator)
            edit.setText("0")
            self.initial_layout.addWidget(label, i, 0)
            self.initial_layout.addWidget(edit, i, 1)
        


    def toggle_input_mode(self):
        is_file_input = self.input_file_radio.isChecked()
        self.size_input.setEnabled(not is_file_input)
        if is_file_input:
            self.load_from_file()
        else:
            self.update_matrix_inputs()

    def load_from_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Открыть файл матрицы", "", "Текстовые файлы (*.txt)")
        
        if not file_name:
            return
        
        try:
            with open(file_name, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                if not lines:
                    raise ValueError("Файл пуст")
                
                matrix_data = []
                for i, line in enumerate(lines):
                    values = line.split()
                    if not values:
                        continue
                    try:
                        row = [float(v) for v in values]
                        matrix_data.append(row)
                    except ValueError:
                        raise ValueError(f"Некорректное значение в строке {i+1}: {line}")
                
                n = len(matrix_data)
                if not all(len(row) == n for row in matrix_data):
                    raise ValueError("Матрица должна быть квадратной")
                
                self.size_input.setValue(n)
                self.update_matrix_inputs()
                
                for i in range(n):
                    for j in range(n):
                        self.entries[i][j].setText(str(matrix_data[i][j]))
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл: {str(e)}")

    def update_reduced_matrix_display(self, reduced_matrix, golonom, min_gran, max_gran):
        # Helper function to recursively clear a layout
        def clear_layout(layout):
            if layout is not None:
                while layout.count():
                    item = layout.takeAt(0)
                    widget = item.widget()
                    if widget:
                        widget.deleteLater()
                    sub_layout = item.layout()
                    if sub_layout:
                        clear_layout(sub_layout)
                    del item

        # Clear all existing content in reduced_matrix_layout
        clear_layout(self.reduced_matrix_layout)
        self.reduced_entries.clear()

        # Recreate the holonomic constraint label
        self.holonomic_label = QLabel(f"Приближённая голономная связь: 0 = {str(sp.simplify(golonom))}")
        self.holonomic_label.setStyleSheet("""
            font-family: monospace;
            font-size: 12px;
            padding: 5px;
            background-color: #f6f6f6;
            border: 1px solid #cccccc;
            border-radius: 3px;
        """)
        self.holonomic_label.setWordWrap(True)
        self.holonomic_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Add section title for reduced model parameters
        reduced_model_title = QLabel("Упрощённая модель и её параметры")
        reduced_model_title.setStyleSheet("""
            font-family: Arial;
            font-size: 14px;
            font-weight: bold;
            padding: 5px;
        """)

        # Display the order of the reduced matrix
        n = reduced_matrix.shape[0]
        order_label = QLabel(f"Порядок модели (n): {n}")
        order_label.setStyleSheet("""
            font-family: Arial;
            font-size: 12px;
            padding: 5px;
        """)

        # Recreate the reduced matrix layout
        self.reduced_matrix_input_layout = QHBoxLayout()
        reduced_matrix_label = QLabel("Матрица Ā:")
        reduced_matrix_label.setFixedWidth(80)
        self.reduced_matrix_grid = QGridLayout()
        self.reduced_matrix_input_layout.addWidget(reduced_matrix_label)
        self.reduced_matrix_input_layout.addLayout(self.reduced_matrix_grid)

        validator = QDoubleValidator()
        validator.setLocale(QLocale("C"))
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)

        self.reduced_entries = []
        for i in range(n):
            row = []
            for j in range(n):
                value_str = f"{reduced_matrix[i][j]:.6f}".rstrip('0').rstrip('.')
                edit = QLineEdit(value_str)
                edit.setReadOnly(True)
                edit.setStyleSheet("""
                    QLineEdit {
                        padding: 5px;
                        border: 1px solid #cccccc;
                        border-radius: 3px;
                        font-family: monospace;
                        font-size: 12px;
                    }
                """)
                font_metrics = QFontMetrics(edit.font())
                text_width = font_metrics.boundingRect(value_str).width() + 10
                edit.setFixedWidth(max(100, text_width))
                self.reduced_matrix_grid.addWidget(edit, i, j)
                row.append(edit)
            self.reduced_entries.append(row)

        # Recreate the interval label
        self.interval_label = QLabel(f"Интервал справедливости: [{min_gran:.2f}, {max_gran:.2f}] с")
        self.interval_label.setStyleSheet("""
            font-family: monospace;
            font-size: 12px;
            padding: 5px;
            background-color: #f6f6f6;
            border: 1px solid #cccccc;
            border-radius: 3px;
        """)
        self.interval_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Calculate efficiency criterion γ = (Γn - Γ1) / Γn
        if max_gran != 0:
            gamma = (max_gran - min_gran) / max_gran
        else:
            gamma = 0.0

        # Display efficiency criterion
        efficiency_label = QLabel(f"Критерий эффективности (γ): {gamma:.6f}")
        efficiency_label.setStyleSheet("""
            font-family: monospace;
            font-size: 12px;
            padding: 5px;
            background-color: #f6f6f6;
            border: 1px solid #cccccc;
            border-radius: 3px;
        """)
        efficiency_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Define a common button stylesheet
        button_stylesheet = """
            QPushButton {
                padding: 10px;
                font-size: 14px;
                border: 1px solid #cccccc;
                border-radius: 5px;
                background-color: #f6f6f6;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """

        # Recreate buttons with consistent minimum width
        self.solution_button = QPushButton("Графики решений")
        self.solution_button.clicked.connect(self.show_transient)
        self.solution_button.setMinimumWidth(250)
        self.solution_button.setStyleSheet(button_stylesheet)

        self.deviation_button = QPushButton("Графики отклонений")
        self.deviation_button.clicked.connect(self.show_errors)
        self.deviation_button.setMinimumWidth(250)
        self.deviation_button.setStyleSheet(button_stylesheet)

        self.proximity_button = QPushButton("Оценки близости процессов в исходной и упрощенной моделях")
        self.proximity_button.setMinimumWidth(250)
        self.proximity_button.setStyleSheet(button_stylesheet)
        self.proximity_button.clicked.connect(self.show_quality)

        self.save_matrix_button = QPushButton("Сохранить результаты декомпозиции в файл")
        self.save_matrix_button.clicked.connect(self.save_reduced_matrix)
        self.save_matrix_button.setMinimumWidth(250)
        self.save_matrix_button.setStyleSheet(button_stylesheet)

        # Add new button for saving error calculation parameters
        self.save_error_params_button = QPushButton("Сохранить параметры для расчета ошибки")
        self.save_error_params_button.clicked.connect(self.save_error_calculation_params)
        self.save_error_params_button.setMinimumWidth(250)
        self.save_error_params_button.setStyleSheet(button_stylesheet)

        # Add new widgets to the layout in the specified order
        self.reduced_matrix_layout.addWidget(self.holonomic_label)
        self.reduced_matrix_layout.addWidget(reduced_model_title)
        self.reduced_matrix_layout.addWidget(order_label)
        self.reduced_matrix_layout.addLayout(self.reduced_matrix_input_layout)
        self.reduced_matrix_layout.addWidget(self.interval_label)
        self.reduced_matrix_layout.addWidget(efficiency_label)
        self.reduced_matrix_layout.addWidget(self.solution_button)
        self.reduced_matrix_layout.addWidget(self.deviation_button)
        self.reduced_matrix_layout.addWidget(self.proximity_button)
        self.reduced_matrix_layout.addWidget(self.save_matrix_button)
        self.reduced_matrix_layout.addWidget(self.save_error_params_button)

    def save_error_calculation_params(self):
        if not self.solution_original or not self.solution_reduced:
            QMessageBox.warning(self, "Ошибка", "Решения для исходной или упрощенной модели не рассчитаны")
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Сохранить параметры для расчета ошибки", "", "Текстовые файлы (*.txt)"
        )

        if not file_name:
            return

        try:
            # Get time points and solutions
            t = self.solution_original.t
            original_vars = self.solution_original.y  # Shape: (n, len(t))
            reduced_vars = self.solution_reduced.y    # Shape: (n-1, len(t))
            n = self.current_size
            excluded_index = self.excluded_index

            # Prepare headers
            headers = ["t"]
            # Add headers for original model variables (x1, x2, ...)
            for i in range(n):
                headers.append(f"x{i+1}")
            # Add headers for reduced model variables (x1_упр, ...)
            original_indices = [i for i in range(n) if i != excluded_index]
            for i in original_indices:
                headers.append(f"x{i+1}_упр")

            # Prepare data
            data = []
            for i in range(len(t)):
                row = [t[i]]
                # Add values from the original model
                for j in range(n):
                    row.append(original_vars[j][i])
                # Add values from the reduced model
                for j in range(len(original_indices)):
                    row.append(reduced_vars[j][i])
                data.append(row)

            # Format the table
            col_width = 12
            # Format values to 6 decimal places, remove trailing zeros
            formatted_data = [[f"{val:.6f}".rstrip('0').rstrip('.') for val in row] for row in data]

            # Create the table string
            # Header row
            header_str = ""
            for i, header in enumerate(headers):
                header_str += f" {header:<{col_width-2}} "
                if i < len(headers) - 1:
                    header_str += "|"

            # Separator line
            separator = "-" * (col_width * len(headers) + len(headers) - 1)

            # Data rows
            rows = []
            for i in range(len(data)):
                row_str = ""
                for j in range(len(headers)):
                    row_str += f" {formatted_data[i][j]:<{col_width-2}} "
                    if j < len(headers) - 1:
                        row_str += "|"
                rows.append(row_str)

            # Combine all parts
            table = [separator, header_str, separator]
            table.extend(rows)
            table.append(separator)

            # Write to file
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write("Параметры для расчета ошибки\n")
                f.write("=" * 50 + "\n\n")
                f.write("\n".join(table) + "\n")

            QMessageBox.information(self, "Успех", "Параметры для расчета ошибки успешно сохранены")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл: {str(e)}")

    def format_table(self, headers, data, row_labels=None, col_width=10):
        """
        Format data into a table string with dashed borders, similar to the provided example.
        headers: List of column headers.
        data: 2D list or numpy array of values.
        row_labels: Optional list of row labels.
        col_width: Width of each column.
        """
        n_cols = len(headers)
        n_rows = len(data)
        
        # Ensure data is a list of lists
        data = [[float(val) for val in row] for row in data]
        
        # Format values to 6 decimal places, remove trailing zeros
        formatted_data = [[f"{val:.6f}".rstrip('0').rstrip('.') for val in row] for row in data]
        
        # Adjust column width if row labels are provided
        label_width = max([len(str(label)) for label in row_labels]) + 2 if row_labels else 6
        col_widths = [label_width] + [col_width] * n_cols if row_labels else [col_width] * n_cols
        
        # Header row
        header_str = " " * col_widths[0] + "|"
        for i, header in enumerate(headers):
            header_str += f" {header:^{col_widths[i+1]-2}} |"
        
        # Separator line
        separator = "-" * col_widths[0] + "+"
        for width in col_widths[1:]:
            separator += "-" * (width - 1) + "+"
        
        # Data rows
        rows = []
        for i in range(n_rows):
            row_str = ""
            if row_labels:
                row_str += f" {row_labels[i]:<{col_widths[0]-2}} |"
            else:
                row_str += f" {'':<{col_widths[0]-2}} |"
            for j in range(n_cols):
                row_str += f" {formatted_data[i][j]:>{col_widths[j+1]-2}} |"
            rows.append(row_str)
        
        # Combine all parts
        table = [separator, header_str, separator]
        table.extend(rows)
        table.append(separator)
        
        return "\n".join(table)

    def save_reduced_matrix(self):
        if not self.reduced_entries or self.original_matrix is None:
            QMessageBox.warning(self, "Ошибка", "Результаты декомпозиции не сформированы")
            return
        
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результаты декомпозиции", "", "Текстовые файлы (*.txt)")
        
        if not file_name:
            return
        
        try:
            n = self.current_size
            n_red = len(self.reduced_entries)
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write("Результаты декомпозиции моделей линейных САУ\n")
                f.write("=" * 50 + "\n\n")
                
                # 1. Исходная система
                f.write("1. Исходная система\n\n")
                f.write(f"1.1. Системная матрица А (порядок n = {n}):\n")
                headers = [f"Колонка {j+1}" for j in range(n)]
                row_labels = [f"Строка {i+1}" for i in range(n)]
                matrix_table = self.format_table(headers, self.original_matrix, row_labels, col_width=12)
                f.write(matrix_table + "\n\n")
                
                # 2. Начальные условия
                f.write("1.2. Начальные условия:\n")
                initial_data = [[self.initial_conditions[i]] for i in range(n)]
                headers = ["Значение"]
                row_labels = [f"x{i+1}(0)" for i in range(n)]
                initial_table = self.format_table(headers, initial_data, row_labels, col_width=12)
                f.write(initial_table + "\n\n")
                
                # 3. Собственные числа
                f.write("1.3. Собственные числа исходной матрицы:\n")
                eig_data = [[self.eigenvalues[i]] for i in range(n)]
                headers = ["Значение"]
                row_labels = [f"λ{i+1}" for i in range(n)]
                eig_table = self.format_table(headers, eig_data, row_labels, col_width=12)
                f.write(eig_table + "\n\n")
                
                # 4. Голономная связь
                f.write("1.4. Приближенная голономная связь:\n")
                f.write(f"0 = {str(sp.simplify(self.golonom))}\n\n")
                
                # Разделитель
                f.write("2. Результаты декомпозиции\n")
                f.write("-" * 50 + "\n\n")
                
                # 5. Упрощенная модель
                f.write(f"2.1. Системная матрица Ā (порядок n = {n_red}):\n")
                reduced_matrix = [[float(self.reduced_entries[i][j].text()) for j in range(n_red)] for i in range(n_red)]
                headers = [f"Колонка {j+1}" for j in range(n_red)]
                row_labels = [f"Строка {i+1}" for i in range(n_red)]
                reduced_table = self.format_table(headers, reduced_matrix, row_labels, col_width=12)
                f.write(reduced_table + "\n\n")
                
                # 6. Интервал справедливости
                f.write("2.2. Интервал справедливости:\n")
                interval_data = [[self.current_bounds[0], self.current_bounds[1]]]
                headers = ["Начало", "Конец"]
                row_labels = ["Интервал"]
                interval_table = self.format_table(headers, interval_data, row_labels, col_width=12)
                f.write(interval_table + "\n\n")
                
                # 7. Ошибки моделирования
                f.write("2.3. Ошибки моделирования:\n")
                original_indices = [i for i in range(n) if i != self.excluded_index]
                criteria_names = [
                    "Оценка типа I",
                    "Оценка типа II",
                    "Оценка типа III"
                ]
                selected_errors = []
                error_labels = []
                for idx in self.selected_criteria:
                    selected_errors.append([self.errors[idx][i] for i in range(len(original_indices))])
                    error_labels.append(criteria_names[idx])
                
                if selected_errors:
                    error_data = list(zip(*selected_errors))
                    headers = error_labels
                    row_labels = [f"x{i+1}" for i in original_indices]
                    error_table = self.format_table(headers, error_data, row_labels, col_width=12)
                    f.write(error_table + "\n")
                
            QMessageBox.information(self, "Успех", "Результаты декомпозиции успешно сохранены")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл: {str(e)}")

    def perform_calculations(self):
        try:
            self.solution_original = None
            self.solution_reduced = None
            self.errors = None
            self.current_size = 0
            self.current_bounds = (0, 0)
            
            n = self.size_input.value()
            matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    try:
                        matrix[i][j] = float(self.entries[i][j].text())
                    except ValueError:
                        matrix[i][j] = 0.0
            
            initial = np.zeros(n)
            for i in range(n):
                try:
                    initial[i] = float(self.initial_layout.itemAt(i*2+1).widget().text())
                except (AttributeError, ValueError):
                    initial[i] = 0.0

            try:
                error = max(1e-6, float(self.error_input.text()))  # δ⁰
                delta_large = max(1e-6, float(self.delta_input.text()))  # Δ⁰
            except ValueError:
                error = 0.1
                delta_large = 0.1

            if np.linalg.cond(matrix) > 1e12:
                raise ValueError("Матрица плохо обусловлена")

            orig_system = Matrix(n, matrix, initial, error, delta_large, -1)
            orig_system.calc()
            
            if not orig_system.gran or not orig_system.max_gran:
                raise ValueError("Не удалось рассчитать границы справедливости")
                    
            min_gran = min(orig_system.gran)
            max_gran = max(orig_system.max_gran)
            
            t_span = (0, min_gran)
            solution = solve_ivp(
                lambda t, y, matrix: matrix @ y,
                t_span,
                initial,
                args=(matrix,),
                dense_output=True,
                method='LSODA'
            )
            
            if not solution.success:
                raise ValueError(f"Моделирование не удалось: {solution.message}")
            
            initial_conditions_new = None
            if solution.sol is not None:
                initial_conditions_new = solution.sol(min_gran)
            else:
                tolerance = max(0.1, 0.05 * min_gran)
                closest_idx = np.argmin(np.abs(solution.t - min_gran))
                if abs(solution.t[closest_idx] - min_gran) <= tolerance:
                    initial_conditions_new = solution.y[:, closest_idx]
                else:
                    raise ValueError(
                        f"Не удалось найти начальные условия в интервале "
                        f"[{min_gran - tolerance:.2f}, {min_gran + tolerance:.2f}]"
                    )
            
            dialog = HolonomicConstraintDialog(orig_system.golonom, n, orig_system.index_max_eigval, self)
            if not dialog.exec():
                QMessageBox.information(self, "Отмена", "Декомпозиция отменена")
                return
            
            excluded_indices = dialog.get_selected_index()
            self.excluded_index = excluded_indices
            orig_system_new = Matrix(n, matrix, initial_conditions_new, error, delta_large, excluded_indices)
            orig_system_new.calc()
            
            initial_conditions_red = np.delete(initial_conditions_new, excluded_indices)
            reduced_system = Matrix(n-1, orig_system_new.matr_resh, initial_conditions_red, error, delta_large, -1)
            reduced_system.calc()
            
            t_span_new = (min_gran, max_gran)  # Updated to use max_gran from Δ⁰
            t_eval = np.linspace(min_gran, max_gran, 500)
            
            self.solution_original = solve_ivp(
                lambda t, y, matrix: matrix @ y,
                t_span_new,
                initial_conditions_new,
                args=(matrix,),
                t_eval=t_eval,
                method='LSODA'
            )
            
            self.solution_reduced = solve_ivp(
                lambda t, y, matrix: matrix @ y,
                t_span_new,
                initial_conditions_red,
                args=(reduced_system.values,),
                t_eval=t_eval,
                method='LSODA'
            )
            
            if not self.solution_original.success or not self.solution_reduced.success:
                raise ValueError("Моделирование упрощенной или исходной системы не удалось")
            
            self.errors = orig_system_new.calculate_errors(reduced_system, excluded_indices, self.selected_criteria)
            
            self.current_size = n
            self.current_bounds = (min_gran, max_gran)
            self.original_matrix = matrix
            self.initial_conditions = initial
            self.eigenvalues = orig_system.eigenvalues
            self.golonom = orig_system.golonom
            
            # Ensure the display is updated rather than creating a new window
            self.update_reduced_matrix_display(reduced_system.values, orig_system.golonom, min_gran, max_gran)
            
        except Exception as e:
            self.solution_original = None
            self.solution_reduced = None
            self.errors = None
            self.current_size = 0
            self.current_bounds = (0, 0)
            self.original_matrix = None
            self.initial_conditions = None
            self.eigenvalues = None
            self.golonom = None
            QMessageBox.critical(self, "Ошибка", f"Не удалось выполнить расчет: {str(e)}")
    def show_transient(self):
        if self.solution_original and self.solution_reduced and hasattr(self, 'current_bounds'):
            t_start, t_end = self.current_bounds
            choice_dialog = GraphChoiceDialog(self)
            if choice_dialog.exec():
                choice = choice_dialog.get_choice()
                window = TransientResponseWindow("Графики решений", self)
                window.plot(
                    self.solution_original,
                    self.solution_reduced,
                    choice,
                    self.current_size,
                    self.current_size-1,
                    t_start,
                    t_end
                )
                window.exec()

    def show_errors(self):
        if self.solution_original and self.solution_reduced:
            window = ErrorResponseWindow("Графики отклонений", self)
            window.plot(self.solution_original, self.solution_reduced, self.current_size, self.excluded_index)
            window.exec()

    def show_quality(self):
        if self.errors and self.selected_criteria:
            dialog = QualityMetricsDialog(self.errors, self.selected_criteria, self.current_size, self.excluded_index, self)
            dialog.exec()
        else:
            QMessageBox.warning(self, "Ошибка", "Оценки близости не рассчитаны или не выбраны критерии.")

    def show_original_model_graphs(self):
        try:
            n = self.size_input.value()
            
            matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    text = self.entries[i][j].text().strip()
                    if not text:
                        raise ValueError(f"Поле матрицы A[{i+1},{j+1}] пустое")
                    try:
                        value = float(text.replace(",", "."))
                        if abs(value) > 1e6:
                            raise ValueError(f"Слишком большое значение в поле матрицы A[{i+1},{j+1}]: {text}")
                        matrix[i][j] = value
                    except ValueError:
                        raise ValueError(f"Некорректное значение в поле матрицы A[{i+1},{j+1}]: {text}")
            
            if np.all(matrix == 0):
                initial = np.zeros(n)
                for i in range(n):
                    text = self.initial_layout.itemAt(i*2+1).widget().text().strip()
                    if not text:
                        raise ValueError(f"Поле начального условия x{i+1}(0) пустое")
                    try:
                        value = float(text.replace(",", "."))
                        if abs(value) > 1e6:
                            raise ValueError(f"Слишком большое значение в поле начального условия x{i+1}(0): {text}")
                        initial[i] = value
                    except ValueError:
                        raise ValueError(f"Некорректное значение в поле начального условия x{i+1}(0)")
                if np.all(initial == 0):
                    raise ValueError("Матрица и начальные условия полностью нулевые. График не будет отображаться, так как система не имеет динамики.")
                else:
                    raise ValueError("Матрица полностью нулевая. График не будет отображаться, так как система не имеет динамики.")
            
            try:
                cond_number = np.linalg.cond(matrix)
                if cond_number > 1e12:
                    raise ValueError(f"Матрица плохо обусловлена (число обусловленности: {cond_number:.2e})")
            except np.linalg.LinAlgError as e:
                raise ValueError("Не удалось вычислить число обусловленности матрицы")
            
            initial = np.zeros(n)
            for i in range(n):
                text = self.initial_layout.itemAt(i*2+1).widget().text().strip()
                if not text:
                    raise ValueError(f"Поле начального условия x{i+1}(0) пустое")
                try:
                    value = float(text.replace(",", "."))
                    if abs(value) > 1e6:
                        raise ValueError(f"Слишком большое значение в поле начального условия x{i+1}(0): {text}")
                    initial[i] = value
                except ValueError:
                    raise ValueError(f"Некорректное значение в поле начального условия x{i+1}(0)")
            
            time_dialog = TimeBoundaryDialog(self)
            if not time_dialog.exec():
                QMessageBox.information(self, "Отмена", "Ввод времени отменен")
                return
            t_end = time_dialog.get_time()
            if t_end <= 0:
                raise ValueError("Правая граница времени должна быть положительной")
            if t_end > 1e6:
                raise ValueError(f"Слишком большое время моделирования: {t_end}")
            
            t_span = (0, t_end)
            t_eval = np.linspace(0, t_end, 500)
            
            try:
                solution = solve_ivp(
                    lambda t, y: matrix @ y,
                    t_span,
                    initial,
                    t_eval=t_eval,
                    method='LSODA',
                    rtol=1e-6,
                    atol=1e-8
                )
                if not solution.success:
                    raise ValueError(f"Моделирование не удалось: {solution.message}")
                if solution.y is None or solution.y.size == 0:
                    raise ValueError("Моделирование не вернуло данных для графика")
            except Exception as e:
                raise ValueError(f"Ошибка при выполнении solve_ivp: {str(e)}")
            
            window = TransientResponseWindow("Графики процессов исходной модели", self)
            window.plot(solution, None, 0, n, 0, 0, t_end)
            if not window.exec():
                QMessageBox.information(self, "Информация", "Окно графика закрыто")
            
        except ValueError as e:
            QMessageBox.critical(self, "Ошибка ввода", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось построить графики: {str(e)}")
    
if __name__ == "__main__":
    QLocale.setDefault(QLocale("C"))
    app = QApplication(sys.argv)
    window = MatrixApp()
    window.show()
    window.showMaximized()
    sys.exit(app.exec())
