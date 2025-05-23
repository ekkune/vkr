from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QMainWindow, QSpinBox, QHBoxLayout, QDialog, QFileDialog,
    QGroupBox, QTextEdit, QTableWidgetItem, QTableWidget, QRadioButton, 
    QMessageBox, QComboBox, QScrollArea, QCheckBox, QSizePolicy,QSplitter
)
from PyQt6.QtGui import QPalette, QColor, QPen, QDoubleValidator, QFontMetrics
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PyQt6.QtCore import Qt, QLocale 
import sys
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt

# Настройка Matplotlib для использования MathText
plt.rc('text', usetex=False)
plt.rc('font', family='Arial', size=12)

class TimeBoundaryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выбор правой границы времени")
        self.layout = QVBoxLayout()
        
        self.time_label = QLabel("Введите правую границу времени (с):")
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

class ChartWindow(QDialog):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setGeometry(100, 100, 800, 600)
        self.chart_view = QChartView()
        self.chart_view.setRubberBand(QChartView.RubberBand.RectangleRubberBand)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.chart_view)
        self.setLayout(self.layout)

class TransientResponseWindow(ChartWindow):
    def plot(self, solution1, solution2, choice, size1, size2, t_start, t_end):
        chart = QChart()
        chart.setTitle(f'Переходные процессы [{t_start:.2f} - {t_end:.2f} с]')
        
        mask1 = (solution1.t >= t_start) & (solution1.t <= t_end)
        y_values = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        if choice in [0, 2]:
            for i in range(size1):
                series = QLineSeries()
                series.setName(f'Исходная x{i+1}')
                series.setColor(QColor(colors[i % len(colors)]))
                y = solution1.y[i][mask1]
                x = solution1.t[mask1]
                y_values.extend(y)
                for j in range(len(x)):
                    series.append(x[j], y[j])
                chart.addSeries(series)
        
        if choice in [1, 2] and solution2 is not None:
            mask2 = (solution2.t >= t_start) & (solution2.t <= t_end)
            for i in range(size2):
                series = QLineSeries()
                series.setName(f'Пониженная x{i+1}')
                series.setColor(QColor(colors[(i+1) % len(colors)]))
                series.setPen(QPen(Qt.PenStyle.DashLine))
                y = solution2.y[i][mask2]
                x = solution2.t[mask2]
                y_values.extend(y)
                for j in range(len(x)):
                    series.append(x[j], y[j])
                chart.addSeries(series)
        
        self.setup_axes(chart, t_start, t_end, y_values)
        self.chart_view.setChart(chart)

    def setup_axes(self, chart, t_start, t_end, y_values=None):
        axis_x = QValueAxis()
        axis_x.setTitleText("Время (с)")
        axis_x.setRange(t_start, t_end)
        axis_x.setLabelFormat("%.2f")
        
        axis_y = QValueAxis()
        axis_y.setTitleText("Значение")
        
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
            
            # Интерполируем решение упрощённой модели на временную сетку исходной
            interp_func = interp1d(t2, solution2.y[reduced_idx], kind='linear', fill_value="extrapolate")
            interp_values = interp_func(t1)
            errors = solution1.y[orig_idx] - interp_values

            for j in range(len(t1)):
                error_series.append(t1[j], errors[j])
            
            chart.addSeries(error_series)

        axis_x = QValueAxis()
        axis_x.setTitleText("Время (с)")
        axis_x.setLabelFormat("%.2f")

        axis_y = QValueAxis()
        axis_y.setTitleText("Ошибка")
        axis_y.setLabelFormat("%.6f")  # повысим точность отображения

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
        criteria_names = [
            "Ошибка типа I (СКО)",
            "Ошибка типа II (Макс. ошибка)",
            "Ошибка типа III (Средняя ошибка)"
        ]
        original_indices = [idx for idx in range(full_size) if idx != excluded_index]

        for i, (e1, e2, e3) in enumerate(zip(errors[0], errors[1], errors[2])):
            orig_var_index = original_indices[i]
            error_text += f"<b>Переменная x{orig_var_index + 1}:</b><br>"
            if 0 in selected_criteria:
                error_text += f"• {criteria_names[0]}: {e1:.6f}<br>"
            if 1 in selected_criteria:
                error_text += f"• {criteria_names[1]}: {e2:.6f}<br>"
            if 2 in selected_criteria:
                error_text += f"• {criteria_names[2]}: {e3:.6f}<br>"
            error_text += "<br>"
        
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
        self.setGeometry(200, 200, 600, 300)
        
        layout = QVBoxLayout()
        
        error1_layout = QHBoxLayout()
        self.error1_cb = QCheckBox("Ошибка типа I (СКО)")
        self.error1_cb.setChecked(True)
        fig1 = Figure(figsize=(3, 0.8), tight_layout=True)
        canvas1 = FigureCanvasQTAgg(fig1)
        ax1 = fig1.add_subplot(111)
        ax1.text(0.1, 0.5, r'$\varepsilon_1 = \sqrt{\frac{\sum |\delta(t)|^2}{\sum |g(t)|^2}}$',
                 fontsize=12, verticalalignment='center')
        ax1.axis('off')
        error1_layout.addWidget(self.error1_cb)
        error1_layout.addWidget(canvas1)
        error1_layout.addStretch()
        
        error2_layout = QHBoxLayout()
        self.error2_cb = QCheckBox("Ошибка типа II (Макс. ошибка)")
        self.error2_cb.setChecked(True)
        fig2 = Figure(figsize=(2.5, 0.8), tight_layout=True)
        canvas2 = FigureCanvasQTAgg(fig2)
        ax2 = fig2.add_subplot(111)
        ax2.text(0.1, 0.5, r'$\varepsilon_2 = \frac{\max |\delta(t)|}{\max |g(t)|}$',
                 fontsize=12, verticalalignment='center')
        ax2.axis('off')
        error2_layout.addWidget(self.error2_cb)
        error2_layout.addWidget(canvas2)
        error2_layout.addStretch()
        
        error3_layout = QHBoxLayout()
        self.error3_cb = QCheckBox("Ошибка типа III (Средняя ошибка)")
        self.error3_cb.setChecked(True)
        fig3 = Figure(figsize=(2.5, 0.8), tight_layout=True)
        canvas3 = FigureCanvasQTAgg(fig3)
        ax3 = fig3.add_subplot(111)
        ax3.text(0.1, 0.5, r'$\varepsilon_3 = \frac{\text{mean} |\delta(t)|}{\text{mean} |g(t)|}$',
                 fontsize=12, verticalalignment='center')
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
        
        # Голономная связь
        holonomic_layout = QHBoxLayout()
        holonomic_label = QLabel(f"Голономная связь (на основе λ_{self.max_eig_index+1}):")
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
        
        # Выбор исключаемой переменной
        var_layout = QHBoxLayout()
        var_label = QLabel("Исключаемая переменная:")
        self.var_combo = QComboBox()
        self.var_combo.addItems([f"x{i+1}" for i in range(self.matrix_size)])
        self.var_combo.currentIndexChanged.connect(self.update_selected_index)
        var_layout.addWidget(var_label)
        var_layout.addWidget(self.var_combo)
        var_layout.addStretch()
        
        # Кнопка ОК
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
    def __init__(self, n, values, initial_conditions, err, iskl_ind=-1):
        self.iskl_ind = iskl_ind
        self.size = n
        self.err = err
        self.values = values
        self.initial_conditions = initial_conditions.reshape(1, -1) if initial_conditions.ndim == 1 else initial_conditions
        self.eigenvalues = np.zeros(n)
        self.eigenvectors = np.zeros((n, n))
        self.coefficients = np.zeros(n)
        self.gran = []
        self.matr_with_x = []
        self.matr_resh = np.zeros((n-1, n-1))
        self.inverse = None
        self.index_max_eigval = None
        self.golonom = None

    def safe_exp(self, x):
        return np.exp(x) if x < 700 else np.exp(700)

    def calc_gran(self):
        self.gran = []
        for i in range(self.size):
            log_term = np.log(self.err)
            gamma = abs((1 / abs(self.eigenvalues[i])) * log_term)
            self.gran.append(max(0, gamma))

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
        # Find the eigenvalue with the largest absolute value
            self.index_max_eigval = np.argmax(np.abs(self.eigenvalues))
            print(self.index_max_eigval)
            # Use the eigenvector corresponding to the largest eigenvalue
            print(self.inverse)
            eigenvector = self.inverse[self.index_max_eigval, :]
            print(eigenvector)
            # Form the holonomic constraint: sum(eigenvector[i] * x[i]) = 0
            self.golonom = sum(float(eigenvector[i]) * sp.symbols(f'x{i+1}') 
                            for i in range(self.size))
            print(self.golonom)
            # Express the user-selected variable (iskl_ind) from the holonomic constraint
            excluded_var = self.iskl_ind
            print(excluded_var)
            x_resh = sp.symbols(f'x{excluded_var + 1}')
            
            try:
                xi = sp.solve(self.golonom, x_resh)[0]
            except (sp.SympifyError, IndexError):
                raise ValueError(f"Не удалось выразить x{excluded_var + 1} через остальные переменные")
            
            # Form the system of equations
            self.matr_with_x = []
            for i in range(self.size):
                temp_expr = sum(self.values[i][j] * sp.symbols(f'x{j + 1}') 
                            for j in range(self.size))
                self.matr_with_x.append(temp_expr)
            
            # Substitute the excluded variable
            substituted_equations = [eq.subs(x_resh, xi) for eq in self.matr_with_x]
            simplified_equations = [sp.simplify(eq) for eq in substituted_equations]
            
            # Form the reduced matrix
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
            print(self.index_max_eigval)
            # Use the eigenvector corresponding to the largest eigenvalue
            print(self.inverse)
            eigenvector = self.inverse[self.index_max_eigval, :]
            print(eigenvector)
            # Form the holonomic constraint: sum(eigenvector[i] * x[i]) = 0
            self.golonom = sum(float(eigenvector[i]) * sp.symbols(f'x{i+1}') 
                            for i in range(self.size))
            print(self.golonom)

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

            # Ошибка типа I: СКО
            sum_diff_sq = np.sum(diff ** 2)
            sum_g_sq = np.sum(g ** 2)
            error1 = np.sqrt((1 / (n - 1)) * (sum_diff_sq / sum_g_sq)) if sum_g_sq != 0 else 0.0

            # Ошибка типа II: нормированная сумма отклонений
            sum_abs_diff = np.sum(np.abs(diff))
            max_abs_g = np.max(np.abs(g))
            error2 = (sum_abs_diff / n) / max_abs_g if max_abs_g != 0 else 0.0

            # Ошибка типа III: отн. среднее
            mean_abs_diff = sum_abs_diff / n
            mean_abs_g = np.mean(np.abs(g))
            error3 = mean_abs_diff / mean_abs_g if mean_abs_g != 0 else 0.0

            error1_list.append(error1)
            error2_list.append(error2)
            error3_list.append(error3)

        return error1_list, error2_list, error3_list

class MatrixApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализ матричных систем")
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
        self.reduced_matrix_grid = None
        self.reduced_entries = []
        self.holonomic_label = None
        self.interval_label = None
        
        self.init_ui()
        
    def init_ui(self):
        # Основной макет заменен на QSplitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Левая панель
        left_panel_container = QWidget()
        left_panel = QVBoxLayout()
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(left_panel_container)
        scroll_area.setStyleSheet("""
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
        
        matrix_group = QGroupBox("Задание исходной модели")
        matrix_group.setAlignment(Qt.AlignmentFlag.AlignCenter)
        matrix_layout = QVBoxLayout()
        
        self.input_file_radio = QRadioButton("Загрузить системную матрицу А из файла")
        self.input_manual_radio = QRadioButton("Ввести элементы системной матрицы А")
        self.input_manual_radio.setChecked(True)
        
        self.input_file_radio.toggled.connect(self.toggle_input_mode)
        
        self.size_label = QLabel("Размер матрицы n = Не определен")
        
        self.size_input = QSpinBox()
        self.size_input.setRange(2, 10)
        self.size_input.setValue(3)
        self.size_input.valueChanged.connect(self.update_matrix_inputs)
        
        self.matrix_input_layout = QHBoxLayout()
        matrix_label = QLabel("Матрица А:")
        matrix_label.setFixedWidth(80)
        self.matrix_grid = QGridLayout()
        self.matrix_input_layout.addWidget(matrix_label)
        self.matrix_input_layout.addLayout(self.matrix_grid)
        
        initial_conditions_layout = QVBoxLayout()
        initial_label = QLabel("Введите начальные условия")
        self.initial_layout = QGridLayout()
        initial_conditions_layout.addWidget(initial_label)
        initial_conditions_layout.addLayout(self.initial_layout)
        
        self.original_model_button = QPushButton("Вывести графики исходной модели")
        self.original_model_button.clicked.connect(self.show_original_model_graphs)
        
        matrix_layout.addWidget(self.input_file_radio)
        matrix_layout.addWidget(self.input_manual_radio)
        matrix_layout.addWidget(self.size_label)
        matrix_layout.addWidget(QLabel("Размер матрицы:"))
        matrix_layout.addWidget(self.size_input)
        matrix_layout.addLayout(self.matrix_input_layout)
        matrix_layout.addLayout(initial_conditions_layout)
        matrix_layout.addWidget(self.original_model_button)
        
        matrix_group.setLayout(matrix_layout)
        
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
        
        self.error_input = QLineEdit("0.1")
        validator = QDoubleValidator()
        validator.setLocale(QLocale("C"))
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.error_input.setValidator(validator)
        self.error_input.setFixedWidth(100)
        self.error_input.setStyleSheet("""
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
        
        self.delta_input = QLineEdit("0.1")
        self.delta_input.setValidator(validator)
        self.delta_input.setFixedWidth(100)
        self.delta_input.setStyleSheet("""
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
        
        error_input_layout.addWidget(delta_upper_label)
        error_input_layout.addWidget(self.delta_input)
        error_input_layout.addStretch()
        
        params_layout.addLayout(error_input_layout)
        params_group.setLayout(params_layout)
        
        criteria_group = QGroupBox("Выбор критерия оценивания декомпозиции модели")
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
        
        decomp_group = QGroupBox("Декомпозиция")
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
        
        left_panel.addWidget(matrix_group)
        left_panel.addWidget(params_group)
        left_panel.addWidget(criteria_group)
        left_panel.addWidget(decomp_group)
        left_panel.addStretch()
        left_panel_container.setLayout(left_panel)
        
        # Правая панель
        right_panel_widget = QWidget()
        self.right_panel = QVBoxLayout()
        
        # Бокс для результатов декомпозиции
        self.reduced_matrix_group = QGroupBox("Результаты декомпозиции")
        self.reduced_matrix_group.setAlignment(Qt.AlignmentFlag.AlignCenter)
        reduced_matrix_layout = QVBoxLayout()
        
        # Голономная связь
        self.holonomic_label = QLabel("Голономная связь: 0 = ")
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
        
        # Интервал справедливости
        self.interval_label = QLabel("Интервал справедливости: [0, 0] с")
        self.interval_label.setStyleSheet("""
            font-family: monospace;
            font-size: 12px;
            padding: 5px;
            background-color: #f6f6f6;
            border: 1px solid #cccccc;
            border-radius: 3px;
        """)
        self.interval_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        # Матрица Ā
        self.reduced_matrix_input_layout = QHBoxLayout()
        reduced_matrix_label = QLabel("Матрица Ā:")
        reduced_matrix_label.setFixedWidth(80)
        self.reduced_matrix_grid = QGridLayout()
        self.reduced_matrix_input_layout.addWidget(reduced_matrix_label)
        self.reduced_matrix_input_layout.addLayout(self.reduced_matrix_grid)
        
        # Кнопки управления
        self.solution_button = QPushButton("Графики решений")
        self.solution_button.clicked.connect(self.show_transient)
        self.solution_button.setMinimumWidth(250)
        self.solution_button.setStyleSheet("""
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
        """)
        
        self.deviation_button = QPushButton("Графики отклонений")
        self.deviation_button.clicked.connect(self.show_errors)
        self.deviation_button.setMinimumWidth(250)
        self.deviation_button.setStyleSheet("""
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
        """)
        
        self.proximity_button = QPushButton("Оценки близости процессов в исходной и упрощенной моделях")
        self.proximity_button.setMinimumWidth(250)
        self.proximity_button.setStyleSheet("""
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
        """)
        self.proximity_button.clicked.connect(self.show_quality)
        self.save_matrix_button = QPushButton("Сохранить матрицу в файл")
        self.save_matrix_button.clicked.connect(self.save_reduced_matrix)
        self.save_matrix_button.setMinimumWidth(250)
        self.save_matrix_button.setStyleSheet("""
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
        """)
        
        reduced_matrix_layout.addWidget(self.holonomic_label)
        reduced_matrix_layout.addWidget(self.interval_label)
        reduced_matrix_layout.addLayout(self.reduced_matrix_input_layout)
        reduced_matrix_layout.addWidget(self.solution_button)
        reduced_matrix_layout.addWidget(self.deviation_button)
        reduced_matrix_layout.addWidget(self.proximity_button)
        reduced_matrix_layout.addWidget(self.save_matrix_button)
        self.reduced_matrix_group.setLayout(reduced_matrix_layout)
        self.reduced_matrix_group.setVisible(False)
        
        self.right_panel.addWidget(self.reduced_matrix_group)
        self.right_panel.addStretch()
        right_panel_widget.setLayout(self.right_panel)
        
        # Установка минимальной ширины для панелей
        scroll_area.setMinimumWidth(300)
        right_panel_widget.setMinimumWidth(200)
        
        # Добавление панелей в QSplitter
        splitter.addWidget(scroll_area)
        splitter.addWidget(right_panel_widget)
        
        # Установка начальных пропорций (60:40)
        splitter.setSizes([600, 400])
        
        # Основной макет для центрального виджета
        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)
        self.central_widget.setLayout(main_layout)
        
        self.update_matrix_inputs()
        self.toggle_input_mode()
    
    def show_criteria_selection(self):
        dialog = CriteriaSelectionDialog(self)
        if dialog.exec():
            self.selected_criteria = dialog.get_selected_criteria()
            print(f"Выбраны критерии: {self.selected_criteria}")

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
                edit = QLineEdit("0")
                edit.setFixedWidth(60)
                edit.setValidator(validator)
                self.matrix_grid.addWidget(edit, i, j)
                row.append(edit)
            self.entries.append(row)
        
        for i in range(n):
            label = QLabel(f"x{i+1}(0):")
            edit = QLineEdit("0")
            edit.setValidator(validator)
            self.initial_layout.addWidget(label, i, 0)
            self.initial_layout.addWidget(edit, i, 1)
        
        self.size_label.setText(f"Размер матрицы n = {n}")

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
        # Очищаем предыдущую сетку
        for i in reversed(range(self.reduced_matrix_grid.count())):
            self.reduced_matrix_grid.itemAt(i).widget().deleteLater()
        self.reduced_entries.clear()
        
        # Устанавливаем голономную связь
        self.holonomic_label.setText(f"Голономная связь: 0 = {str(sp.simplify(golonom))}")
        
        # Устанавливаем интервал справедливости
        self.interval_label.setText(f"Интервал справедливости: [{min_gran:.2f}, {max_gran:.2f}] с")
        
        n = reduced_matrix.shape[0]
        validator = QDoubleValidator()
        validator.setLocale(QLocale("C"))
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        
        self.reduced_entries = []
        for i in range(n):
            row = []
            for j in range(n):
                value_str = f"{reduced_matrix[i][j]:.6f}".rstrip('0').rstrip('.')
                edit = QLineEdit(value_str)
                edit.setReadOnly(True)  # Запрет на редактирование
                edit.setStyleSheet("""
                    QLineEdit {
                        padding: 5px;
                        border: 1px solid #cccccc;
                        border-radius: 3px;
                        font-family: monospace;
                        font-size: 12px;
                    }
                """)
                # Масштабирование ширины поля под содержимое
                font_metrics = QFontMetrics(edit.font())
                text_width = font_metrics.boundingRect(value_str).width() + 10  # Дополнительный отступ
                edit.setFixedWidth(max(100, text_width))  # Минимальная ширина 100 пикселей
                self.reduced_matrix_grid.addWidget(edit, i, j)
                row.append(edit)
            self.reduced_entries.append(row)

    def save_reduced_matrix(self):
        if not self.reduced_entries:
            QMessageBox.warning(self, "Ошибка", "Матрица пониженного порядка не сформирована")
            return
        
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Сохранить матрицу", "", "Текстовые файлы (*.txt)")
        
        if not file_name:
            return
        
        try:
            n = len(self.reduced_entries)
            with open(file_name, 'w') as f:
                for i in range(n):
                    row = [self.reduced_entries[i][j].text() for j in range(n)]
                    f.write(" ".join(row) + "\n")
            QMessageBox.information(self, "Успех", "Матрица успешно сохранена")
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
                error = max(1e-6, float(self.error_input.text()))
            except ValueError:
                error = 0.1

            # Проверка матрицы на вырожденность
            if np.linalg.cond(matrix) > 1e12:
                raise ValueError("Матрица плохо обусловлена")

            # Вычисление исходной системы
            orig_system = Matrix(n, matrix, initial, error, -1)
            orig_system.calc()
            
            if not orig_system.gran:
                raise ValueError("Не удалось рассчитать границы справедливости")
                
            min_gran = min(orig_system.gran)
            max_gran = max(orig_system.gran)
            
            # Моделирование для поиска новых начальных условий
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
            
            # Находим новые начальные условия при t=min(gran)
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
            
            print(f"Новые начальные условия при t={min_gran:.2f}: {initial_conditions_new}")
               
            # Показываем голономную связь и выбор переменной
            dialog = HolonomicConstraintDialog(orig_system.golonom, n, orig_system.index_max_eigval, self)
            if not dialog.exec():
                QMessageBox.information(self, "Отмена", "Декомпозиция отменена")
                return
            
            excluded_indices = dialog.get_selected_index()
            self.excluded_index = excluded_indices
            # Update the excluded index for the reduced system
            orig_system_new = Matrix(n, matrix, initial_conditions_new, error, excluded_indices)
            orig_system_new.calc()
            # Формируем редуцированную модель
            initial_conditions_red = np.delete(initial_conditions_new, excluded_indices)
            reduced_system = Matrix(n-1, orig_system_new.matr_resh, initial_conditions_red, error, -1)
            reduced_system.calc()
            
            t_span_new = (min_gran, max_gran)
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
                raise ValueError("Моделирование редуцированной или исходной системы не удалось")
            
            self.errors = orig_system_new.calculate_errors(reduced_system, excluded_indices, self.selected_criteria)
            
            self.current_size = n
            self.current_bounds = (min_gran, max_gran)
            
            # Обновляем отображение матрицы пониженного порядка и показываем бокс
            self.update_reduced_matrix_display(reduced_system.values, orig_system.golonom, min_gran, max_gran)
            self.reduced_matrix_group.setVisible(True)
            
            print("Расчет выполнен успешно!")
            print(f"Размер системы: {n}x{n}")
            print(f"Границы справедливости: {min_gran:.2f} - {max_gran:.2f} с")
            print(f"Ошибки: Тип I: {np.mean(self.errors[0]):.4f}, Тип II: {np.mean(self.errors[1]):.4f}, Тип III: {np.mean(self.errors[2]):.4f}")

        except Exception as e:
            print(f"Критическая ошибка: {str(e)}")
            self.solution_original = None
            self.solution_reduced = None
            self.errors = None
            self.current_size = 0
            self.current_bounds = (0, 0)
            self.reduced_matrix_group.setVisible(False)  # Скрываем результаты
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
            print(f"Размер матрицы: {n}x{n}")
            
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
            print("Матрица A:")
            print(matrix)
            
            if np.all(matrix == 0):
                raise ValueError("Матрица не может быть полностью нулевой")
            
            try:
                cond_number = np.linalg.cond(matrix)
                print(f"Число обусловленности матрицы: {cond_number:.2e}")
                if cond_number > 1e12:
                    raise ValueError(f"Матрица плохо обусловлена (число обусловленности: {cond_number:.2e})")
            except np.linalg.LinAlgError as e:
                print(f"Ошибка при вычислении числа обусловленности: {str(e)}")
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
                    raise ValueError(f"Некорректное значение в поле начального условия x{i+1}(0): {text}")
            print("Начальные условия:")
            print(initial)
            
            
            time_dialog = TimeBoundaryDialog(self)
            if not time_dialog.exec():
                QMessageBox.information(self, "Отмена", "Ввод времени отменен")
                return
            t_end = time_dialog.get_time()
            print(f"Правая граница времени (t_end): {t_end}")
            if t_end <= 0:
                raise ValueError("Правая граница времени должна быть положительной")
            if t_end > 1e6:
                raise ValueError(f"Слишком большое время моделирования: {t_end}")
            
            t_span = (0, t_end)
            t_eval = np.linspace(0, t_end, 500)
            print("Интервал интегрирования (t_span):", t_span)
            print("Точки вывода (t_eval):", t_eval[:5], "...", t_eval[-5:])
            
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
                print("Результат solve_ivp:")
                print(f"solution: {solution}")
                print(f"solution.success: {solution.success}")
                print(f"solution.message: {solution.message}")
                print(f"solution.t.shape: {getattr(solution, 't', None).shape if solution is not None else 'None'}")
                print(f"solution.y.shape: {getattr(solution, 'y', None).shape if solution is not None else 'None'}")
            except Exception as e:
                print(f"Исключение в solve_ivp: {str(e)}")
                raise ValueError(f"Ошибка при выполнении solve_ivp: {str(e)}")
            
            if solution is None:
                print("Ошибка: solution is None")
                raise ValueError("Моделирование не выполнено: результат solve_ivp равен None")
            if not solution.success:
                print(f"Ошибка: solution.success is False, message: {solution.message}")
                raise ValueError(f"Моделирование не удалось: {solution.message}")
            if solution.y is None or solution.y.size == 0:
                raise ValueError("Моделирование не вернуло данных для графика")
            
            print("Запуск построения графика")
            window = TransientResponseWindow("Графики исходной модели", self)
            window.plot(solution, None, 0, n, 0, 0, t_end)
            window.exec()
        
        except ValueError as e:
            print(f"ValueError: {str(e)}")
            QMessageBox.critical(self, "Ошибка ввода", str(e))
        except Exception as e:
            print(f"Необработанное исключение: {str(e)}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось построить графики: {str(e)}")

if __name__ == "__main__":
    QLocale.setDefault(QLocale("C"))
    app = QApplication(sys.argv)
    window = MatrixApp()
    window.show()
    window.showMaximized()
    sys.exit(app.exec())
