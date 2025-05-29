# Импорт необходимых библиотек
import logging
import matplotlib
matplotlib.use('QtAgg')  # Использование бэкенда QtAgg для Matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QMainWindow, QSpinBox, QHBoxLayout, QDialog, QFileDialog,
    QGroupBox, QTextEdit, QTableWidgetItem, QTableWidget, QRadioButton, 
    QMessageBox, QComboBox, QScrollArea, QCheckBox, QSizePolicy, QSplitter,
    QMenu, QHeaderView
)
from PyQt6.QtGui import QPalette, QColor, QPen, QDoubleValidator, QFontMetrics, QPainter, QAction, QMouseEvent
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PyQt6.QtCore import Qt, QLocale, QPoint, QRectF, QPointF
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import sys
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

# Настройка поля ввода без запятых
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

# Настройка Matplotlib для текста
plt.rc('text', usetex=False)
plt.rc('font', family='Arial', size=12)

# Диалог для выбора временного интервала
class TimeBoundaryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выбор правой границы временного интервала")
        self.layout = QVBoxLayout()
        
        self.time_label = QLabel("Введите правую границу временного интервала T (с):")
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

# Окно для отображения графиков с масштабированием
logging.basicConfig(filename='chart.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


class ChartWindow(QDialog):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setGeometry(100, 100, 800, 600)
        
        # Инициализация QChartView
        self.chart_view = QChartView()
        self.chart_view.setRubberBand(QChartView.RubberBand.RectangleRubberBand)
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.chart_view.setInteractive(True)
        self.chart_view.setStyleSheet("""
            QChartView {
                background-color: white;
                border: 1px solid #cccccc;
            }
            QChartView:rubberBand {
                border: 2px dashed #0000FF;
                background-color: rgba(0, 0, 255, 50);
            }
        """)
        
        # Макет окна
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.chart_view)
        self.setLayout(self.layout)
        
        # Подсказка
        self.tooltip = QLabel(self.chart_view)
        self.tooltip.setStyleSheet("""
            background-color: rgba(255, 255, 255, 220);
            border: 1px solid black;
            padding: 4px;
            font-size: 12px;
        """)
        self.tooltip.setVisible(True)
        self.tooltip.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        
        # Переменные
        self.series_data = []
        self.initial_axis_ranges = None
        self.rubber_band_start = None
        self.pan_start = None
        self.pan_start_axes = None
        
        # Отслеживание мыши
        self.chart_view.setMouseTracking(True)
        self.setMouseTracking(True)
        
        # Установка фильтра событий на QChartView
        self.chart_view.viewport().installEventFilter(self)
    
    def eventFilter(self, obj, event):
        """Фильтрация событий для QChartView, чтобы обрабатывать панорамирование и координаты."""
        if obj == self.chart_view.viewport():
            if event.type() == event.Type.MouseMove:
                pos = self.chart_view.mapFromGlobal(event.globalPosition().toPoint())
                mouse_event = QMouseEvent(
                    event.Type.MouseMove,
                    QPointF(pos),
                    event.globalPosition(),
                    event.button(),
                    event.buttons(),
                    event.modifiers()
                )
                self.mouseMoveEvent(mouse_event)
                return False
            elif event.type() == event.Type.MouseButtonPress:
                pos = self.chart_view.mapFromGlobal(event.globalPosition().toPoint())
                mouse_event = QMouseEvent(
                    event.Type.MouseButtonPress,
                    QPointF(pos),
                    event.globalPosition(),
                    event.button(),
                    event.buttons(),
                    event.modifiers()
                )
                if event.button() == Qt.MouseButton.MiddleButton:
                    self.mousePressEvent(mouse_event)
                    return True
                return False
            elif event.type() == event.Type.MouseButtonRelease:
                pos = self.chart_view.mapFromGlobal(event.globalPosition().toPoint())
                mouse_event = QMouseEvent(
                    event.Type.MouseButtonRelease,
                    QPointF(pos),
                    event.globalPosition(),
                    event.button(),
                    event.buttons(),
                    event.modifiers()
                )
                if event.button() == Qt.MouseButton.MiddleButton:
                    self.mouseReleaseEvent(mouse_event)
                    return True
                return False
        return super().eventFilter(obj, event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Обработка движения мыши для подсказок и панорамирования."""
        logging.debug(f"Mouse move at {event.pos()}")
        pos = event.pos()
        chart = self.chart_view.chart()
        if not chart:
            logging.debug("No chart available")
            self.tooltip.setText("Нет графика")
            self.tooltip.adjustSize()
            self.position_tooltip(pos)
            return
        
        # Конвертация в координаты сцены и значения графика
        scene_pos = self.chart_view.mapToScene(pos)  # Убрано .toPoint()
        cursor_value = chart.mapToValue(QPointF(scene_pos.x(), scene_pos.y()))
        
        # Поиск ближайшей серии
        closest_series = None
        min_distance = float('inf')
        closest_y = None
        
        for series in chart.series():
            if isinstance(series, QLineSeries):
                points = [(series.at(i).x(), series.at(i).y()) for i in range(series.count())]
                if points:
                    x_values = np.array([p[0] for p in points])
                    y_values = np.array([p[1] for p in points])
                    scene_points = [chart.mapToPosition(QPointF(x, y)) for x, y in points]
                    distances = [((sp.x() - scene_pos.x())**2 + (sp.y() - scene_pos.y())**2)**0.5 
                                 for sp in scene_points]
                    distance = min(distances)
                    if distance < min_distance and distance < 50:
                        min_distance = distance
                        closest_series = series
                        if x_values[0] <= cursor_value.x() <= x_values[-1]:
                            closest_y = np.interp(cursor_value.x(), x_values, y_values)
                        elif cursor_value.x() < x_values[0]:
                            closest_y = y_values[0]
                        else:
                            closest_y = y_values[-1]
        
        # Обновление подсказки
        if closest_series:
            self.tooltip.setText(f"{closest_series.name() or 'График'}: X={cursor_value.x():.3f}, Y={closest_y:.3f}")
        else:
            self.tooltip.setText(f"X={cursor_value.x():.3f}, Y={cursor_value.y():.3f}")
        self.tooltip.adjustSize()
        self.position_tooltip(pos)
        
        # Панорамирование
        if self.pan_start is not None:
            self.handle_panning(event.pos())
    
    def position_tooltip(self, pos: QPoint):
        """Позиционирование подсказки."""
        tooltip_x = pos.x() + 10
        tooltip_y = pos.y() - self.tooltip.height() - 10
        tooltip_x = min(tooltip_x, self.width() - self.tooltip.width() - 5)
        tooltip_y = max(tooltip_y, 5)
        self.tooltip.move(tooltip_x, tooltip_y)
    
    def mousePressEvent(self, event: QMouseEvent):
        """Обработка нажатия кнопок мыши."""
        logging.debug(f"Mouse press at {event.pos()}, button: {event.button()}")
        if event.button() == Qt.MouseButton.LeftButton:
            self.rubber_band_start = event.pos()
            self.pan_start = None
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.pan_start = event.pos()
            self.rubber_band_start = None
            chart = self.chart_view.chart()
            x_axis = y_axis = None
            for axis in chart.axes():
                if axis.orientation() == Qt.Orientation.Horizontal:
                    x_axis = axis
                elif axis.orientation() == Qt.Orientation.Vertical:
                    y_axis = axis
            if x_axis and y_axis:
                self.pan_start_axes = {
                    'x': (x_axis.min(), x_axis.max()),
                    'y': (y_axis.min(), y_axis.max())
                }
                logging.debug(f"Panning started with axes: {self.pan_start_axes}")
        elif event.button() == Qt.MouseButton.RightButton:
            self.show_context_menu(event.pos())
            self.rubber_band_start = None
            self.pan_start = None
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Обработка отпускания кнопок мыши."""
        logging.debug(f"Mouse release at {event.pos()}, button: {event.button()}")
        if event.button() == Qt.MouseButton.LeftButton and self.rubber_band_start:
            chart = self.chart_view.chart()
            start_pos = self.rubber_band_start
            end_pos = event.pos()
            
            start_scene = self.chart_view.mapToScene(start_pos)  # Убрано .toPoint()
            end_scene = self.chart_view.mapToScene(end_pos)  # Убрано .toPoint()
            start_value = chart.mapToValue(QPointF(start_scene.x(), start_scene.y()))
            end_value = chart.mapToValue(QPointF(end_scene.x(), end_scene.y()))
            
            is_zoom_in = start_pos.x() < end_pos.x()
            logging.debug(f"Zoom direction: {'in' if is_zoom_in else 'out'}")
            
            rect = QRectF(
                min(start_value.x(), end_value.x()),
                min(start_value.y(), end_value.y()),
                abs(end_value.x() - start_value.x()),
                abs(end_value.y() - start_value.y())
            )
            logging.debug(f"Rectangle: {rect}")
            
            if rect.width() > 0 and rect.height() > 0:
                if is_zoom_in:
                    chart.zoomIn(rect)
                    logging.debug("Applied zoom in")
                else:
                    x_axis = y_axis = None
                    for axis in chart.axes():
                        if axis.orientation() == Qt.Orientation.Horizontal:
                            x_axis = axis
                        elif axis.orientation() == Qt.Orientation.Vertical:
                            y_axis = axis
                    if x_axis and y_axis:
                        x_range = x_axis.max() - x_axis.min()
                        y_range = y_axis.max() - y_axis.min()
                        zoom_factor = 2.0
                        new_x_range = x_range * zoom_factor
                        new_y_range = y_range * zoom_factor
                        x_center = (x_axis.max() + x_axis.min()) / 2
                        y_center = (y_axis.max() + y_axis.min()) / 2
                        x_axis.setRange(x_center - new_x_range / 2, x_center + new_x_range / 2)
                        y_axis.setRange(y_center - new_y_range / 2, y_center + new_y_range / 2)
                        logging.debug("Applied zoom out")
            
            self.rubber_band_start = None
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.pan_start = None
            self.pan_start_axes = None
            logging.debug("Panning stopped")
    
    def handle_panning(self, current_pos: QPoint):
        """Обработка панорамирования."""
        if self.pan_start is None or self.pan_start_axes is None:
            return
        
        chart = self.chart_view.chart()
        x_axis = y_axis = None
        for axis in chart.axes():
            if axis.orientation() == Qt.Orientation.Horizontal:
                x_axis = axis
            elif axis.orientation() == Qt.Orientation.Vertical:
                y_axis = axis
        
        if x_axis and y_axis:
            start_scene = self.chart_view.mapToScene(self.pan_start)  # Убрано .toPoint()
            current_scene = self.chart_view.mapToScene(current_pos)  # Убрано .toPoint()
            start_value = chart.mapToValue(QPointF(start_scene.x(), start_scene.y()))
            current_value = chart.mapToValue(QPointF(current_scene.x(), current_scene.y()))
            
            delta_x_value = start_value.x() - current_value.x()
            delta_y_value = start_value.y() - current_value.y()
            
            x_min = self.pan_start_axes['x'][0] + delta_x_value
            x_max = self.pan_start_axes['x'][1] + delta_x_value
            y_min = self.pan_start_axes['y'][0] + delta_y_value
            y_max = self.pan_start_axes['y'][1] + delta_y_value
            
            x_axis.setRange(x_min, x_max)
            y_axis.setRange(y_min, y_max)
            logging.debug(f"Panning to x: ({x_min}, {x_max}), y: ({y_min}, {y_max})")
    
    def show_context_menu(self, pos: QPoint):
        """Контекстное меню."""
        menu = QMenu(self)
        zoom_menu = menu.addMenu("Масштаб")
        
        percentages = [25, 50, 100, 200, 300, 500]
        for percent in percentages:
            action = QAction(f"{percent}%", self)
            action.triggered.connect(lambda checked, p=percent: self.apply_zoom(p / 100.0))
            zoom_menu.addAction(action)
        
        reset_action = QAction("Сбросить масштаб", self)
        reset_action.triggered.connect(self.reset_zoom)
        menu.addAction(reset_action)
        
        menu.exec(self.chart_view.mapToGlobal(pos))
    
    def apply_zoom(self, factor: float):
        """Масштабирование через меню."""
        chart = self.chart_view.chart()
        x_axis = y_axis = None
        for axis in chart.axes():
            if axis.orientation() == Qt.Orientation.Horizontal:
                x_axis = axis
            elif axis.orientation() == Qt.Orientation.Vertical:
                y_axis = axis
        
        if x_axis and y_axis:
            x_range = x_axis.max() - x_axis.min()
            y_range = y_axis.max() - y_axis.min()
            x_center = (x_axis.max() + x_axis.min()) / 2
            y_center = (y_axis.max() + y_axis.min()) / 2
            new_x_range = x_range / factor
            new_y_range = y_range / factor
            x_axis.setRange(x_center - new_x_range / 2, x_center + new_x_range / 2)
            y_axis.setRange(y_center - new_y_range / 2, y_center + new_y_range / 2)
            logging.debug(f"Applied menu zoom: factor={factor}")
    
    def reset_zoom(self):
        """Сброс масштаба."""
        if self.initial_axis_ranges:
            chart = self.chart_view.chart()
            chart.zoomReset()
            for axis in chart.axes():
                if axis.orientation() == Qt.Orientation.Horizontal:
                    axis.setRange(self.initial_axis_ranges['x'][0], self.initial_axis_ranges['x'][1])
                else:
                    axis.setRange(self.initial_axis_ranges['y'][0], self.initial_axis_ranges['y'][1])
            logging.debug("Reset zoom")
    
    def setup_axes(self, chart: QChart, t_start: float, t_end: float, y_values=None):
        """Настройка осей."""
        axis_x = QValueAxis()
        axis_x.setTitleText("t, с")
        axis_x.setRange(t_start, t_end)
        axis_x.setLabelFormat("%.2f")
        
        axis_y = QValueAxis()
        axis_y.setTitleText("xi(t)")
        
        if y_values:
            y_min = min(y_values)
            y_max = max(y_values)
            margin = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
            axis_y.setRange(y_min - margin, y_max + margin)
        
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        
        for series in chart.series():
            series.attachAxis(axis_x)
            series.attachAxis(axis_y)
        
        self.initial_axis_ranges = {
            'x': (t_start, t_end),
            'y': (axis_y.min(), axis_y.max())
        }
        logging.debug(f"Setup axes: x=({t_start}, {t_end}), y=({axis_y.min()}, {axis_y.max()})")
class TransientResponseWindow(ChartWindow):
    def plot(self, solution1, solution2, choice, size1, size2, t_start, t_end, excluded_index=None):
        chart = QChart()
        
        y_values = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Список индексов переменных, исключая исключённую
        original_indices = [i for i in range(size1) if i != excluded_index] if excluded_index is not None else list(range(size1))
        
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
            for reduced_idx, orig_idx in enumerate(original_indices):
                series = QLineSeries()
                # Нумерация переменных упрощённой модели соответствует исходным, исключая исключённую
                series.setName(f'x̄{orig_idx+1}')
                series.setColor(QColor(colors[orig_idx % len(colors)]))
                series.setPen(QPen(Qt.PenStyle.DashLine))
                y = solution2.y[reduced_idx][mask2]
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

# Окно для отображения ошибок
class ErrorResponseWindow(ChartWindow):
    def plot(self, solution1, solution2, full_size, excluded_index):
        from scipy.interpolate import interp1d
        chart = QChart()
        chart.setTitle("Графики отклонений решений x̄ᵢ от xᵢ")

        original_indices = [i for i in range(full_size) if i != excluded_index]
        t1 = solution1.t
        t2 = solution2.t

        errors_list = []  # Для хранения всех значений ошибок для настройки осей

        for reduced_idx, orig_idx in enumerate(original_indices):
            error_series = QLineSeries()
            error_series.setName(f'x{orig_idx + 1} - x̄{orig_idx + 1}')
            
            interp_func = interp1d(t2, solution2.y[reduced_idx], kind='linear', fill_value="extrapolate")
            interp_values = interp_func(t1)
            errors = solution1.y[orig_idx] - interp_values

            for j in range(len(t1)):
                error_series.append(t1[j], errors[j])
            
            chart.addSeries(error_series)
            self.series_data.append((error_series, t1, errors))
            errors_list.extend(errors)  # Собираем все ошибки для определения диапазона Y

        # Настройка осей с помощью метода setup_axes
        t_start = min(t1)
        t_end = max(t1)
        self.setup_axes(chart, t_start, t_end, errors_list)

        self.chart_view.setChart(chart)

# Диалог для отображения оценок качества
import sys
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit

class QualityMetricsDialog(QDialog):
    def __init__(self, errors, selected_criteria, full_size, excluded_index, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Оценки близости процессов xᵢ и x̄ᵢ")
        self.setGeometry(200, 200, 600, 450)
        
        layout = QVBoxLayout()
        
        # Заголовок с обновлённым текстом
        title_label = QLabel("Количественные оценки близости процессов xᵢ и x̄ᵢ — δᵢʲ, где i — номер переменной состояния, j — номер типа оценки, i = 1:n :")
        title_label.setStyleSheet("""
            font-family: Arial;
            font-size: 18px;
            font-weight: bold;
            padding: 10px;
            color: #333333;
        """)
        layout.addWidget(title_label)

        # Таблица
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)  # Запрет редактирования
        self.table.setAlternatingRowColors(True)  # Чередуемые цвета строк для читаемости
        self.table.setStyleSheet("""
            QTableWidget {
                font-family: Arial;
                font-size: 14px;
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 5px;
                background-color: #ffffff;
            }
            QTableWidget::item {
                padding: 5px;
                border: 1px solid #e0e0e0;
            }
            QTableWidget::item:selected {
                background-color: #1f77b4;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #f6f6f6;
                padding: 5px;
                border: 1px solid #cccccc;
                font-weight: bold;
            }
        """)

        # Подготовка данных для таблицы
        logging.debug(f"Received errors: {errors}, selected_criteria: {selected_criteria}, full_size: {full_size}, excluded_index: {excluded_index}")
        original_indices = [i for i in range(full_size) if i != excluded_index]
        criteria_names = ["I", "II", "III"]  # Римские цифры для критериев
        selected_errors = []
        for idx in selected_criteria:
            if idx < len(errors) and errors[idx]:
                selected_errors.append(errors[idx])
            else:
                selected_errors.append([None] * len(original_indices))  # Пустые данные для отсутствующих критериев

        # Если нет выбранных критериев, показываем пустую таблицу с прочерками
        if not selected_criteria:
            selected_criteria = [0]  # Добавляем фиктивный критерий для отображения
            selected_errors = [[None] * len(original_indices)]  # Пустые данные
            criteria_names = ["—"]

        # Настройка таблицы
        num_variables = full_size  # Количество переменных, включая исключённую
        self.table.setColumnCount(num_variables)  # Количество столбцов = количество переменных
        self.table.setRowCount(len(selected_criteria))  # Строки для каждого критерия

        # Заголовки столбцов (i = 1, 2, 3 и т.д.)
        headers = [f"i = {i + 1}" for i in range(num_variables)]
        self.table.setHorizontalHeaderLabels(headers)

        # Заполнение таблицы
        for row, criterion_idx in enumerate(selected_criteria):
            for col in range(num_variables):
                if col == excluded_index:
                    item = QTableWidgetItem("—")
                else:
                    # Находим индекс для ошибки, учитывая исключённую переменную
                    adjusted_col = original_indices.index(col) if col in original_indices else -1
                    if adjusted_col == -1 or selected_errors[row][0] is None:
                        item = QTableWidgetItem("—")
                    else:
                        error_value = selected_errors[row][adjusted_col]
                        item = QTableWidgetItem(f"{error_value:.6f}".rstrip('0').rstrip('.'))
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row, col, item)

        # Установка вертикальных заголовков (критерии)
        self.table.setVerticalHeaderLabels([criteria_names[idx] for idx in selected_criteria])

        # Установка одинаковой начальной высоты строк
        row_height = 40  # Фиксированная высота строки в пикселях
        for row in range(self.table.rowCount()):
            self.table.setRowHeight(row, row_height)

        # Настройка режима изменения размера строк на фиксированный
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        # Убрано setStretchLastSection, так как фиксированная высота не требует растяжения

        # Установка одинаковой ширины для всех столбцов
        for col in range(num_variables):
            self.table.setColumnWidth(col, 100)  # Фиксированная ширина 100 пикселей для каждого столбца

        layout.addWidget(self.table)
        self.setLayout(layout)

# Диалог для отображения матрицы пониженного порядка
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

# Диалог для выбора типа графика
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

# Диалог для выбора критериев оценки
class CriteriaSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выбор критерия оценивания близвости процессов xᵢ и x̄ᵢ")
        self.setGeometry(200, 300, 800, 500)
        
        layout = QVBoxLayout()
        title_label = QLabel("Выберите тип количественной(-ых) оценки(-ок) близости процессов xᵢ и x̄ᵢ, где i = 1:n :")
        title_label.setStyleSheet("""
            font-family: Arial;
            font-size: 18px;
            font-weight: bold;
            padding: 10px;
            color: #333333;
        """)
        layout.addWidget(title_label)
        error1_layout = QHBoxLayout()
        self.error1_cb = QCheckBox("Оценка типа I")
        self.error1_cb.setChecked(True)
        fig1 = Figure(figsize=(4.5, 1.2), tight_layout=True)
        canvas1 = FigureCanvasQTAgg(fig1)
        ax1 = fig1.add_subplot(111)
        ax1.text(0.1, 0.5, r'${\delta_i}^1 = \sqrt{\frac{1}{N-1} \frac{\sum_{j=1}^N (x_i(t_j) - \bar{x}_i(t_j))^2}{\sum_{j=1}^N x_i^2(t_j)}}$',
                 fontsize=18, verticalalignment='center')
        ax1.axis('off')
        error1_layout.addWidget(self.error1_cb)
        error1_layout.addWidget(canvas1)
        error1_layout.addStretch()
        
        error2_layout = QHBoxLayout()
        self.error2_cb = QCheckBox("Оценка типа II")
        self.error2_cb.setChecked(True)
        fig2 = Figure(figsize=(4.5, 1.2), tight_layout=True)
        canvas2 = FigureCanvasQTAgg(fig2)
        ax2 = fig2.add_subplot(111)
        ax2.text(0.1, 0.5, r'${\delta_i}^2= \frac{\frac{1}{N} \sum_{j=1}^N |x_i(t_j) - \bar{x}_i(t_j)|}{\max |x_i(t_j)|}$',
                 fontsize=18, verticalalignment='center')
        ax2.axis('off')
        error2_layout.addWidget(self.error2_cb)
        error2_layout.addWidget(canvas2)
        error2_layout.addStretch()
        
        error3_layout = QHBoxLayout()
        self.error3_cb = QCheckBox("Оценка типа III")
        self.error3_cb.setChecked(True)
        fig3 = Figure(figsize=(4.5, 1.2), tight_layout=True)
        canvas3 = FigureCanvasQTAgg(fig3)
        ax3 = fig3.add_subplot(111)
        ax3.text(0.1, 0.5, r'${\delta_i}^3 = \frac{\frac{1}{N} \sum_{j=1}^N |x_i(t_j) - \bar{x}_i(t_j)|}{\frac{1}{N} \sum_{j=1}^N |x_i(t_j)|}$',
                 fontsize=18, verticalalignment='center')
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

# Диалог для выбора голономной связи
class HolonomicConstraintDialog(QDialog):
    def __init__(self, golonom, matrix_size, max_eig_index, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Голономная связь и выбор исключаемой переменной")
        self.setGeometry(200, 200, 1200, 150)
        self.golonom = golonom
        self.matrix_size = matrix_size
        self.max_eig_index = max_eig_index
        self.selected_index = 0
        
        layout = QVBoxLayout()
        
        holonomic_layout = QHBoxLayout()
        holonomic_label = QLabel("Приближённая голономная связь между переменными исходной модели:")
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
        var_label = QLabel("Выберите переменную для исключения из исходной модели:")
        var_label.setStyleSheet("font-size: 14px; padding: 5px;")
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

# Класс для работы с матрицами
class Matrix:
    def __init__(self, n, values, initial_conditions, err, delta_large, iskl_ind=-1):
        self.iskl_ind = iskl_ind  # Индекс исключаемой переменной
        self.size = n  # Размер матрицы
        self.err = err  # δ⁰ (малая погрешность)
        self.delta_large = delta_large  # Δ⁰ (большая погрешность)
        self.values = values  # Значения матрицы
        self.initial_conditions = initial_conditions.reshape(1, -1) if initial_conditions.ndim == 1 else initial_conditions
        self.eigenvalues = np.zeros(n)  # Собственные значения
        self.eigenvectors = np.zeros((n, n))  # Собственные векторы
        self.coefficients = np.zeros(n)  # Коэффициенты
        self.gran = []  # Минимальные границы
        self.max_gran = []  # Максимальные границы
        self.matr_with_x = []  # Матрица с выражениями
        self.matr_resh = np.zeros((n-1, n-1))  # Пониженная матрица
        self.inverse = None  # Обратная матрица
        self.index_max_eigval = None  # Индекс максимального собственного значения
        self.golonom = None  # Голономная связь

    def safe_exp(self, x):
        # Безопасная экспонента для избежания переполнения
        return np.exp(x) if x < 700 else np.exp(700)

    def calc_gran(self):
        # Расчет границ справедливости
        self.gran = []  # Для минимальных границ (δ⁰)
        self.max_gran = []  # Для максимальных границ (Δ⁰)
        for i in range(self.size):
            log_term_min = np.log(self.err)
            log_term_max = np.log(self.delta_large)
            gamma_min = abs((1 / abs(self.eigenvalues[i])) * log_term_min)
            gamma_max = abs((1 / abs(self.eigenvalues[i])) * log_term_max)
            self.gran.append(max(0, gamma_min))
            self.max_gran.append(max(0, gamma_max))

    def calculate_eigenvalues_and_eigenvectors(self):
        # Вычисление собственных значений и векторов
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.values)

    def calculate_inverse(self):
        # Вычисление обратной матрицы собственных векторов
        self.inverse_eig = np.linalg.inv(self.eigenvectors)

    def calculate_coefficients(self):
        # Вычисление коэффициентов
        self.calculate_inverse()
        if self.inverse_eig is not None:
            self.coefficients = self.inverse_eig @ self.initial_conditions.T

    def calculate_inverse_from_eigenvectors(self):
        # Вычисление обратной матрицы из собственных векторов
        diagonal_matrix = np.diag(self.coefficients.flatten())
        try:
            self.inverse = np.linalg.inv(self.eigenvectors @ diagonal_matrix)
        except np.linalg.LinAlgError:
            self.inverse = None

    def expr_max_exp(self):
        # Выражение максимальной экспоненты и голономной связи
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
                    # Проверка на корректность коэффициента
                    try:
                        coeff_float = float(coeff)
                        if not np.isfinite(coeff_float):
                            coeff_float = 0.0
                    except (ValueError, TypeError):
                        coeff_float = 0.0
                    coefficients.append(coeff_float)
                if current_row < self.matr_resh.shape[0]:
                    self.matr_resh[current_row, :len(coefficients)] = coefficients
                    current_row += 1
        else:
            self.index_max_eigval = np.argmax(np.abs(self.eigenvalues))
            eigenvector = self.inverse[self.index_max_eigval, :]
            self.golonom = sum(float(eigenvector[i]) * sp.symbols(f'x{i+1}') 
                            for i in range(self.size))
    def calc(self):
        # Полный расчет матрицы
        self.calculate_eigenvalues_and_eigenvectors()
        self.calculate_coefficients()
        self.calculate_inverse_from_eigenvectors()
        self.expr_max_exp()
        self.calc_gran()

    def calculate_errors(self, reduced_model, excluded_index, selected_criteria):
        # Расчет ошибок моделирования
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
        
        # Инициализация переменных
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
        self.original_solution_data = None  # Данные симуляции исходной модели
        
        self.init_ui()
    def update_entry_width(self, edit):
        """Динамическое масштабирование ширины поля ввода на основе содержимого."""
        text = edit.text() or "0"  # Если поле пустое, используем "0" для расчета
        font_metrics = QFontMetrics(edit.font())
        text_width = font_metrics.boundingRect(text).width() + 12  # Дополнительный отступ
        min_width = 60  # Минимальная ширина поля
        max_width = 150  # Максимальная ширина поля
        edit.setFixedWidth(min(max_width, max(min_width, text_width)))
        

    def init_ui(self):
    # Установка глобального стиля для приложения
        app = QApplication.instance()
        app.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
                font-size: 14px;
                color: #2c3e50;
                background-color: #ffffff;
            }
            QMainWindow {
                background-color: #f8fafc;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 16px;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                margin-top: 12px;
                padding: 12px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 12px;
                color: #2563eb;
                background-color: transparent;
            }
            QLabel {
                color: #2c3e50;
                padding: 6px;
                background-color: transparent;
            }
            QLineEdit, QSpinBox, QComboBox {
                padding: 8px;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                background-color: #ffffff;
            }
            QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
                border: 1px solid #2563eb;
                background-color: #ffffff;
            }
            QPushButton {
                background-color: #2563eb;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
            QPushButton:pressed {
                background-color: #1e40af;
            }
            QRadioButton, QCheckBox {
                padding: 6px;
                color: #2c3e50;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: 1px solid #e2e8f0;
                background: #f1f5f9;
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #94a3b8;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QMenuBar {
                background-color: #ffffff;
                color: #2c3e50;
                font-size: 14px;
            }
            QMenu {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                padding: 6px;
                border-radius: 6px;
            }
            QMenu::item {
                padding: 6px 28px;
                background: transparent;
            }
            QMenu::item:selected {
                background-color: #2563eb;
                color: white;
                border-radius: 4px;
            }
            QTableWidget {
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                background-color: #ffffff;
                gridline-color: #e2e8f0;
            }
            QTableWidget::item {
                padding: 6px;
            }
            QHeaderView::section {
                background-color: #f1f5f9;
                padding: 6px;
                border: 1px solid #e2e8f0;
                font-weight: 500;
            }
            QDialog {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
            }
        """)

        # Создание меню
        menubar = self.menuBar()
        file_menu = menubar.addMenu("Файл")
        
        self.save_decomp_action = QAction("Сохранить результаты декомпозиции", self)
        self.save_decomp_action.triggered.connect(self.save_reduced_matrix)
        self.save_decomp_action.setEnabled(False)
        file_menu.addAction(self.save_decomp_action)
        
        self.save_error_params_action = QAction("Сохранить значения переменных исходной и упрощенной моделей", self)
        self.save_error_params_action.triggered.connect(self.save_error_calculation_params)
        self.save_error_params_action.setEnabled(False)
        file_menu.addAction(self.save_error_params_action)
        
        self.save_original_vars_action = QAction("Сохранить значения переменных исходной модели", self)
        self.save_original_vars_action.triggered.connect(self.save_original_model_variables)
        self.save_original_vars_action.setEnabled(False)
        file_menu.addAction(self.save_original_vars_action)
        
        about_menu = menubar.addMenu("О программе")
        about_action = QAction("О программе", self)
        about_action.triggered.connect(self.show_about)
        about_menu.addAction(about_action)
        
        # Создание основного разделителя
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Левая панель для ввода данных
        left_panel_container = QWidget()
        left_panel = QVBoxLayout()
        left_panel.setSpacing(12)
        left_panel.setContentsMargins(10, 10, 10, 10)
        
        # Группа для задания исходной модели
        matrix_group = QGroupBox("Задание исходной модели")
        matrix_group.setAlignment(Qt.AlignmentFlag.AlignCenter)
        matrix_layout = QVBoxLayout()
        matrix_layout.setSpacing(10)
        
        self.size_input = QSpinBox()
        self.size_input.setRange(2, 1000)
        self.size_input.setValue(3)
        self.size_input.valueChanged.connect(self.update_matrix_inputs)
        
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
        
        self.matrix_input_layout = QHBoxLayout()
        matrix_label = QLabel("Матрица А:")
        matrix_label.setFixedWidth(100)
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
        
        matrix_layout.addWidget(QLabel("Порядок модели (n):"))
        matrix_layout.addWidget(self.size_input)
        matrix_layout.addLayout(radio_layout)
        matrix_layout.addLayout(self.matrix_input_layout)
        matrix_layout.addLayout(initial_conditions_layout)
        matrix_layout.addWidget(self.original_model_button)
        
        matrix_group.setLayout(matrix_layout)
        
        # Группа для задания погрешностей
        params_group = QGroupBox("Задание допустимой погрешности упрощенной модели и точности установления процессов")
        params_group.setAlignment(Qt.AlignmentFlag.AlignCenter)
        params_layout = QVBoxLayout()
        params_layout.setSpacing(10)
        
        error_input_layout = QHBoxLayout()
        
        delta_lower_label = QLabel("Введите δ<sup>0</sup>:")
        delta_lower_label.setTextFormat(Qt.TextFormat.RichText)
        
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
        
        delta_upper_label = QLabel("Введите Δ<sup>0</sup>:")
        delta_upper_label.setTextFormat(Qt.TextFormat.RichText)
        
        self.delta_input = NoCommaLineEdit()
        self.delta_input.setValidator(validator)
        self.delta_input.setFixedWidth(100)
        self.delta_input.setText("0.1")
        
        error_input_layout.addWidget(delta_upper_label)
        error_input_layout.addWidget(self.delta_input)
        error_input_layout.addStretch()
        
        params_layout.addLayout(error_input_layout)
        params_group.setLayout(params_layout)
        
        # Группа для выбора критериев
        criteria_group = QGroupBox("Выбор критерия оценивания близости процессов исходной и упрощенной моделей")
        criteria_group.setAlignment(Qt.AlignmentFlag.AlignCenter)
        criteria_layout = QVBoxLayout()
        criteria_layout.setSpacing(10)
        
        self.criteria_button = QPushButton("Выбрать критерий")
        self.criteria_button.clicked.connect(self.show_criteria_selection)
        
        criteria_layout.addWidget(self.criteria_button)
        criteria_layout.addStretch()
        criteria_group.setLayout(criteria_layout)
        
        # Группа для декомпозиции
        decomp_group = QGroupBox("Декомпозиция исходной модели")
        decomp_group.setAlignment(Qt.AlignmentFlag.AlignCenter)
        decomp_layout = QVBoxLayout()
        decomp_layout.setSpacing(10)
        
        self.calc_button = QPushButton("Осуществить декомпозицию модели")
        self.calc_button.clicked.connect(self.perform_calculations)
        decomp_layout.addWidget(self.calc_button)
        decomp_layout.addStretch()
        decomp_group.setLayout(decomp_layout)
        
        left_panel.addWidget(matrix_group)
        left_panel.addWidget(params_group)
        left_panel.addWidget(criteria_group)
        left_panel.addWidget(decomp_group)
        left_panel.addStretch()
        left_panel_container.setLayout(left_panel)
        
        left_scroll_area = QScrollArea()
        left_scroll_area.setWidgetResizable(True)
        left_scroll_area.setWidget(left_panel_container)
        
        # Правая панель для результатов декомпозиции
        right_panel_container = QWidget()
        right_panel = QVBoxLayout()
        right_panel.setSpacing(12)
        right_panel.setContentsMargins(10, 10, 10, 10)
        
        self.reduced_matrix_group = QGroupBox("Результаты декомпозиции")
        self.reduced_matrix_group.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.reduced_matrix_layout = QVBoxLayout()
        self.reduced_matrix_layout.setSpacing(10)
        
        self.placeholder_label = QLabel()
        self.reduced_matrix_layout.addWidget(self.placeholder_label)
        
        self.reduced_matrix_group.setLayout(self.reduced_matrix_layout)
        right_panel.addWidget(self.reduced_matrix_group)
        right_panel.addStretch()
        right_panel_container.setLayout(right_panel)
        
        right_scroll_area = QScrollArea()
        right_scroll_area.setWidgetResizable(True)
        right_scroll_area.setWidget(right_panel_container)
        
        left_scroll_area.setMinimumWidth(400)
        right_scroll_area.setMinimumWidth(300)
        
        splitter.addWidget(left_scroll_area)
        splitter.addWidget(right_scroll_area)
        splitter.setSizes([600, 400])
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.addWidget(splitter)
        self.central_widget.setLayout(main_layout)
        
        self.right_panel = right_panel
        
        self.update_matrix_inputs()
        self.toggle_input_mode()
    def show_about(self):
        # Отображение информации о программе
        about_dialog = QMessageBox(self)
        about_dialog.setWindowTitle("О программе")
        about_dialog.setStyleSheet("""
            QMessageBox {
                background-color: #f6f6f6;
                font-family: Arial;
                font-size: 14px;
            }
            QMessageBox QLabel {
                color: #333333;
                padding: 10px;
            }
            QMessageBox QPushButton {
                background-color: #1f77b4;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                font-size: 13px;
            }
            QMessageBox QPushButton:hover {
                background-color: #1565c0;
            }
            QMessageBox QPushButton:pressed {
                background-color: #0d47a1;
            }
        """)
        
        about_text = """
            <div style='text-align: center;'>
                <h2 style='color: #1f77b4; margin-bottom: 10px;'>Декомпозиция моделей линейных САУ</h2>
                <p style='margin: 5px 0;'><b>Версия:</b> 1.0</p>
                <p style='margin: 5px 0;'><b>Разработчик:</b> Ермакова П.А., А-02-21</p>
                <p style='margin: 5px 0;'><b>Руководитель:</b> Сидорова Е.Ю.</p>
                <p style='margin: 5px 0;'><b>Организация:</b> Москва, НИУ "МЭИ", кафедра УИТ</p>
                <p style='margin: 5px 0;'><b>Год:</b> 2025</p>
            </div>
        """
        about_dialog.setText(about_text)
        about_dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        about_dialog.setIcon(QMessageBox.Icon.Information)
        about_dialog.exec()
    
    def show_criteria_selection(self):
        # Отображение диалога выбора критериев
        dialog = CriteriaSelectionDialog(self)
        if dialog.exec():
            self.selected_criteria = dialog.get_selected_criteria()

    def update_entry_width(self, edit):
        """Динамическое масштабирование ширины поля ввода на основе содержимого."""
        text = edit.text() or "0"  # Если поле пустое, используем "0" для расчета
        font_metrics = QFontMetrics(edit.font())
        text_width = font_metrics.boundingRect(text).width() + 12  # Дополнительный отступ
        min_width = 60  # Минимальная ширина поля
        max_width = 150  # Максимальная ширина поля
        edit.setFixedWidth(min(max_width, max(min_width, text_width)))

    def update_matrix_inputs(self):
        # Обновление полей ввода матрицы
        n = self.size_input.value()
        
        for i in reversed(range(self.matrix_grid.count())):
            self.matrix_grid.itemAt(i).widget().deleteLater()
        self.entries.clear()
        
        for i in reversed(range(self.initial_layout.count())):
            self.initial_layout.itemAt(i).widget().deleteLater()
        
        validator = QDoubleValidator()
        validator.setLocale(QLocale("C"))
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        
        # Сбор всех значений для определения максимальной ширины только для матрицы
        max_text_width = 0
        font_metrics = QFontMetrics(self.font())
        for i in range(n):
            for j in range(n):
                text = self.entries[i][j].text().strip() if self.entries and i < len(self.entries) and j < len(self.entries[i]) else "0"
                text_width = font_metrics.boundingRect(text).width() + 12
                max_text_width = max(max_text_width, text_width)
        
        max_text_width = min(150, max(60, max_text_width))  # Ограничение ширины
        
        self.entries = []
        for i in range(n):
            row = []
            for j in range(n):
                edit = NoCommaLineEdit()
                edit.setValidator(validator)
                edit.setText("0")
                edit.setFixedWidth(max_text_width)
                edit.textChanged.connect(lambda: self.adjust_all_entries_width())
                self.matrix_grid.addWidget(edit, i, j)
                row.append(edit)
            self.entries.append(row)
        
        for i in range(n):
            label = QLabel(f"x{i+1}(0):")
            edit = NoCommaLineEdit()
            edit.setValidator(validator)
            edit.setText("0")
            edit.setMinimumWidth(300)  # Большая фиксированная ширина для начальных условий
            self.initial_layout.addWidget(label, i, 0)
            self.initial_layout.addWidget(edit, i, 1)

    def adjust_all_entries_width(self):
        """Масштабирование всех полей ввода матрицы под самое длинное число."""
        max_text_width = 0
        font_metrics = QFontMetrics(self.font())
        
        # Проверяем только поля матрицы
        for i in range(len(self.entries)):
            for j in range(len(self.entries[i])):
                text = self.entries[i][j].text().strip() or "0"
                text_width = font_metrics.boundingRect(text).width() + 12
                max_text_width = max(max_text_width, text_width)
        
        max_text_width = min(150, max(60, max_text_width))  # Ограничение ширины
        
        # Устанавливаем ширину только для полей матрицы
        for i in range(len(self.entries)):
            for j in range(len(self.entries[i])):
                self.entries[i][j].setFixedWidth(max_text_width)

    def toggle_input_mode(self):
        # Переключение режима ввода (вручную или из файла)
        is_file_input = self.input_file_radio.isChecked()
        self.size_input.setEnabled(not is_file_input)
        if is_file_input:
            self.load_from_file()
        else:
            self.update_matrix_inputs()

    def load_from_file(self):
        # Загрузка матрицы из файла
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

        clear_layout(self.reduced_matrix_layout)
        self.reduced_entries.clear()

        self.holonomic_label = QLabel(f"Приближённая голономная связь: 0 = {str(sp.simplify(golonom))}")
        self.holonomic_label.setWordWrap(True)
        self.holonomic_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.excluded_var_label = QLabel(f"Исключаемая переменная: x{self.excluded_index + 1}")

        reduced_model_title = QLabel("Упрощённая модель и её параметры: ")

        n = reduced_matrix.shape[0]
        order_label = QLabel(f"Порядок модели (n): {n}")

        self.reduced_matrix_input_layout = QHBoxLayout()
        reduced_matrix_label = QLabel("Матрица Ā:")
        reduced_matrix_label.setFixedWidth(100)
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
                value = reduced_matrix[i][j]
                # Проверка на корректность значения
                if not np.isfinite(value):
                    value_str = "0.0"
                else:
                    formatted_value = f"{value:.6f}".rstrip('0').rstrip('.')
                    value_str = formatted_value if formatted_value else "0.0"
                edit = QLineEdit(value_str)
                edit.setReadOnly(True)
                font_metrics = QFontMetrics(edit.font())
                text_width = font_metrics.boundingRect(value_str).width() + 10
                edit.setFixedWidth(max(100, text_width))
                self.reduced_matrix_grid.addWidget(edit, i, j)
                row.append(edit)
            self.reduced_entries.append(row)

        self.interval_label = QLabel(f"Интервал справедливости упрощённой модели [Г1, Гn]: [{min_gran:.2f}, {max_gran:.2f}] (c)")
        self.interval_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        if max_gran != 0:
            gamma = (max_gran - min_gran) / max_gran
        else:
            gamma = 0.0

        efficiency_label = QLabel(f"Критерий эффективности упрощённой модели(γ): {gamma:.6f}")
        efficiency_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.solution_button = QPushButton("Графики процессов xᵢ и x̄ᵢ")
        self.solution_button.clicked.connect(self.show_transient)
        self.solution_button.setMinimumWidth(250)

        self.deviation_button = QPushButton("Графики отклонений x̄ᵢ от xᵢ")
        self.deviation_button.clicked.connect(self.show_errors)
        self.deviation_button.setMinimumWidth(250)

        self.proximity_button = QPushButton("Оценки близости процессов xᵢ и x̄ᵢ")
        self.proximity_button.setMinimumWidth(250)
        self.proximity_button.clicked.connect(self.show_quality)

        self.reduced_matrix_layout.addWidget(self.holonomic_label)
        self.reduced_matrix_layout.addWidget(self.excluded_var_label)
        self.reduced_matrix_layout.addWidget(reduced_model_title)
        self.reduced_matrix_layout.addWidget(order_label)
        self.reduced_matrix_layout.addLayout(self.reduced_matrix_input_layout)
        self.reduced_matrix_layout.addWidget(self.interval_label)
        self.reduced_matrix_layout.addWidget(efficiency_label)
        self.reduced_matrix_layout.addWidget(self.solution_button)
        self.reduced_matrix_layout.addWidget(self.deviation_button)
        self.reduced_matrix_layout.addWidget(self.proximity_button)

    def save_original_model_variables(self):
        # Сохранение значений переменных исходной модели
        if not self.original_solution_data:
            QMessageBox.warning(self, "Ошибка", "Данные симуляции исходной модели не доступны. Сначала постройте графики исходной модели.")
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Сохранить значения переменных исходной модели", "", "Текстовые файлы (*.txt)"
        )

        if not file_name:
            return

        try:
            t = self.original_solution_data['t']
            y = self.original_solution_data['y']
            n = self.original_solution_data['n']

            headers = ["t"] + [f"x{i+1}" for i in range(n)]
            data = []
            for i in range(len(t)):
                row = [t[i]] + [y[j][i] for j in range(n)]
                data.append(row)

            col_width = 12
            formatted_data = [[f"{val:.6f}".rstrip('0').rstrip('.') for val in row] for row in data]

            header_str = ""
            for i, header in enumerate(headers):
                header_str += f" {header:<{col_width-2}} "
                if i < len(headers) - 1:
                    header_str += "|"

            separator = "-" * (col_width * len(headers) + len(headers) - 1)

            rows = []
            for i in range(len(data)):
                row_str = ""
                for j in range(len(headers)):
                    row_str += f" {formatted_data[i][j]:<{col_width-2}} "
                    if j < len(headers) - 1:
                        row_str += "|"
                rows.append(row_str)

            table = [separator, header_str, separator]
            table.extend(rows)
            table.append(separator)

            with open(file_name, 'w', encoding='utf-8') as f:
                f.write("Значения переменных исходной модели\n")
                f.write("=" * 50 + "\n\n")
                f.write("\n".join(table) + "\n")

            QMessageBox.information(self, "Успех", "Значения переменных исходной модели успешно сохранены")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл: {str(e)}")

    def save_error_calculation_params(self):
        # Сохранение параметров для расчета ошибки
        if not self.solution_original or not self.solution_reduced:
            QMessageBox.warning(self, "Ошибка", "Решения для исходной или упрощенной модели не рассчитаны")
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Сохранить значения переменных исходной и упрощенной моделей", "", "Текстовые файлы (*.txt)"
        )

        if not file_name:
            return

        try:
            t = self.solution_original.t
            original_vars = self.solution_original.y
            reduced_vars = self.solution_reduced.y
            n = self.current_size
            excluded_index = self.excluded_index

            headers = ["t"]
            for i in range(n):
                headers.append(f"x{i+1}")
            original_indices = [i for i in range(n) if i != excluded_index]
            for i in original_indices:
                headers.append(f"x̄{i+1}")

            data = []
            for i in range(len(t)):
                row = [t[i]]
                for j in range(n):
                    row.append(original_vars[j][i])
                for j in range(len(original_indices)):
                    row.append(reduced_vars[j][i])
                data.append(row)

            col_width = 12
            formatted_data = [[f"{val:.6f}".rstrip('0').rstrip('.') for val in row] for row in data]

            header_str = ""
            for i, header in enumerate(headers):
                header_str += f" {header:<{col_width-2}} "
                if i < len(headers) - 1:
                    header_str += "|"

            separator = "-" * (col_width * len(headers) + len(headers) - 1)

            rows = []
            for i in range(len(data)):
                row_str = ""
                for j in range(len(headers)):
                    row_str += f" {formatted_data[i][j]:<{col_width-2}} "
                    if j < len(headers) - 1:
                        row_str += "|"
                rows.append(row_str)

            table = [separator, header_str, separator]
            table.extend(rows)
            table.append(separator)

            with open(file_name, 'w', encoding='utf-8') as f:
                f.write("Значения исходной и упрощенной моделей\n")
                f.write("=" * 50 + "\n\n")
                f.write("\n".join(table) + "\n")

            QMessageBox.information(self, "Успех", "Значения исходной и упрощенной моделей успешно сохранены")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл: {str(e)}")

    def format_table(self, headers, data, row_labels=None, col_width=10):
        # Форматирование данных в таблицу
        n_cols = len(headers)
        n_rows = len(data)
        
        # Не пытаемся преобразовывать данные в float, оставляем их как есть
        formatted_data = []
        for row in data:
            formatted_row = []
            for val in row:
                if isinstance(val, (int, float)) and np.isfinite(val):
                    # Если значение числовое и конечное, форматируем его
                    formatted_val = f"{val:.6f}".rstrip('0').rstrip('.')
                else:
                    # Если значение не числовое (например, "—"), оставляем как есть
                    formatted_val = str(val)
                formatted_row.append(formatted_val)
            formatted_data.append(formatted_row)
        
        label_width = max([len(str(label)) for label in row_labels]) + 2 if row_labels else 6
        col_widths = [label_width] + [col_width] * n_cols if row_labels else [col_width] * n_cols
        
        header_str = " " * col_widths[0] + "|"
        for i, header in enumerate(headers):
            header_str += f" {header:^{col_widths[i+1]-2}} |"
        
        separator = "-" * col_widths[0] + "+"
        for width in col_widths[1:]:
            separator += "-" * (width - 1) + "+"
        
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
        
        table = [separator, header_str, separator]
        table.extend(rows)
        table.append(separator)
    
        return "\n".join(table)
    def save_reduced_matrix(self):
        # Сохранение результатов декомпозиции
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
            reduced_matrix = []
            
            # Проверка и преобразование значений reduced_entries
            for i in range(n_red):
                row = []
                for j in range(n_red):
                    value_str = self.reduced_entries[i][j].text().strip()
                    if not value_str:
                        value_str = "0.0"  # Заменяем пустую строку на 0.0
                    try:
                        value = float(value_str)
                        if not np.isfinite(value):
                            value = 0.0  # Заменяем NaN или inf на 0.0
                        row.append(value)
                    except ValueError as ve:
                        raise ValueError(f"Некорректное значение в поле матрицы Ā[{i+1},{j+1}]: '{value_str}' ({str(ve)})")
                reduced_matrix.append(row)

            reduced_eigenvalues = np.linalg.eigvals(np.array(reduced_matrix))
            with open(file_name, 'w', encoding='utf-8') as f:
                
                
                # Исходная модель
                f.write("=" * 50 + "\n\n")
                f.write("1. Исходная модель\n\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Системная матрица исходной модели А (порядок n = {n}):\n")
                headers = [f"Столбец {j+1}" for j in range(n)]
                row_labels = [f"Строка {i+1}" for i in range(n)]
                matrix_table = self.format_table(headers, self.original_matrix, row_labels, col_width=12)
                f.write(matrix_table + "\n\n")
                
                f.write("Начальные условия:\n")
                initial_data = [[self.initial_conditions[i]] for i in range(n)]
                headers = ["Значение"]
                row_labels = [f"x{i+1}(0)" for i in range(n)]
                initial_table = self.format_table(headers, initial_data, row_labels, col_width=12)
                f.write(initial_table + "\n\n")
                
                f.write("Собственные числа матрицы А:\n")
                eig_data = [[self.eigenvalues[i]] for i in range(n)]
                headers = ["Значение"]
                row_labels = [f"λ{i+1}" for i in range(n)]
                eig_table = self.format_table(headers, eig_data, row_labels, col_width=12)
                f.write(eig_table + "\n\n")
                
                # Результаты декомпозиции
                f.write("=" * 50 + "\n\n")
                f.write("2. Результаты декомпозиции\n")
                f.write("=" * 50 + "\n\n")
                
                
                f.write(f"Приближённая голономная связь между переменными исходной модели:\n")
                f.write(f"0 = {str(sp.simplify(self.golonom))}\n\n")
                f.write(f"Исключаемая переменная: x{self.excluded_index + 1}\n\n")
                f.write("Упрощённая модель и её параметры:\n\n")
                
                f.write(f"Системная матрица Ā упрощённой модели (порядок n = {n_red}):\n")
                headers = [f"Столбец {j+1}" for j in range(n_red)]
                row_labels = [f"Строка {i+1}" for i in range(n_red)]
                reduced_table = self.format_table(headers, reduced_matrix, row_labels, col_width=12)
                f.write(reduced_table + "\n\n")

                f.write("Собственные числа матрицы Ā:\n")
                reduced_eig_data = [[reduced_eigenvalues[i]] for i in range(n_red)]
                headers = ["Значение"]
                row_labels = [f"λ{i+1}" for i in range(n_red)]
                reduced_eig_table = self.format_table(headers, reduced_eig_data, row_labels, col_width=12)
                f.write(reduced_eig_table + "\n")
                
                f.write("Интервал справедливости упрощённой модели [Г1, Гn]:\n")
                interval_data = [[self.current_bounds[0], self.current_bounds[1]]]
                headers = ["Г1", "Гn"]
                row_labels = ["t, сек"]
                interval_table = self.format_table(headers, interval_data, row_labels, col_width=12)
                f.write(interval_table + "\n\n")
                
                f.write("Количественные оценки близости процессов исходной и упрощённой моделей (δᵢʲ, где i — номер переменной состояния, j — номер типа оценки):\n")
                original_indices = [i for i in range(n) if i != self.excluded_index]
                criteria_names = ["I", "II", "III"]
                selected_errors = []
                error_labels = []
                for idx in self.selected_criteria:
                    selected_errors.append([self.errors[idx][i] for i in range(len(original_indices))])
                    error_labels.append(criteria_names[idx])
                
                if selected_errors:
                    headers = [f"i = {i + 1}" for i in range(n)]
                    row_labels = [criteria_names[idx] for idx in self.selected_criteria]
                    error_data = []
                    for row in range(len(self.selected_criteria)):
                        row_data = []
                        for col in range(n):
                            if col == self.excluded_index:
                                row_data.append("—")
                            else:
                                adjusted_col = original_indices.index(col) if col in original_indices else -1
                                if adjusted_col == -1 or not selected_errors[row]:  # Проверка на пустой список ошибок
                                    row_data.append("—")
                                else:
                                    error_value = selected_errors[row][adjusted_col]
                                    row_data.append(error_value if error_value is not None else "—")
                        error_data.append(row_data)
                    error_table = self.format_table(headers, error_data, row_labels, col_width=12)
                    f.write(error_table + "\n")
            
            QMessageBox.information(self, "Успех", "Результаты декомпозиции успешно сохранены")
        except ValueError as ve:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл: {str(ve)}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл: {str(e)}")
    def perform_calculations(self):
        # Выполнение декомпозиции
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
                delta_large = max(1e-6, float(self.delta_input.text()))
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
                raise ValueError("Моделирование упрощенной или исходной системы не удалось")
            
            self.errors = orig_system_new.calculate_errors(reduced_system, excluded_indices, self.selected_criteria)
            
            self.current_size = n
            self.current_bounds = (min_gran, max_gran)
            self.original_matrix = matrix
            self.initial_conditions = initial
            self.eigenvalues = orig_system.eigenvalues
            self.golonom = orig_system.golonom
            
            # Активация пунктов меню после успешной декомпозиции
            self.save_decomp_action.setEnabled(True)
            self.save_error_params_action.setEnabled(True)
            self.save_original_vars_action.setEnabled(True)
            
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
            self.save_decomp_action.setEnabled(False)
            self.save_error_params_action.setEnabled(False)
            self.save_original_vars_action.setEnabled(False)
            QMessageBox.critical(self, "Ошибка", f"Не удалось выполнить расчет: {str(e)}")

    def show_transient(self):
        # Отображение графиков решений
        if self.solution_original and self.solution_reduced and hasattr(self, 'current_bounds'):
            t_start, t_end = self.current_bounds
            choice_dialog = GraphChoiceDialog(self)
            if choice_dialog.exec():
                choice = choice_dialog.get_choice()
                window = TransientResponseWindow("Графики процессов", self)
                window.plot(
                    self.solution_original,
                    self.solution_reduced,
                    choice,
                    self.current_size,
                    self.current_size-1,
                    t_start,
                    t_end,
                    self.excluded_index  # Передаём индекс исключённой переменной
                )
                window.exec()

    def show_errors(self):
        # Отображение графиков отклонений
        if self.solution_original and self.solution_reduced:
            window = ErrorResponseWindow("Графики отклонений процессов x̄ᵢ от xᵢ", self)
            window.plot(self.solution_original, self.solution_reduced, self.current_size, self.excluded_index)
            window.exec()

    def show_quality(self):
        # Отображение оценок качества
        if self.errors and self.selected_criteria:
            dialog = QualityMetricsDialog(self.errors, self.selected_criteria, self.current_size, self.excluded_index, self)
            dialog.exec()
        else:
            QMessageBox.warning(self, "Ошибка", "Оценки близости не рассчитаны или не выбраны критерии.")

    def show_original_model_graphs(self):
        # Отображение графиков процессов исходной модели
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
            
            initial_conditions = np.zeros(n)
            for i in range(n):
                text = self.initial_layout.itemAt(i*2+1).widget().text().strip()
                if not text:
                    raise ValueError(f"Поле начального условия x{i+1}(0) пустое")
                try:
                    value = float(text.replace(",", "."))
                    if abs(value) > 1e6:
                        raise ValueError(f"Слишком большое значение в поле начального условия x{i+1}(0): {text}")
                    initial_conditions[i] = value
                except ValueError:
                    raise ValueError(f"Некорректное значение в поле начального условия x{i+1}(0): {text}")

            # Запрос правой границы временного интервала
            time_dialog = TimeBoundaryDialog(self)
            if not time_dialog.exec():
                QMessageBox.information(self, "Отмена", "Построение графиков отменено")
                return
            t_end = time_dialog.get_time()
            if t_end <= 0:
                raise ValueError("Правая граница временного интервала должна быть положительной")

            t_span = (0, t_end)
            t_eval = np.linspace(0, t_end, 500)
            
            # Решение дифференциального уравнения
            solution = solve_ivp(
                lambda t, y, matrix: matrix @ y,
                t_span,
                initial_conditions,
                args=(matrix,),
                t_eval=t_eval,
                method='LSODA'
            )
            
            if not solution.success:
                raise ValueError(f"Моделирование не удалось: {solution.message}")
            
            # Сохранение данных симуляции для последующего сохранения в файл
            self.original_solution_data = {
                't': solution.t,
                'y': solution.y,
                'n': n
            }
            
            # Отображение графиков
            window = TransientResponseWindow("Графики процессов исходной модели", self)
            window.plot(solution, None, 0, n, 0, 0, t_end)
            window.exec()  # Убрано оповещение о закрытии
            
        except ValueError as e:
            QMessageBox.critical(self, "Ошибка ввода", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось построить графики: {str(e)}")
# Точка входа в приложение
if __name__ == "__main__":
    QLocale.setDefault(QLocale("C"))
    app = QApplication(sys.argv)
    window = MatrixApp()
    window.show()
    window.showMaximized()
    sys.exit(app.exec())