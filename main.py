import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton, QMainWindow
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCharts import QChart, QChartView, QLineSeries

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
        
        self.entries = []
        for i in range(3):
            row_entries = []
            for j in range(3):
                entry = QLineEdit()
                entry.setPlaceholderText(f"Element ({i+1},{j+1})")
                row_entries.append(entry)
                self.grid_layout.addWidget(entry, i, j)
            self.entries.append(row_entries)
        
        self.plot_button = QPushButton("Plot Exponents")
        self.plot_button.clicked.connect(self.plot_exponents)
        
        self.layout.addLayout(self.grid_layout)
        self.layout.addWidget(self.plot_button)
        
        self.central_widget.setLayout(self.layout)
        
        self.chart_view = QChartView()
        self.layout.addWidget(self.chart_view)

    def set_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
        #palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        self.setPalette(palette)

    def plot_exponents(self):
        chart = QChart()
        for i in range(3):
            for j in range(3):
                try:
                    value = float(self.entries[i][j].text())
                    series = QLineSeries()
                    if (i == j):
                        for x in range(50):
                            series.append(x, np.exp(-value * x))
                        chart.addSeries(series)
                except ValueError:
                    pass
        
        chart.createDefaultAxes()
        chart.setTitle("Exponents of Matrix Elements")
        
        self.chart_view.setChart(chart)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MatrixApp()
    window.show()
    sys.exit(app.exec())