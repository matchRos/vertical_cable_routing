from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class MainWindow(QMainWindow):
    def __init__(self, controller) -> None:
        super().__init__()
        self.controller = controller

        self.setWindowTitle("Cable Routing Studio")
        self.resize(1400, 800)

        self.step_table = QTableWidget()
        self.step_table.setColumnCount(2)
        self.step_table.setHorizontalHeaderLabels(["Step", "Result"])
        self.step_table.verticalHeader().setVisible(False)
        self.step_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.step_table.setSelectionMode(QTableWidget.SingleSelection)
        self.step_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.step_table.setAlternatingRowColors(True)
        self.step_table.horizontalHeader().setStretchLastSection(True)
        self.step_table.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("border: 1px solid gray;")

        self.next_button = QPushButton("Next Step")
        self.run_selected_button = QPushButton("Run Selected")
        self.jump_pointer_button = QPushButton("Auto-run to selected")
        self.reset_button = QPushButton("Reset")
        self.save_trace_button = QPushButton("Save Cable Trace")
        self.load_trace_button = QPushButton("Load Cable Trace")
        self.save_checkpoint_button = QPushButton("Save Checkpoint")
        self.load_checkpoint_button = QPushButton("Load Checkpoint")

        self.trace_mode_combo = QComboBox()
        self.trace_mode_combo.addItem("Auto from config", "auto_from_config")
        self.trace_mode_combo.addItem("Auto white rings (first clip)", "auto_white_rings_from_clip")
        self.trace_mode_combo.addItem("Auto from clip A", "auto_from_clip_a")
        self.trace_mode_combo.addItem("Manual two clicks", "manual_two_clicks")

        self._build_layout()
        self._connect_signals()

    def _build_layout(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Pipeline Steps"))
        left_layout.addWidget(self.step_table)

        center_layout = QVBoxLayout()
        center_layout.addWidget(QLabel("Logs"))
        center_layout.addWidget(self.log_box)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Current Visualization"))
        right_layout.addWidget(self.image_label, stretch=1)
        right_layout.addWidget(self.next_button)
        right_layout.addWidget(self.run_selected_button)
        right_layout.addWidget(self.jump_pointer_button)
        right_layout.addWidget(QLabel("Trace Start Mode"))
        right_layout.addWidget(self.trace_mode_combo)
        right_layout.addWidget(self.save_trace_button)
        right_layout.addWidget(self.load_trace_button)
        right_layout.addWidget(self.save_checkpoint_button)
        right_layout.addWidget(self.load_checkpoint_button)
        right_layout.addWidget(self.reset_button)

        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addLayout(center_layout, stretch=2)
        main_layout.addLayout(right_layout, stretch=3)

    def _connect_signals(self) -> None:
        self.next_button.clicked.connect(self.controller.on_next_step)
        self.run_selected_button.clicked.connect(self.controller.on_run_selected)
        self.jump_pointer_button.clicked.connect(self.controller.on_auto_run_to_selected)
        self.reset_button.clicked.connect(self.controller.on_reset)
        self.save_trace_button.clicked.connect(self.controller.on_save_trace)
        self.load_trace_button.clicked.connect(self.controller.on_load_trace)
        self.save_checkpoint_button.clicked.connect(self.controller.on_save_checkpoint)
        self.load_checkpoint_button.clicked.connect(self.controller.on_load_checkpoint)
        self.trace_mode_combo.currentIndexChanged.connect(self.controller.on_trace_start_mode_changed)

    def populate_step_table(self, step_names) -> None:
        self.step_table.setRowCount(len(step_names))
        for row, step_name in enumerate(step_names):
            self.step_table.setItem(row, 0, QTableWidgetItem(step_name))
            self.step_table.setItem(row, 1, QTableWidgetItem(""))
        self.step_table.resizeColumnsToContents()

    def selected_step_name(self) -> str:
        row = self.step_table.currentRow()
        if row < 0:
            return ""
        item = self.step_table.item(row, 0)
        return item.text() if item is not None else ""

    def set_current_step(self, step_name: str) -> None:
        for row in range(self.step_table.rowCount()):
            item = self.step_table.item(row, 0)
            if item is not None and item.text() == step_name:
                self.step_table.selectRow(row)
                self.step_table.scrollToItem(item)
                return

    def clear_step_results(self) -> None:
        for row in range(self.step_table.rowCount()):
            item = self.step_table.item(row, 1)
            if item is None:
                item = QTableWidgetItem("")
                self.step_table.setItem(row, 1, item)
            item.setText("")
            item.setBackground(QBrush())

    def set_step_result(self, step_name: str, text: str, color_hex: str) -> None:
        for row in range(self.step_table.rowCount()):
            name_item = self.step_table.item(row, 0)
            if name_item is None or name_item.text() != step_name:
                continue
            result_item = self.step_table.item(row, 1)
            if result_item is None:
                result_item = QTableWidgetItem("")
                self.step_table.setItem(row, 1, result_item)
            result_item.setText(text)
            result_item.setBackground(QBrush(QColor(color_hex)))
            self.step_table.resizeColumnToContents(0)
            return

    def ask_save_trace_path(self) -> str:
        path, _ = QFileDialog.getSaveFileName(self, "Save Cable Trace", "cable_trace.csv", "CSV Files (*.csv)")
        return path

    def ask_load_trace_path(self) -> str:
        path, _ = QFileDialog.getOpenFileName(self, "Load Cable Trace", "", "CSV Files (*.csv)")
        return path

    def ask_save_checkpoint_path(self) -> str:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Studio Checkpoint",
            "studio_checkpoint.pkl",
            "Studio Checkpoints (*.pkl)",
        )
        return path

    def ask_load_checkpoint_path(self) -> str:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Studio Checkpoint",
            "",
            "Studio Checkpoints (*.pkl)",
        )
        return path

    def confirm_checkpoint_joint_mismatch(self, message: str) -> bool:
        reply = QMessageBox.warning(
            self,
            "YuMi Position Mismatch",
            message + "\n\nLoad checkpoint anyway?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes
