import sys
import os
import csv
import re
import time
from datetime import datetime

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QPushButton, QFileDialog, QVBoxLayout,
    QHBoxLayout, QGridLayout, QFrame, QMessageBox,
    QSizePolicy, QTabWidget, QLineEdit, QComboBox,
    QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QFormLayout, QScrollArea
)


class CameraWorker(QThread):
    """Reads an MJPEG/HTTP camera stream away from the GUI thread."""
    frame_received = pyqtSignal(object)
    status_changed = pyqtSignal(str, bool)

    def __init__(self, url, parent=None):
        super().__init__(parent)
        self.url = url
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        cap = None
        try:
            # Use FFmpeg timeouts when the installed OpenCV build supports them.
            # If those options are unavailable, fall back to normal VideoCapture.
            cap = cv2.VideoCapture()
            opened = False
            try:
                opened = cap.open(
                    self.url,
                    cv2.CAP_FFMPEG,
                    [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1500,
                    ],
                )
            except (TypeError, AttributeError, cv2.error):
                opened = False
            if not opened:
                cap.release()
                cap = cv2.VideoCapture(self.url)
            if not cap.isOpened():
                self.status_changed.emit(
                    "● CAMERA CONNECTION FAILED — CHECK URL / WI-FI", False
                )
                return

            self.status_changed.emit("● CONNECTED — WAITING FOR VIDEO", True)
            started_at = time.time()
            consecutive_failures = 0
            got_frame = False

            while self._running:
                ok, frame = cap.read()
                if ok and frame is not None:
                    got_frame = True
                    consecutive_failures = 0
                    self.frame_received.emit(frame.copy())
                else:
                    consecutive_failures += 1
                    if not got_frame and time.time() - started_at > 10:
                        self.status_changed.emit(
                            "● NO VIDEO FRAMES — CHECK STREAM PATH", False
                        )
                        break
                    if consecutive_failures > 100:
                        self.status_changed.emit(
                            "● STREAM INTERRUPTED — CHECK CAMERA", False
                        )
                        break
                    self.msleep(40)
        except Exception as exc:
            self.status_changed.emit(f"● CAMERA ERROR: {exc}", False)
        finally:
            if cap is not None:
                cap.release()
            if not self._running:
                self.status_changed.emit("● CAMERA STOPPED", False)


class GroupBarChart(QWidget):
    """Small in-app bar chart; uses PyQt painting, so matplotlib is not required."""
    GROUPS = [
        "Uncoated",
        "Natural coating A",
        "Natural coating B",
        "Standard coating",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.group_corroguard_data = {group: {"mean": None, "count": 0} for group in self.GROUPS}
        self.setMinimumHeight(290)
        self.setMinimumWidth(560)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)

    def set_data(self, data):
        self.group_data = {
            group: data.get(group, {"mean": None, "count": 0})
            for group in self.GROUPS
        }
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        bounds = self.rect()
        painter.fillRect(bounds, QColor("#192630"))
        painter.setPen(QPen(QColor("#2B414E"), 1))
        painter.drawRoundedRect(bounds.adjusted(0, 0, -1, -1), 12, 12)

        width = bounds.width()
        height = bounds.height()
        left, right, top, bottom = 58, 18, 42, 83
        plot_left = left
        plot_right = max(left + 40, width - right)
        plot_top = top
        plot_bottom = max(top + 40, height - bottom)
        plot_height = plot_bottom - plot_top
        plot_width = plot_right - plot_left

        available_values = [
            float(info["mean"])
            for info in self.group_data.values()
            if info.get("mean") is not None
        ]
        max_value = max(available_values, default=0.0)
        if max_value < 1.0:
            axis_max = 1.0
        elif max_value < 5.0:
            axis_max = min(100.0, float(np.ceil(max_value * 1.25)))
        elif max_value < 20.0:
            axis_max = min(100.0, float(np.ceil(max_value * 1.2 / 5.0) * 5.0))
        elif max_value < 50.0:
            axis_max = min(100.0, float(np.ceil(max_value * 1.2 / 10.0) * 10.0))
        else:
            axis_max = min(100.0, float(np.ceil(max_value * 1.1 / 20.0) * 20.0))
        axis_max = max(1.0, axis_max)

        painter.setPen(QColor("#91A8B5"))
        painter.drawText(12, 22, "MEAN SUSPECTED RUST-COLOR COVERAGE (%)")
        for tick_index in range(5):
            value = axis_max * tick_index / 4.0
            y = int(plot_bottom - plot_height * tick_index / 4.0)
            painter.setPen(QPen(QColor("#2B414E"), 1))
            painter.drawLine(plot_left, y, plot_right, y)
            painter.setPen(QColor("#91A8B5"))
            tick_text = f"{value:.2f}%" if axis_max <= 1.0 else f"{value:g}%"
            painter.drawText(4, y + 5, 48, 18, Qt.AlignRight | Qt.AlignVCenter, tick_text)

        slot_width = plot_width / 4.0
        for index, group in enumerate(self.GROUPS):
            info = self.group_data.get(group, {})
            mean_value = info.get("mean")
            count = int(info.get("count", 0) or 0)
            center_x = plot_left + slot_width * (index + 0.5)
            bar_width = min(58.0, slot_width * 0.48)

            if mean_value is None or count == 0:
                painter.setPen(QColor("#91A8B5"))
                painter.drawText(
                    int(center_x - slot_width * 0.48), plot_bottom - 24,
                    int(slot_width * 0.96), 20,
                    Qt.AlignCenter, "No data"
                )
            else:
                value = max(0.0, min(100.0, float(mean_value)))
                bar_height = plot_height * value / axis_max
                # A true zero remains zero-height; do not imply measured corrosion.
                x = int(center_x - bar_width / 2.0)
                y = int(plot_bottom - bar_height)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor("#00D9FF"))
                if bar_height > 0:
                    painter.drawRoundedRect(
                        x, y, int(bar_width), max(2, int(bar_height)), 4, 4
                    )
                painter.setPen(QColor("#EAF4F8"))
                label_y = max(plot_top + 2, y - 23)
                painter.drawText(
                    int(center_x - slot_width * 0.48), label_y,
                    int(slot_width * 0.96), 20,
                    Qt.AlignCenter, f"{value:.2f}%"
                )
                if value < 1.0:
                    painter.setPen(QColor("#F4C76B"))
                    painter.drawText(
                        int(center_x - slot_width * 0.48), label_y + 17,
                        int(slot_width * 0.96), 18,
                        Qt.AlignCenter, "LOW / PRELIMINARY"
                    )

            if group == "Uncoated":
                group_label = "Uncoated"
            elif group == "Natural coating A":
                group_label = "Natural coating A"
            elif group == "Natural coating B":
                group_label = "Natural coating B"
            else:
                group_label = "Standard coating"
            painter.setPen(QColor("#D0DEE5"))
            painter.drawText(
                int(center_x - slot_width * 0.49), plot_bottom + 8,
                int(slot_width * 0.98), 34,
                Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, group_label
            )
            painter.setPen(QColor("#91A8B5"))
            count_label = f"n = {count}" if count else "n = 0"
            painter.drawText(
                int(center_x - slot_width * 0.49), plot_bottom + 43,
                int(slot_width * 0.98), 18, Qt.AlignCenter, count_label
            )

        painter.end()


class CorroGuard(QMainWindow):
    COATING_GROUPS = [
        "Uncoated", "Natural coating A", "Natural coating B", "Standard coating"
    ]
    LOW_RUST_COLOR_PERCENT = 1.0  # UI flag only; not a scientific corrosion threshold

    def __init__(self):
        super().__init__()

        self.setWindowTitle("CORROGUARD AI-X | Corrosion Monitor")
        self.resize(1250, 850)

        # Existing image-analysis state
        self.image_paths = []
        self.current_index = -1
        self.original_image = None
        self.processed_image = None
        self.last_analyzed_image = None
        self.current_rust_percentage = None
        self.last_analysis_source = "IMAGE"
        self.current_source = "IMAGE"

        # Live camera state
        self.camera_worker = None
        self.live_mode = False
        self.last_live_analysis_time = 0.0

        # Saved research data (created when first needed)
        self.data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "corroguard_data"
        )
        self.images_dir = os.path.join(self.data_dir, "images")
        self.history_path = os.path.join(self.data_dir, "measurements.csv")
        self.records = self.load_records()

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #101820;
                color: #EAF4F8;
                font-family: Segoe UI;
                font-size: 13px;
            }
            QFrame#panel {
                background-color: #192630;
                border: 1px solid #2B414E;
                border-radius: 12px;
            }
            QLabel#title {
                color: #00D9FF;
                font-size: 26px;
                font-weight: bold;
            }
            QLabel#muted { color: #91A8B5; }
            QLabel#metric {
                color: #00D9FF;
                font-size: 27px;
                font-weight: bold;
            }
            QPushButton {
                background-color: #087F9C;
                color: white;
                border: none;
                border-radius: 7px;
                padding: 9px 12px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #09A0BE; }
            QPushButton#secondary {
                background-color: #263A47;
                border: 1px solid #3C5665;
            }
            QPushButton#secondary:hover { background-color: #354E5D; }
            QLineEdit, QComboBox {
                background-color: #0B1218;
                color: #EAF4F8;
                border: 1px solid #3C5665;
                border-radius: 6px;
                padding: 8px;
                selection-background-color: #087F9C;
            }
            QCheckBox { color: #D0DEE5; spacing: 7px; }
            QTabWidget::pane {
                border: 1px solid #2B414E;
                border-radius: 8px;
                top: -1px;
            }
            QTabBar::tab {
                background: #192630;
                color: #91A8B5;
                border: 1px solid #2B414E;
                padding: 10px 18px;
                margin-right: 4px;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                color: #00D9FF;
                background: #263A47;
                border-bottom: 2px solid #00D9FF;
            }
            QTableWidget {
                background-color: #0B1218;
                alternate-background-color: #192630;
                color: #EAF4F8;
                gridline-color: #2B414E;
                border: 1px solid #2B414E;
                border-radius: 8px;
            }
            QHeaderView::section {
                background-color: #263A47;
                color: #00D9FF;
                padding: 7px;
                border: 1px solid #2B414E;
                font-weight: bold;
            }
        """)

        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(22, 18, 22, 18)
        main.setSpacing(12)

        # Global header remains in the same main window
        header = QHBoxLayout()
        heading = QVBoxLayout()
        title = QLabel("CORROGUARD AI-X")
        title.setObjectName("title")
        subtitle = QLabel("AI-ASSISTED STEEL CORROSION MONITORING")
        subtitle.setObjectName("muted")
        heading.addWidget(title)
        heading.addWidget(subtitle)
        self.status = QLabel("● READY")
        self.status.setStyleSheet("color: #70E000; font-weight: bold;")
        header.addLayout(heading)
        header.addStretch()
        header.addWidget(self.status)
        main.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.build_dashboard_tab(), "LIVE MONITOR")
        self.tabs.addTab(self.build_history_tab(), "MONITOR & COMPARE")
        self.tabs.addTab(self.build_prediction_tab(), "PREDICT & PROTECT")
        main.addWidget(self.tabs, 1)

        self.refresh_history_tables()

    # ------------------------------------------------------------------
    # Dashboard tab: preserves the existing dashboard style and controls
    # ------------------------------------------------------------------
    def build_dashboard_tab(self):
        page = QWidget()
        main = QVBoxLayout(page)
        main.setContentsMargins(10, 12, 10, 8)
        main.setSpacing(10)

        # Live camera connection controls, still inside this app window
        camera_panel = QFrame()
        camera_panel.setObjectName("panel")
        camera_layout = QHBoxLayout(camera_panel)
        camera_layout.setContentsMargins(12, 10, 12, 10)
        camera_layout.setSpacing(8)

        camera_label = QLabel("ROBOT STREAM URL")
        camera_label.setStyleSheet("font-weight: bold; color: #00D9FF;")
        self.camera_url = QLineEdit()
        self.camera_url.setPlaceholderText(
            "Example: http://CAMERA-IP:81/stream"
        )
        self.camera_url.setToolTip(
            "Enter the ESP32-CAM video stream URL. The exact path depends on the firmware."
        )
        self.connect_button = QPushButton("Connect Live")
        self.connect_button.clicked.connect(self.start_live_camera)
        self.stop_button = QPushButton("Stop Live")
        self.stop_button.setObjectName("secondary")
        self.stop_button.clicked.connect(self.stop_live_camera)
        self.capture_button = QPushButton("Capture Frame")
        self.capture_button.setObjectName("secondary")
        self.capture_button.clicked.connect(self.capture_frame)
        camera_layout.addWidget(camera_label)
        camera_layout.addWidget(self.camera_url, 1)
        camera_layout.addWidget(self.connect_button)
        camera_layout.addWidget(self.stop_button)
        camera_layout.addWidget(self.capture_button)
        main.addWidget(camera_panel)

        toolbar = QHBoxLayout()
        self.import_button = QPushButton("＋ Import Images")
        self.import_button.clicked.connect(self.import_images)

        self.previous_button = QPushButton("← Previous")
        self.previous_button.setObjectName("secondary")
        self.previous_button.clicked.connect(lambda: self.change_image(-1))

        self.next_button = QPushButton("Next →")
        self.next_button.setObjectName("secondary")
        self.next_button.clicked.connect(lambda: self.change_image(1))

        self.analyze_button = QPushButton("Analyze Sample")
        self.analyze_button.clicked.connect(self.analyze_current)

        self.auto_analyze = QCheckBox("Auto-analyze live feed")
        self.auto_analyze.setChecked(True)
        self.auto_analyze.setToolTip(
            "Estimate rust-colored coverage automatically while the camera is connected."
        )

        toolbar.addWidget(self.import_button)
        toolbar.addWidget(self.previous_button)
        toolbar.addWidget(self.next_button)
        toolbar.addStretch()
        toolbar.addWidget(self.auto_analyze)
        toolbar.addWidget(self.analyze_button)
        main.addLayout(toolbar)

        content = QHBoxLayout()
        content.setSpacing(14)

        image_panel = QFrame()
        image_panel.setObjectName("panel")
        image_layout = QVBoxLayout(image_panel)
        image_title = QLabel("LIVE CAMERA / SAMPLE IMAGE")
        image_title.setStyleSheet("font-weight: bold; color: #00D9FF;")

        self.image_label = QLabel("Connect the robot camera or import a steel sample image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(480, 390)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setStyleSheet(
            "background-color: #0B1218; border-radius: 8px;"
            "color: #91A8B5; padding: 12px;"
        )
        self.filename_label = QLabel("No image or live camera connected")
        self.filename_label.setObjectName("muted")
        image_layout.addWidget(image_title)
        image_layout.addWidget(self.image_label, 1)
        image_layout.addWidget(self.filename_label)

        results_panel = QFrame()
        results_panel.setObjectName("panel")
        results = QVBoxLayout(results_panel)
        results.setSpacing(10)
        results_title = QLabel("ANALYSIS RESULTS")
        results_title.setStyleSheet("font-weight: bold; color: #00D9FF;")
        results.addWidget(results_title)

        self.coverage_value = self.make_metric_card(
            results, "SUSPECTED RUST COVERAGE", "—"
        )
        self.other_value = self.make_metric_card(results, "OTHER PIXELS", "—")
        self.pixel_value = self.make_metric_card(
            results, "SUSPECTED RUST PIXELS", "—"
        )

        details_title = QLabel("INTERPRETATION")
        details_title.setStyleSheet("font-weight: bold; color: #00D9FF;")
        results.addWidget(details_title)
        self.details = QLabel(
            "Connect the robot camera or import an image, then analyze the sample."
        )
        self.details.setWordWrap(True)
        self.details.setAlignment(Qt.AlignTop)
        self.details.setStyleSheet("color: #D0DEE5;")
        results.addWidget(self.details, 1)

        self.method = QLabel(
            "METHOD: HSV COLOR THRESHOLDING\n"
            "STATUS: PRELIMINARY ESTIMATE"
        )
        self.method.setObjectName("muted")
        self.method.setWordWrap(True)
        results.addWidget(self.method)

        content.addWidget(image_panel, 3)
        content.addWidget(results_panel, 2)
        main.addLayout(content, 1)

        footer = QLabel(
            "RESEARCH PROTOTYPE  •  Results depend on lighting, camera settings, "
            "sample appearance, and the selected color threshold."
        )
        footer.setObjectName("muted")
        footer.setAlignment(Qt.AlignCenter)
        main.addWidget(footer)
        return page

    # ------------------------------------------------------------------
    # Monitor & compare tab
    # ------------------------------------------------------------------
    def build_history_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 10)
        layout.setSpacing(12)

        intro = QLabel(
            "RECORD REPEATED MEASUREMENTS and compare recorded rust-color estimates "
            "for uncoated and coated steel samples. Use consistent lighting, camera "
            "distance, and exposure conditions for fair comparisons."
        )
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        layout.addWidget(intro)

        form_panel = QFrame()
        form_panel.setObjectName("panel")
        form = QHBoxLayout(form_panel)
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(10)

        form.addWidget(QLabel("SAMPLE ID"))
        self.sample_id_input = QLineEdit("Steel-01")
        self.sample_id_input.setPlaceholderText("e.g. Steel-01")
        form.addWidget(self.sample_id_input, 1)
        form.addWidget(QLabel("COATING GROUP"))
        self.coating_combo = QComboBox()
        self.coating_combo.addItems(self.COATING_GROUPS)
        form.addWidget(self.coating_combo, 1)
        self.record_button = QPushButton("Record Measurement")
        self.record_button.clicked.connect(self.record_measurement)
        form.addWidget(self.record_button)
        layout.addWidget(form_panel)

        storage_note = QLabel(
            "Each saved measurement is added to corroguard_data/measurements.csv; "
            "a matching image is saved in corroguard_data/images/."
        )
        storage_note.setObjectName("muted")
        storage_note.setWordWrap(True)
        layout.addWidget(storage_note)

        # Side-by-side comparison of two chosen images, in the same app window.
        compare_panel = QFrame()
        compare_panel.setObjectName("panel")
        compare_layout = QVBoxLayout(compare_panel)
        compare_layout.setContentsMargins(12, 12, 12, 12)
        compare_layout.setSpacing(9)
        compare_header = QHBoxLayout()
        compare_title = QLabel("SIDE-BY-SIDE IMAGE COMPARISON")
        compare_title.setStyleSheet("font-weight: bold; color: #00D9FF;")
        self.compare_images_button = QPushButton("Choose 2 Images & Compare")
        self.compare_images_button.clicked.connect(self.compare_two_images)
        compare_header.addWidget(compare_title)
        compare_header.addStretch()
        compare_header.addWidget(self.compare_images_button)
        compare_layout.addLayout(compare_header)

        preview_row = QHBoxLayout()
        self.compare_preview_a = QLabel("IMAGE A\nNot selected")
        self.compare_preview_b = QLabel("IMAGE B\nNot selected")
        for preview in (self.compare_preview_a, self.compare_preview_b):
            preview.setAlignment(Qt.AlignCenter)
            preview.setMinimumSize(280, 170)
            preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            preview.setStyleSheet(
                "background-color: #0B1218; border-radius: 8px; "
                "color: #91A8B5; padding: 8px;"
            )
        preview_row.addWidget(self.compare_preview_a, 1)
        preview_row.addWidget(self.compare_preview_b, 1)
        compare_layout.addLayout(preview_row)
        self.compare_summary = QLabel(
            "Choose two images to compare their preliminary rust-colored coverage estimates."
        )
        self.compare_summary.setObjectName("muted")
        self.compare_summary.setWordWrap(True)
        compare_layout.addWidget(self.compare_summary)
        layout.addWidget(compare_panel)

        records_title = QLabel("MEASUREMENT HISTORY")
        records_title.setStyleSheet("font-weight: bold; color: #00D9FF;")
        layout.addWidget(records_title)
        self.history_table = QTableWidget(0, 5)
        self.history_table.setHorizontalHeaderLabels(
            ["Date / Time", "Sample ID", "Coating Group", "Rust Estimate", "Source"]
        )
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.verticalHeader().setVisible(False)
        layout.addWidget(self.history_table, 3)

        compare_title = QLabel("COATING GROUP SUMMARY")
        compare_title.setStyleSheet("font-weight: bold; color: #00D9FF;")
        layout.addWidget(compare_title)
        self.comparison_table = QTableWidget(0, 3)
        self.comparison_table.setHorizontalHeaderLabels(
            ["Coating Group", "Measurements", "Mean Rust Estimate"]
        )
        self.comparison_table.setAlternatingRowColors(True)
        self.comparison_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.comparison_table.verticalHeader().setVisible(False)
        self.comparison_table.setMaximumHeight(180)
        layout.addWidget(self.comparison_table)

        graph_title = QLabel("FOUR-GROUP COMPARISON GRAPH")
        graph_title.setStyleSheet("font-weight: bold; color: #00D9FF;")
        layout.addWidget(graph_title)
        self.group_chart = GroupBarChart()
        layout.addWidget(self.group_chart)

        graph_note = QLabel(
            "Bars show the mean of recorded rust-colored coverage estimates for each group. "
            "Groups with no saved measurements are marked No data. For means below 1%, "
            "the graph flags the result as LOW / PRELIMINARY. The chart scale adjusts to "
            "make small values easier to see; compare only measurements collected under similar conditions."
        )
        graph_note.setWordWrap(True)
        graph_note.setObjectName("muted")
        layout.addWidget(graph_note)

        compare_note = QLabel(
            "Group averages are descriptive only. LOW* means the estimate is below 1.00% "
            "and should be treated as a low-signal preliminary result—not proof that rust "
            "is absent. Compare samples exposed for similar durations and under the same "
            "conditions; image color is not a validated measurement of actual metal loss."
        )
        compare_note.setObjectName("muted")
        compare_note.setWordWrap(True)
        layout.addWidget(compare_note)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(page)
        return scroll

    # ------------------------------------------------------------------
    # Prediction & protection tab
    # ------------------------------------------------------------------
    def build_prediction_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 10)
        layout.setSpacing(14)

        title = QLabel("CORROSION TREND ESTIMATE")
        title.setStyleSheet("font-weight: bold; color: #00D9FF; font-size: 17px;")
        layout.addWidget(title)

        intro = QLabel(
            "CorroGuard can calculate a simple linear trend from measurements saved for "
            "one sample. This is an experimental extrapolation—not a trained AI model "
            "and not a validated prediction. Collect measurements across multiple times "
            "before using this feature."
        )
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        layout.addWidget(intro)

        forecast_panel = QFrame()
        forecast_panel.setObjectName("panel")
        forecast_layout = QVBoxLayout(forecast_panel)
        forecast_layout.setContentsMargins(16, 16, 16, 16)
        row = QHBoxLayout()
        row.addWidget(QLabel("SAMPLE ID"))
        self.forecast_sample_input = QLineEdit("Steel-01")
        self.forecast_sample_input.setPlaceholderText("Same ID used when recording results")
        row.addWidget(self.forecast_sample_input, 1)
        self.forecast_button = QPushButton("Estimate Next Trend Point")
        self.forecast_button.clicked.connect(self.estimate_trend)
        row.addWidget(self.forecast_button)
        forecast_layout.addLayout(row)

        self.forecast_result = QLabel(
            "At least three saved measurements at distinct times are needed for a trend estimate."
        )
        self.forecast_result.setWordWrap(True)
        self.forecast_result.setStyleSheet("color: #D0DEE5; font-size: 14px;")
        forecast_layout.addWidget(self.forecast_result)
        layout.addWidget(forecast_panel)

        protect_title = QLabel("PROTECTIVE COATING EXPERIMENT")
        protect_title.setStyleSheet("font-weight: bold; color: #00D9FF; font-size: 17px;")
        layout.addWidget(protect_title)

        protect_panel = QFrame()
        protect_panel.setObjectName("panel")
        protect_layout = QVBoxLayout(protect_panel)
        protect_layout.setContentsMargins(16, 14, 16, 14)
        protect_text = QLabel(
            "• Prepare comparable steel samples and label each one.\n"
            "• Include an uncoated control and the coating groups being tested.\n"
            "• Keep exposure time, salt concentration, camera angle, distance, and lighting consistent.\n"
            "• Take repeated measurements and, where practical, record mass before and after exposure.\n"
            "• Compare the results only after data is collected; do not assume a coating works in advance."
        )
        protect_text.setWordWrap(True)
        protect_text.setStyleSheet("color: #D0DEE5; line-height: 1.5;")
        protect_layout.addWidget(protect_text)
        layout.addWidget(protect_panel)

        limitation = QLabel(
            "SCIENTIFIC LIMITATION: Orange/brown color can come from lighting, reflections, "
            "or other materials. This version estimates color coverage, not corrosion depth, "
            "metal loss, or confirmed rust. The trend calculation is only as reliable as the "
            "measurement consistency and amount of data."
        )
        limitation.setWordWrap(True)
        limitation.setStyleSheet("color: #91A8B5; border: 1px solid #2B414E; padding: 10px; border-radius: 8px;")
        layout.addWidget(limitation)
        layout.addStretch(1)
        return page

    # ------------------------------------------------------------------
    # Reusable UI/data helpers
    # ------------------------------------------------------------------
    def make_metric_card(self, layout, heading, value):
        card = QFrame()
        card.setObjectName("panel")
        card_layout = QVBoxLayout(card)
        label = QLabel(heading)
        label.setObjectName("muted")
        number = QLabel(value)
        number.setObjectName("metric")
        card_layout.addWidget(label)
        card_layout.addWidget(number)
        layout.addWidget(card)
        return number

    def set_status(self, text, success=False):
        self.status.setText(text)
        color = "#70E000" if success else "#00D9FF"
        self.status.setStyleSheet(f"color: {color}; font-weight: bold;")

    def load_records(self):
        if not os.path.exists(self.history_path):
            return []
        loaded = []
        try:
            with open(self.history_path, "r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    try:
                        row["rust_percentage"] = float(row["rust_percentage"])
                        loaded.append(row)
                    except (KeyError, TypeError, ValueError):
                        continue
        except OSError:
            pass
        return loaded

    def save_records_file(self):
        os.makedirs(self.data_dir, exist_ok=True)
        fields = [
            "timestamp", "sample_id", "coating", "rust_percentage",
            "source", "image_file"
        ]
        with open(self.history_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for record in self.records:
                writer.writerow({key: record.get(key, "") for key in fields})

    def refresh_history_tables(self):
        if not hasattr(self, "history_table"):
            return

        latest_first = sorted(
            self.records, key=lambda item: item.get("timestamp", ""), reverse=True
        )
        self.history_table.setRowCount(len(latest_first))
        for row_index, record in enumerate(latest_first):
            try:
                stamp = datetime.fromisoformat(record.get("timestamp", "")).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            except (ValueError, TypeError):
                stamp = record.get("timestamp", "")
            try:
                value = float(record.get("rust_percentage", 0.0))
            except (ValueError, TypeError):
                value = 0.0
            estimate_label = f"{value:.2f}%"
            if value < self.LOW_RUST_COLOR_PERCENT:
                estimate_label += " (LOW*)"
            values = [
                stamp,
                record.get("sample_id", ""),
                record.get("coating", ""),
                estimate_label,
                record.get("source", ""),
            ]
            for col, value_text in enumerate(values):
                item = QTableWidgetItem(str(value_text))
                if col == 3:
                    item.setTextAlignment(Qt.AlignCenter)
                self.history_table.setItem(row_index, col, item)

        grouped = {group: [] for group in self.COATING_GROUPS}
        for record in self.records:
            group = record.get("coating", "Unknown")
            if group not in grouped:
                grouped[group] = []
            try:
                grouped[group].append(float(record.get("rust_percentage", 0.0)))
            except (ValueError, TypeError):
                continue

        self.comparison_table.setRowCount(len(self.COATING_GROUPS))
        chart_data = {}
        for row_index, group in enumerate(self.COATING_GROUPS):
            values = grouped.get(group, [])
            mean_value = float(np.mean(values)) if values else None
            if mean_value is None:
                mean_text = "No data"
            else:
                mean_text = f"{mean_value:.2f}%"
                if mean_value < self.LOW_RUST_COLOR_PERCENT:
                    mean_text += " (LOW*)"
            cells = [group, str(len(values)), mean_text]
            for col, value_text in enumerate(cells):
                self.comparison_table.setItem(
                    row_index, col, QTableWidgetItem(str(value_text))
                )
            chart_data[group] = {"mean": mean_value, "count": len(values)}

        if hasattr(self, "group_chart"):
            self.group_chart.set_data(chart_data)

    # ------------------------------------------------------------------
    # Existing image import/navigation remains available
    # ------------------------------------------------------------------
    def import_images(self):
        # Switch from live mode to imported-image mode before loading files.
        self.stop_live_camera(update_status=False)
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Steel Sample Images",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if not paths:
            return
        self.image_paths = paths
        self.current_index = 0
        self.load_current_image()
        self.tabs.setCurrentIndex(0)

    def change_image(self, direction):
        if not self.image_paths:
            QMessageBox.information(
                self, "No images", "Please import images first."
            )
            return
        new_index = self.current_index + direction
        if 0 <= new_index < len(self.image_paths):
            self.current_index = new_index
            self.load_current_image()

    def load_current_image(self):
        path = self.image_paths[self.current_index]
        image = cv2.imread(path)
        if image is None:
            QMessageBox.warning(self, "Image error", "Could not read this image.")
            return

        self.original_image = image
        self.processed_image = None
        self.last_analyzed_image = None
        self.current_rust_percentage = None
        self.current_source = "IMAGE"
        filename = path.replace("/", os.sep).split(os.sep)[-1]
        self.filename_label.setText(
            f"{self.current_index + 1} of {len(self.image_paths)}  |  {filename}"
        )
        self.show_image(image)
        self.coverage_value.setText("—")
        self.other_value.setText("—")
        self.pixel_value.setText("—")
        self.details.setText(
            "Image loaded. Click Analyze Sample to calculate the preliminary "
            "rust-colored area. Background pixels are included in the percentage."
        )
        self.method.setText(
            "METHOD: HSV COLOR THRESHOLDING\n"
            "STATUS: PRELIMINARY ESTIMATE"
        )
        self.set_status("● IMAGE LOADED", True)

    # ------------------------------------------------------------------
    # Live robot camera
    # ------------------------------------------------------------------
    def start_live_camera(self):
        url = self.camera_url.text().strip()
        if not url:
            self.set_status("● ENTER THE ROBOT CAMERA STREAM URL", False)
            self.details.setText(
                "Enter your ESP32-CAM stream URL above. A common format for the "
                "standard ESP32 CameraWebServer is http://CAMERA-IP:81/stream, "
                "but your kit firmware may use a different path."
            )
            return
        if not (url.startswith("http://") or url.startswith("https://")):
            self.set_status("● STREAM URL MUST START WITH HTTP:// OR HTTPS://", False)
            return

        if not self.stop_live_camera(update_status=False):
            self.set_status("● PREVIOUS CAMERA CONNECTION IS STILL CLOSING", False)
            return
        self.live_mode = True
        self.current_source = "LIVE CAMERA"
        self.current_rust_percentage = None
        self.last_analyzed_image = None
        self.processed_image = None
        self.coverage_value.setText("—")
        self.other_value.setText("—")
        self.pixel_value.setText("—")
        self.filename_label.setText("Connecting to robot camera...")
        self.set_status("● CONNECTING TO ROBOT CAMERA", False)

        self.camera_worker = CameraWorker(url, self)
        self.camera_worker.frame_received.connect(self.on_live_frame)
        self.camera_worker.status_changed.connect(self.on_camera_status)
        self.camera_worker.start()

    def on_camera_status(self, message, success):
        self.set_status(message, success)
        if not success and "STOPPED" not in message:
            self.details.setText(
                message + "\n\nCheck that the ESP32-CAM is connected to Wi-Fi, "
                "that the laptop is on the same network, and that the stream URL "
                "uses the correct path for your firmware."
            )

    def on_live_frame(self, frame):
        if not self.live_mode:
            return
        self.original_image = frame.copy()
        self.current_source = "LIVE CAMERA"
        self.filename_label.setText(
            f"LIVE CAMERA  |  {frame.shape[1]} × {frame.shape[0]} pixels"
        )

        if self.auto_analyze.isChecked():
            now = time.monotonic()
            # Process roughly twice per second to keep the GUI responsive.
            if now - self.last_live_analysis_time >= 0.5:
                self.last_live_analysis_time = now
                self.analyze_current()
        else:
            # Keep the camera visibly live when automatic analysis is paused.
            self.processed_image = None
            self.show_image(frame)

    def stop_live_camera(self, update_status=True):
        self.live_mode = False
        worker = self.camera_worker
        if worker is not None:
            if worker.isRunning():
                worker.stop()
                worker.wait(3000)
            if worker.isRunning():
                # Keep the reference alive rather than destroying a running QThread.
                if update_status:
                    self.set_status("● CAMERA IS STILL CLOSING — TRY AGAIN", False)
                return False
            self.camera_worker = None
        if update_status:
            self.set_status("● CAMERA STOPPED", False)
            if self.original_image is not None:
                self.show_image(
                    self.processed_image
                    if self.processed_image is not None
                    else self.original_image
                )
        return True

    def capture_frame(self):
        if self.original_image is None:
            QMessageBox.information(
                self, "No frame", "Connect the robot camera or import an image first."
            )
            return
        try:
            os.makedirs(self.images_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = os.path.join(self.images_dir, f"capture_{stamp}.jpg")
            if not cv2.imwrite(path, self.original_image):
                raise OSError("OpenCV could not write the image file.")
            self.filename_label.setText(f"Captured frame saved: {os.path.basename(path)}")
            self.set_status("● FRAME CAPTURED", True)
        except (OSError, cv2.error) as exc:
            self.set_status("● CAPTURE FAILED", False)
            self.details.setText(f"Could not save the frame: {exc}")

    # ------------------------------------------------------------------
    # Existing HSV rust-colored pixel analysis, reused for image and live feed
    # ------------------------------------------------------------------
    def estimate_rust_color_coverage(self, image):
        """Return mask and rust-colored pixel statistics for an image."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_rust = np.array([5, 45, 35])
        upper_rust = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_rust, upper_rust)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        total_pixels = int(mask.size)
        rust_pixels = int(cv2.countNonZero(mask))
        percentage = (rust_pixels / total_pixels * 100.0) if total_pixels else 0.0
        return mask, percentage, rust_pixels, total_pixels

    def make_rust_overlay(self, image, mask):
        overlay = image.copy()
        overlay[mask > 0] = [0, 0, 255]
        return cv2.addWeighted(image, 0.75, overlay, 0.25, 0)

    def low_signal_message(self, percentage):
        if percentage < self.LOW_RUST_COLOR_PERCENT:
            return (
                "LOW RUST-COLOR SIGNAL — PRELIMINARY: less than 1.00% of pixels "
                "matched the selected orange/brown range. This may mean little visible "
                "rust, but lighting, reflections, image framing, or the color threshold "
                "may also explain the low value. It does not prove corrosion is absent."
            )
        return (
            "PRELIMINARY COLOR-BASED ESTIMATE: highlighted pixels match the selected "
            "orange/brown range; they are not confirmed rust. Results depend on lighting, "
            "camera settings, and the materials in the image."
        )

    def analyze_current(self):
        if self.original_image is None:
            QMessageBox.information(
                self, "No image", "Connect the camera or import a sample image first."
            )
            return

        analysis_image = self.original_image.copy()
        mask, rust_percentage, rust_pixels, total_pixels = \
            self.estimate_rust_color_coverage(analysis_image)
        other_percentage = 100.0 - rust_percentage
        result = self.make_rust_overlay(analysis_image, mask)

        self.last_analyzed_image = analysis_image.copy()
        self.current_rust_percentage = rust_percentage
        self.last_analysis_source = self.current_source
        self.processed_image = result
        self.show_image(result)

        self.coverage_value.setText(f"{rust_percentage:.2f}%")
        self.other_value.setText(f"{other_percentage:.2f}%")
        self.pixel_value.setText(f"{rust_pixels:,}")
        source_description = (
            "live camera frame" if self.current_source == "LIVE CAMERA"
            else "entire imported image"
        )
        self.details.setText(
            f"Suspected rust-colored pixels occupy {rust_percentage:.2f}% of the "
            f"{source_description}.\n\n"
            f"Other pixels: {other_percentage:.2f}%.\n\n"
            f"{self.low_signal_message(rust_percentage)}\n\n"
            "The full frame is analyzed, so background pixels are included."
        )
        if rust_percentage < self.LOW_RUST_COLOR_PERCENT:
            self.method.setText(
                "METHOD: HSV COLOR THRESHOLDING\n"
                "STATUS: LOW RUST-COLOR SIGNAL — PRELIMINARY (<1.00%)\n"
                "NOT PROOF THAT CORROSION IS ABSENT"
            )
        else:
            self.method.setText(
                "METHOD: HSV COLOR THRESHOLDING\n"
                "RED OVERLAY: SUSPECTED RUST-LIKE COLORS\n"
                "STATUS: PRELIMINARY ESTIMATE; BACKGROUND INCLUDED"
            )
        self.set_status(
            "● LIVE ANALYSIS UPDATED" if self.current_source == "LIVE CAMERA"
            else "● ANALYSIS COMPLETE", True
        )

    def compare_two_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Choose Exactly Two Steel Sample Images",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if not paths:
            return
        if len(paths) != 2:
            QMessageBox.information(
                self, "Choose two images",
                "Please select exactly two images to compare."
            )
            return

        images = [cv2.imread(path) for path in paths]
        if any(image is None for image in images):
            QMessageBox.warning(
                self, "Image error", "At least one selected image could not be read."
            )
            return

        stats = []
        for image in images:
            mask, percentage, rust_pixels, total_pixels = \
                self.estimate_rust_color_coverage(image)
            preview = self.make_rust_overlay(image, mask)
            stats.append((percentage, rust_pixels, total_pixels))
            self.set_comparison_preview(preview, len(stats) - 1)

        for i, (percentage, rust_pixels, total_pixels) in enumerate(stats):
            low_note = " — LOW / PRELIMINARY" if percentage < self.LOW_RUST_COLOR_PERCENT else " — PRELIMINARY"
            if i == 0:
                self.compare_preview_a.setToolTip(paths[i])
            else:
                self.compare_preview_b.setToolTip(paths[i])
            # Caption labels live directly below each preview in compare_summary.
        difference = stats[1][0] - stats[0][0]
        sign = "+" if difference > 0 else ""
        a_name = os.path.basename(paths[0])
        b_name = os.path.basename(paths[1])
        a_flag = "LOW / PRELIMINARY" if stats[0][0] < self.LOW_RUST_COLOR_PERCENT else "PRELIMINARY"
        b_flag = "LOW / PRELIMINARY" if stats[1][0] < self.LOW_RUST_COLOR_PERCENT else "PRELIMINARY"
        self.compare_summary.setText(
            f"A: {a_name} — {stats[0][0]:.2f}% ({a_flag}); "
            f"B: {b_name} — {stats[1][0]:.2f}% ({b_flag}).\n"
            f"Difference (B − A): {sign}{difference:.2f} percentage points.\n"
            "These are rust-colored pixel estimates across each full image, including background. "
            "For a fair comparison, use similar lighting, camera angle, distance, and framing. "
            "A low value does not prove corrosion is absent."
        )
        self.set_status("● TWO IMAGES COMPARED", True)

    def set_comparison_preview(self, image, index):
        preview = self.compare_preview_a if index == 0 else self.compare_preview_b
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage)
        preview.setPixmap(
            pixmap.scaled(
                preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    # ------------------------------------------------------------------
    # Log and compare measurements
    # ------------------------------------------------------------------
    def record_measurement(self):
        if self.current_rust_percentage is None or self.last_analyzed_image is None:
            QMessageBox.information(
                self, "No analysis result",
                "Analyze an image first, or connect the camera with Auto-analyze live feed enabled."
            )
            return

        sample_id = self.sample_id_input.text().strip()
        if not sample_id:
            QMessageBox.information(self, "Sample ID required", "Enter a sample ID first.")
            return

        try:
            os.makedirs(self.images_dir, exist_ok=True)
            timestamp = datetime.now().isoformat(timespec="seconds")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", sample_id).strip("_") or "sample"
            image_name = f"{safe_id}_{stamp}.jpg"
            image_path = os.path.join(self.images_dir, image_name)
            if not cv2.imwrite(image_path, self.last_analyzed_image):
                raise OSError("OpenCV could not save the measurement image.")

            record = {
                "timestamp": timestamp,
                "sample_id": sample_id,
                "coating": self.coating_combo.currentText(),
                "rust_percentage": float(self.current_rust_percentage),
                "source": self.last_analysis_source,
                "image_file": os.path.relpath(image_path, self.data_dir),
            }
            self.records.append(record)
            self.save_records_file()
            self.refresh_history_tables()
            self.set_status("● MEASUREMENT RECORDED", True)
            self.details.setText(
                f"Recorded {sample_id} ({record['coating']}) at "
                f"{record['rust_percentage']:.2f}% suspected rust-colored coverage. "
                "Open MONITOR & COMPARE to review the history."
            )
            self.tabs.setCurrentIndex(1)
        except (OSError, cv2.error) as exc:
            self.set_status("● RECORD FAILED", False)
            QMessageBox.warning(self, "Save error", f"Could not record measurement:\n{exc}")

    def estimate_trend(self):
        sample_id = self.forecast_sample_input.text().strip()
        if not sample_id:
            self.forecast_result.setText("Enter a sample ID first.")
            return

        sample_records = [
            record for record in self.records
            if record.get("sample_id", "").strip().casefold() == sample_id.casefold()
        ]
        usable = []
        for record in sample_records:
            try:
                when = datetime.fromisoformat(record["timestamp"])
                value = float(record["rust_percentage"])
                usable.append((when, value))
            except (KeyError, ValueError, TypeError):
                continue
        usable.sort(key=lambda item: item[0])

        if len(usable) < 3 or len(set(item[0] for item in usable)) < 3:
            self.forecast_result.setText(
                f"Not enough data for {sample_id}. Record at least three measurements "
                "at three distinct times for that same sample."
            )
            return

        start_time = usable[0][0]
        x_days = np.array([
            (when - start_time).total_seconds() / 86400.0 for when, _ in usable
        ], dtype=float)
        y_values = np.array([value for _, value in usable], dtype=float)

        if np.ptp(x_days) <= 0:
            self.forecast_result.setText(
                "Measurements need distinct times before a trend can be estimated."
            )
            return

        slope, intercept = np.polyfit(x_days, y_values, 1)
        intervals = np.diff(x_days)
        positive_intervals = intervals[intervals > 0]
        step_days = float(np.median(positive_intervals)) if len(positive_intervals) else 0.0
        if step_days <= 0:
            self.forecast_result.setText(
                "The time spacing is not sufficient to estimate a next measurement point."
            )
            return

        next_x = x_days[-1] + step_days
        estimated = float(np.clip(slope * next_x + intercept, 0.0, 100.0))
        latest_date = usable[-1][0]
        forecast_date = latest_date + (usable[-1][0] - usable[-2][0])
        trend_word = "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "roughly stable"
        self.forecast_result.setText(
            f"Sample: {sample_id}\n"
            f"Measurements used: {len(usable)}\n"
            f"Latest estimate: {y_values[-1]:.2f}% on {latest_date.strftime('%Y-%m-%d %H:%M')}\n"
            f"Observed linear trend: {trend_word} ({slope:.2f} percentage points per day)\n"
            f"Illustrative next-point estimate: {estimated:.2f}% around "
            f"{forecast_date.strftime('%Y-%m-%d %H:%M')}.\n\n"
            "This is a simple linear extrapolation of recorded color-coverage values, "
            "not a validated AI prediction. Treat it as a demonstration until tested "
            "against later measurements."
        )

    # ------------------------------------------------------------------
    # Image rendering / resize / close
    # ------------------------------------------------------------------
    def show_image(self, image):
        if image is None or not hasattr(self, "image_label"):
            return
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        qimage = QImage(
            rgb.data, width, height, bytes_per_line, QImage.Format_RGB888
        ).copy()
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.original_image is not None and hasattr(self, "image_label"):
            image = (
                self.processed_image
                if self.processed_image is not None
                else self.original_image
            )
            self.show_image(image)
        if hasattr(self, "compare_preview_a"):
            for preview in (self.compare_preview_a, self.compare_preview_b):
                pixmap = preview.pixmap()
                if pixmap is not None and not pixmap.isNull():
                    preview.setPixmap(
                        pixmap.scaled(
                            preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                        )
                    )

    def closeEvent(self, event):
        self.live_mode = False
        worker = self.camera_worker
        if worker is not None and worker.isRunning():
            worker.stop()
            worker.wait(3000)
            if worker.isRunning():
                # Last-resort shutdown so the application does not leave a QThread alive.
                worker.terminate()
                worker.wait(1000)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CorroGuard()
    window.show()
    sys.exit(app.exec_())
