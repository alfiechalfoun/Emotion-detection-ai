from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QLabel, QDateEdit
from PyQt5.QtCore import QDate
from database import get_emotion_history_by_date
import struct

class HistoryWindow(QWidget):
    def __init__(self, controller, conn, user_id):
        super().__init__()
        self.controller = controller
        self.conn = conn
        self.user_id = user_id
        self.setWindowTitle("Emotion History")
        self.setGeometry(200, 200, 600, 400)

        layout = QVBoxLayout()

        # Label and Date Picker
        self.date_label = QLabel("Select Date:")
        layout.addWidget(self.date_label)

        self.date_picker = QDateEdit()
        self.date_picker.setCalendarPopup(True)
        self.date_picker.setDate(QDate.currentDate())  # Default to today's date
        # ✅ Prevent selecting future dates
        self.date_picker.setMaximumDate(QDate.currentDate())
        self.date_picker.dateChanged.connect(self.load_history)  # Update table on date change
        layout.addWidget(self.date_picker)

        # Table for history
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(3)
        self.history_table.setHorizontalHeaderLabels(["Time", "Emotion", "Confidence"])
        layout.addWidget(self.history_table)

        # back button
        self.back_button = QPushButton("Go Back")
        self.back_button.clicked.connect(self.goto_back)
        layout.addWidget(self.back_button)

        self.setLayout(layout)

        # Load initial history for today's date
        self.load_history()

    def load_history(self):
        """Load user history for the selected date into the table."""
        selected_date = self.date_picker.date().toString("yyyy-MM-dd")
        history_data = get_emotion_history_by_date(self.conn, self.user_id, selected_date)
        self.history_table.setRowCount(len(history_data))
        for row, (timestamp, emotion, confidence) in enumerate(history_data):
            time_only = timestamp.split(" ")[1]  # ✅ Extract only the time (HH:MM:SS)

                    # ✅ Convert stored raw bytes to float properly
            try:
                confidence_float = struct.unpack('f', confidence)[0]  # Convert bytes to float
                confidence_str = f"{confidence_float:.7f}"  # Format float to fit within 10 characters
            except Exception as e:
                print(f"Error decoding confidence: {e}")
                confidence_str = '0.0000000'  # Default if conversion fails


            self.history_table.setItem(row, 0, QTableWidgetItem(time_only))
            self.history_table.setItem(row, 1, QTableWidgetItem(emotion))
            self.history_table.setItem(row, 2, QTableWidgetItem(str(confidence_str)))  # Convert confidence to string

    def goto_back(self):
        self.controller.show_main_page()