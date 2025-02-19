from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from run import Run
import cv2
from database import get_user_id, log_emotion


class RunVideo(QWidget, Run):
    def __init__(self, controller, user):
        QWidget.__init__(self)
        Run.__init__(self)
        self.controller = controller
        self.current_user = user
        self.current_userID = get_user_id(self.controller.conn, self.current_user)
        self.Frame_count = 0 
        self.confidence = 0.0
        
        # Set up the window
        self.setWindowTitle("Run Video")
        self.setGeometry(1920, 1080, 1920, 1080)

        # Layout
        layout = QVBoxLayout()

        # QLabel for video feed
        self.video_label = QLabel()
        self.video_label.setFixedSize(700, 400)
        layout.addWidget(self.video_label)

        # QLabel for emotion prediction
        self.emotion_label = QLabel("Emotion: None")
        self.emotion_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(self.emotion_label)

        # Stop Button
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_video)
        layout.addWidget(self.stop_button)

        self.setLayout(layout)

        # Initialize QTimer for video updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Start video feed
        self.start_video()

    def start_video(self):
        """Start the video feed and load the model."""
        try:
            self.load_modle()  # Load the emotion detection model
            self.vid.open(0)  # Open the default camera
            if not self.vid.isOpened():
                raise RuntimeError("Failed to open the camera.")
            self.timer.start(30)  # Trigger frame updates every 30ms
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not start video: {e}")

    def stop_video(self):
        """Stop the video feed and release resources."""
        self.timer.stop()
        self.vid.release()
        self.video_label.clear()
        self.close()
        if self.current_user == None:
            self.controller.show_welcome_screen()
        else:
            self.controller.show_main_page()

    def update_frame(self):
        """Capture frame, process it, and display it in the QLabel."""
        ret, frame = self.vid.read()
        if not ret:
            self.stop_video()
            return
        
        # Process the frame
        frame_with_faces = self.detect_bounding_box(frame)
        self.Face_crop()
        self.prediction = self.predict_emotion()
        self.Frame_count += 1 

        # Update the emotion label
        if self.prediction:  # Ensure prediction is not None
            self.emotion_label.setText(f"Emotion: {self.prediction}")  # ✅ Update label

        # logs the emotion label
        if self.current_userID != None and self.Frame_count == 25:
            log_emotion(self.controller.conn, self.current_userID, self.prediction, self.confidence)
            self.Frame_count = 0 
        

        # --- Resize the frame to match the QLabel size ---
        desired_width = self.video_label.width()
        desired_height = self.video_label.height()
        frame_resized = cv2.resize(frame_with_faces, (desired_width, desired_height), interpolation=cv2.INTER_AREA)

        # Convert the frame to QImage
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_frame.shape
        bytes_per_line = channel * width
        qimg = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Update QLabel with the QPixmap
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        """Handle the widget close event."""
        self.video_label.setFixedSize(10, 10)
        self.stop_video()
        super().closeEvent(event)
