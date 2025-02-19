from Vid.Video import Video
from emotion_detection_model.emotion_ai import Modle
import time
import cv2

class main(Video, Modle):
    def __init__(self):
        Video.__init__(self)
        Modle.__init__(self)
        self.count = 0

    def initUI(self):
        # Create instances of screens
        self.welcome_screen = StartupPage(self)
        self.login_page = LoginPage(self)
        self.signup_page = SignUpWindow(self)
        self.main_page = MainPage(self, self.current_user)
        # self.video = RunVideo(self, self.current_user)

        # Add widgets to the stack
        self.addWidget(self.welcome_screen)
        self.addWidget(self.login_page)
        self.addWidget(self.signup_page)
        self.addWidget(self.main_page)

        # Set the initial screen
        self.setCurrentWidget(self.welcome_screen)  
    
    def show_video(self):
        self.video = RunVideo(self, self.current_user)
        self.addWidget(self.video)
        self.setCurrentWidget(self.video)

    def run_video(self):
        self.run = True
        self.load_modle()
        while self.run:
            ret, self.frame = self.vid.read()
            if not ret:
                print('Error, failed to get the frame.')
                break

            frame_with_faces = self.detect_bounding_box(self.frame)
            cv2.imshow('Video', frame_with_faces)

            self.Face_crop()
            self.predict_emotion()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.run = False
    

if __name__ == '__main__':
    obj = main()
    obj.run_video()