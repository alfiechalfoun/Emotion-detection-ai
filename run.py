from Vid.Video import Video
from emotion_detection_model.emotion_ai import Modle
import time
import cv2


class Run(Video, Modle):
    def __init__(self):
        Video.__init__(self)
        Modle.__init__(self)
        self.count = 0

    def predict_emotion(self):
        self.count += 1 
        if self.count == 5:
            self.count = 0
            if hasattr(self, 'resised_face'):
                prediction, confidence = self.predict('CurrentFace.jpg')
                self.change_box_coloure(prediction)
                return(prediction)
            
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
    obj = Run()
    obj.run_video()