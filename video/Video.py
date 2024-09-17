import cv2
from maxpooling import Maxpooling

class Video(Maxpooling):
    def __init__(self):
        self.face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.vid = cv2.VideoCapture(0)
        if not self.vid.isOpened():
            print('Error, cannot find camera.')

    def detect_bounding_box(self, vid):
        self.gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
        self.faces = self.face_classifier.detectMultiScale(self.gray_image, 1.1, 4, minSize=(200, 200))
        for (x, y, w, h) in self.faces:
            cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return vid
    
    def Face_crop(self):
        if len(self.faces) > 0:
            for (x, y, w, h) in self.faces:
                self.face_crop = self.gray_image[y:y+h, x:x+w]
                self.resised_face = super().resize_with_max_pooling(self.face_crop)
            cv2.imwrite('CurrentFace.jpg', self.resised_face)


    def run(self):
        self.run = True
        while self.run:
            ret, self.frame = self.vid.read()
            if not ret:
                print('Error, failed to get the frame.')
                break

            frame_with_faces = self.detect_bounding_box(self.frame)
            cv2.imshow('Video', frame_with_faces)

            self.Face_crop()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.run = False

    def release_resources(self):
        self.vid.release()
        cv2.destroyAllWindows()
        

if __name__ == '__main__':
    video_obj = Video()
    try:
        video_obj.run()
    finally:
        video_obj.release_resources()
