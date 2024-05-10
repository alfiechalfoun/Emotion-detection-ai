import cv2

class Video:
    def __init__(self):
        self.face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.vid = cv2.VideoCapture(0)
        if not self.vid.isOpened():
            print('Error, cannot find camera.')

    def detect_bounding_box(self, vid):
        gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray_image, 1.1, 4, minSize=(40, 40))
        for (x, y, w, h) in faces:
            cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return vid

    def run(self):
        while True:
            ret, frame = self.vid.read()
            if not ret:
                print('Error, failed to get the frame.')
                break

            frame_with_faces = self.detect_bounding_box(frame)
            cv2.imshow('Video', frame_with_faces)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def release_resources(self):
        self.vid.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    video_obj = Video()
    try:
        video_obj.run()
    finally:
        video_obj.release_resources()
