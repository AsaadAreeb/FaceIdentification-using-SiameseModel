# kivy dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
# kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
# kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Other
import tensorflow as tf
import cv2 as cv
from layers import L1Distance
import os
import numpy as np

class CameraApp(App):

    # main block
    def build(self):
        # main layout components
        self.web_cam = Image(size_hint=(1, .8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1, .1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1, .1))

        # Layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.verification_label)
        layout.add_widget(self.button)

        # load model
        self.model = tf.keras.load_model('siamese_model.h5', custom_objects={'L1Distance': L1Distance})

        # video capture setup
        self.capture = cv.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout

    # get webcam feed continuously
    def update(self, *args):
        _, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]

        # Flip horizontal and convert image to texture
        buf = cv.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture
    
    # load images from files and convert to 100x100px
    def preprocess(self, file_path):
        """
        1. Read image as it is from file path.
        2. Load in the image.
        3. Preprocessing -
                        i. Resize image (100x100x3)
                        ii. Rescale the image to between (0-1)
        """
        # 1.
        byte_image = tf.io.read_file(file_path)
        
        # 2.
        image = tf.io.decode_jpeg(byte_image)
        
        # 3.
        image = tf.image.resize(image, (100, 100))
        image /= 255.0
        
        return image
    
    # function for verification of individual
    def verify(self, *args):
        # thresholds
        detection_threshold = 0.7
        verification_threshold = 0.6

        # path to input image from webcam
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        _, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv.imwrite(SAVE_PATH, frame)

        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_image = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_image = self.preprocess(os.path.join('application_data', 'verification_images', image))
            
            result = self.model.predict(list(np.expand_dims([input_image, validation_image], axis=1)))
            results.append(result)
            
        # Detection threshold: Metric above which prediction is considered positive
        detection = np.sum(np.array(results) > detection_threshold)
        # verification threshold: Proportion of positive predictions / total positive samples
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
        verified = verification > verification_threshold

        # set verification text
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'

        # log details
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)
        
        return results, verified

if __name__ == '__main__':
    CameraApp().run()