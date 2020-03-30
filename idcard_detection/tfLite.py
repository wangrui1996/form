import cv2
import numpy as np
import tensorflow as tf
import os

def get_Absolute_coordinates(Relative_coordinates, image_height, image_width):
    # return xmin, ymin, xmax, ymax
    return int(Relative_coordinates[0] * image_width), \
           int(Relative_coordinates[1] * image_height), \
           int(Relative_coordinates[2] * image_width), \
           int(Relative_coordinates[3] * image_height)

class Detect:
    def __init__(self, model_path):
        self.image_idx = 1
        # Load TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        input_dtype = self.input_details[0]['dtype']
        self.input_data = np.array(np.random.random_sample(self.input_shape), dtype=input_dtype)

    def detect(self, image, score=0.5):
        if image.shape[0] != self.input_shape[1] or image.shape[1] != self.input_shape[2]:
            image = cv2.resize(image, (self.input_shape[2], self.input_shape[1]))
        self.input_data[0, :, :, :] = image
        self.input_data = self.input_data - 127
        self.input_data = self.input_data / 127
        self.interpreter.set_tensor(self.input_details[0]['index'], self.input_data)
        self.interpreter.invoke()
        self.targets_positions = self.interpreter.get_tensor(self.output_details[0]['index'])
        self.targets_label = self.interpreter.get_tensor(self.output_details[1]['index'])
        self.targets_score = self.interpreter.get_tensor(self.output_details[2]['index'])
        self.targets_number = self.interpreter.get_tensor(self.output_details[3]['index'])
        self.detect_rect_list = []
        for idx in range(int(self.targets_number)):
            if self.targets_score[0][idx] <= score:
                print(self.targets_score[0][idx])
                break
            rp = self.targets_positions[0][idx]
            self.detect_rect_list.append((max(0, rp[1]),
                                     max(0, rp[0]),
                                     min(1.0, rp[3]),
                                     min(1.0, rp[2])))

    def draw_image(self, image, draw_rect=False):
        img_h, img_w,_ = image.shape
        have_target = False
        for detect_rect in self.detect_rect_list:
            have_target = True
            xmin, ymin, xmax, ymax = get_Absolute_coordinates(detect_rect, img_h, img_w)
            if draw_rect:
                image = cv2.rectangle(image, (xmin, ymin), (xmax , ymax), (0, 255, 255), 1)
            else:
                image[ymin:ymax, xmin:xmax, :] = image[ymin:ymax, xmin:xmax, :] * 0.5
#            region = image[ymin:ymax, xmin:xmax, :].copy()
#            region = cv2.cvtColor(region, cv2.COLOR_RGB2BGR)
#            self.image_idx = self.image_idx + 1
#            print(self.targets_label[0][idx])
        return have_target, image

    def get_detection_rect(self):
        return self.detect_rect_list

    def conver_to_abs_axis(self, rect, image_height, image_width):
        return get_Absolute_coordinates(rect, image_height, image_width)


