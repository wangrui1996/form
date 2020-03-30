import cv2
import numpy as np
import tensorflow as tf


class classfier:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        # Test the TensorFlow Lite model on random input data.
        input_shape = self.input_details[0]['shape']
        self.input_width = input_shape[-2]
        self.input_height = input_shape[-3]
        self.input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

    def progress(self, image):
        image = cv2.resize(image, (self.input_width, self.input_height))
        self.input_data[0] = image
        self.input_data = self.input_data / 127.5 - 1
        self.interpreter.set_tensor(self.input_details[0]['index'], self.input_data)

        self.interpreter.invoke()
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        tflite_results = self.interpreter.get_tensor(self.output_details[0]['index'])
        max_score = -1
        max_idx = -1
        for idx, score in enumerate(tflite_results[0]):
            print(idx, score)
            if max_score < score:
                max_score = score
                max_idx = idx
        return max_idx

class Detect:
    def __init__(self, detect_model_path, classfier_model_path):
        # Load TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(detect_model_path)
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        input_dtype = self.input_details[0]['dtype']
        self.input_data = np.array(np.random.random_sample(self.input_shape), dtype=input_dtype)
        self.classfier = classfier(classfier_model_path)

    def detect(self, image):
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

    def draw_image(self, image, score=0.2):
        img_h, img_w,_ = image.shape
        have_target = False
        for idx in range(int(self.targets_number)):
            if self.targets_score[0][idx] <= score:
                print(self.targets_score[0][idx])
                break
            def get_Absolute_coordinates(Relative_coordinates, image_height, image_width):
                return max(0, int(Relative_coordinates[1] * image_width)), \
                max(0, int(Relative_coordinates[0] * image_height)), \
                min(image_width, int(Relative_coordinates[3] * image_width)), \
                min(image_height, int(Relative_coordinates[2] * image_height))
            xmin, ymin, xmax, ymax = get_Absolute_coordinates(self.targets_positions[0][idx], img_h, img_w)
            region = image[ymin:ymax, xmin:xmax, :].copy()
            region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            if self.classfier.progress(region) == 1:
                cv2.putText(image, 'Have', (xmin, ymax), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 255, 255), 12)
                have_target = True
            else:
                cv2.putText(image, 'No', (xmin, ymax), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 255, 255), 12)
            image = cv2.rectangle(image, (xmin, ymin), (xmax , ymax), (0, 0, 255), int(min(img_h,img_w) / 100))
#            print(self.targets_label[0][idx])
        return have_target, image



