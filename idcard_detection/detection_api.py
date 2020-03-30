import cv2
#import torn_detection.Detect as Detect
#detect_model_path = "./torn_detection/models/detect.tflite"
#classifier_model_path = "./torn_detection/models/classifier.tflite"
#image_path = "demo.jpg"
#image = cv2.imread(image_path)
#input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#detect = Detect.Detect(detect_model_path, classifier_model_path)
#def detect_image(image, score=0.5):
#    detect.detect(image)
#    return detect.draw_image(image, score)

idcard_detect_path = './idcard_detection/models/id.tflite'

from idcard_detection.tfLite import Detect
idcard_detect = Detect(idcard_detect_path)
def detect_imagev2(image, package_score=0.5, torn_score=0.5):
    idcard_detect.detect(image, package_score)
    image_show = image.copy()
    image_show = cv2.cvtColor(image_show, cv2.COLOR_BGR2RGB)
    have_target, image_show = idcard_detect.draw_image(image_show)
    if not have_target:
        return have_target, image_show
    image_h, image_w, _ = image.shape
    for rect in idcard_detect.get_detection_rect():
        xmin, ymin, xmax, ymax = idcard_detect.conver_to_abs_axis(rect, image_h, image_w)
        package_roi = image[ymin:ymax, xmin:xmax, :]
        package_roi = cv2.cvtColor(package_roi, cv2.COLOR_RGB2BGR)
        cv2.imwrite("id.jpg", package_roi)
        exit(0)
        package_roi = cv2.resize(package_roi, (300, 300))
        cv2.imshow("package_roi", package_roi)
        cv2.waitKey(0)
    return True, image_show



