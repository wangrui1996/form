#-*- coding:utf-8 -*-
import os
import ocr
import time
import shutil
import numpy as np
from PIL import Image
from glob import glob
image_files = glob('./data/*.*')
import cv2

if __name__ == '__main__':
    result_dir = './test_result'
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)

    for image_file in sorted(image_files):
        #image = np.array(Image.open(image_file).convert('RGB'))
        image = cv2.imread(image_file)
        cv2.imshow("demo1", cv2.resize(image, (300,300)))

        t = time.time()
        result, image_framed = ocr.model(image, True)
        output_file = os.path.join(result_dir, image_file.split('/')[-1])
        cv2.imshow("demo2", cv2.resize(image_framed,(300,300)))
        Image.fromarray(image_framed).save(output_file)
        print("Mission complete, it took {:.3f}s".format(time.time() - t))
        print("\nRecognition Result:\n")
        for key in result:
            print(result[key][1])
        cv2.waitKey(0)

