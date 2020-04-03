import os
import numpy
import cv2

from src.inference_ocrs import predict
from src.utils import gray_and_fliter, gradient_and_binary, gradient_and_binary_erode
from PIL import ImageFont, Image, ImageDraw

fontpath = os.path.join(os.path.dirname(__file__),"simsun.ttf")  # <== 这里是宋体路径
font = ImageFont.truetype(fontpath, 16)
idx_img = 0

class Progress:
    #def __int__(self, debug=False):
    def __init__(self, debug=False):
        self.debug = debug
    #    pass

    def set_debug(self, debug):
        self.debug = debug

    def main(self, image): # image is RGB
        img_blurred = gray_and_fliter(image)
        img_binary = gradient_and_binary(img_blurred)
        #cv2.imwrite("img_birary.jpg", img_binary)
        self.image = image
        if self.debug:
            #cv2.imshow("img_binary", img_binary)
            #cv2.imshow("rgb", image)
            self.show_image = image.copy()
        #cv2.imshow("demo", img_binary)
        #cv2.waitKey(0)
        self.img_binary = img_binary

        self.cut_image()


        if self.debug:
            pass
            #cv2.imshow("demo", self.show_image)
            #cv2.waitKey(0)

    def crop_img_upper(self, img_binary, threshold=50):
        binary_map = (255 - img_binary) / 255
        binary = numpy.ones(binary_map.shape) - binary_map
        binary_map = numpy.sum(binary, axis=0)
        stay_upper = False
        start = 0
        end = len(binary_map)
        for idx in range(len(binary_map)):
            if binary_map[idx] > threshold:
                print(binary_map[idx])
                start = idx
                break
        for idx in range(len(binary_map)-1, -1, -1):
            if binary_map[idx] > threshold:
                end = idx
                print(binary_map[idx])
                break
        start = max(0, start- 20)
        end = min(len(binary_map)-1, end+20)
        return start, end
    def progress_praragraph(self, image):
        img_blurred = gray_and_fliter(image)
        img_binary, img_erode = gradient_and_binary_erode(img_blurred)
        #cv2.imshow("demo", img_erode)
        start, end = self.crop_img_upper(img_erode)
        print(start, end)
        #cv2.imshow("source", img_binary)
        #cv2.imshow("rsult", img_binary[:, start:end])
        self.offset_width = start


        self.image = image[:, start:end, :]
        self.show_image = image.copy()
        self.img_binary = img_binary[:, start:end]
        #cv2.imshow("demo", img_binary)
        self.praragraph()

    def praragraph(self, threshold=2):
        binary_map = (255 - cv2.Canny(self.img_binary, 200, 220)) / 255
        binary = numpy.ones(binary_map.shape) - binary_map
        binary_map = numpy.sum(binary, axis=1)
        stay_upper = False
        words_result = {"words_result": []}
        words_result_num = 0
        for idx in range(len(binary_map)):
            if binary_map[idx] > threshold:
                if stay_upper:
                    continue
                else:
                    stay_upper = True
                    start = idx
            else:
                if stay_upper:
                    end = idx
                    stay_upper = False
                    if end - start < 5:
                        continue
                    else:
                        start = max(0, start - 3)
                        end = min(end + 3, len(binary_map))
                        ocr_text = predict(self.image[start:end, :])
                        word_result = {}
                        word_result.update({"location": {
                            "width": int(-1),
                            "top": int(start),
                            "left": int(self.offset_width),
                            "height": int(start-end)
                        }})
                        h,w,_= self.show_image.shape
                        cv2.line(self.show_image, (0, start), (w, start), (0,255,0))
                        cv2.line(self.show_image, (0, end), (w,end), (0,0,255))

                        img_pil = Image.fromarray(self.show_image)
                        draw = ImageDraw.Draw(img_pil)
                        draw.text((int(self.offset_width), int(start)), ocr_text, font=font, fill=(255, 0, 0, 0))
                        self.show_image = numpy.array(img_pil)
                        word_result.update({"words": ocr_text})
                        words_result["words_result"].append(word_result)
                        words_result_num = words_result_num + 1
                else:
                    continue
        self.words_result = words_result
        self.words_result_num = words_result_num
        #return words_result


    def cut_image(self):
        image_binary = self.img_binary
        _, contours, hierarchy = cv2.findContours(
            image_binary,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        image = self.image

        """
        img, contours, hierarchy =  cv2.findContours(输入图像, 层次类型, 逼近方法)
        参数：
            输入图像： 该方法会修改输入图像，建议传入输入图像的拷贝
            层次类型： 
                cv2.RETR_TREE 会得到图像中整体轮廓层次
                cv2.RETR_EXTERNAL 只得到最外面的轮廓
            逼近方法：

        返回值：
            img: 修改后的图像
            contours: 图像的轮廓
            hierarchy: 图像和轮廓的层次

        """
        # 原图像转换成bgr图像
        # color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 按照面积大小排序

        countours_res = []

        def point_judge(center, bbox):
            """
            用于将矩形框的边界按顺序排列
            :param center: 矩形中心的坐标[x, y]
            :param bbox: 矩形顶点坐标[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            :return: 矩形顶点坐标,依次是 左下, 右下, 左上, 右上
            """
            left = []
            right = []
            for i in range(4):
                if bbox[i][0] > center[0]:  # 只要是x坐标比中心点坐标大,一定是右边
                    right.append(bbox[i])
                else:
                    left.append(bbox[i])
            if len(left) != len(right):
                return None, None, None, None
            if right[0][1] > right[1][1]:  # 如果y点坐标大,则是右上
                right_down = right[1]
                right_up = right[0]
            else:
                right_down = right[0]
                right_up = right[1]

            if left[0][1] > left[1][1]:  # 如果y点坐标大,则是左上
                left_down = left[1]
                left_up = left[0]
            else:
                left_down = left[0]
                left_up = left[1]
            return left_down, right_down, left_up, right_up

        #print(len(contours))
        init = False
        bottom = -1

        words_result = {"words_result":[]}
        words_result_num = 0
        bboxes = []
        def nms(bbox, box):
            print(box)
            b_c = box[0]
            b_left_down, b_right_down, b_left_up, b_right_up = box[1:]
            b_width = b_left_down[0] - b_right_down[0]
            b_height = b_left_up[1] - b_left_down[1]
            for bx in bbox:

                bb_c = bx[0]
                bb_left_down, bb_right_down, bb_left_up, bb_right_up = bx[1:]
                bb_width = bb_left_down[0] - bb_right_down[0]
                bb_height = bb_left_up[1] - bb_left_down[1]
                if abs(bb_c[0] - b_c[0])+min(bb_width, b_width)/2 < max(bb_width, b_width)//2 \
                        and abs(bb_c[1] - b_c[1]) + min(bb_height, b_height)/2 < max(bb_height, b_height)//2:
                    return bbox
            bbox.append(box)
            return bbox
        def nmsv2(bbox, box):
            def roi(bbox1_left_down, bbox1_right_up, bbox2_left_down, bbox2_right_up):
                if bbox1_right_up[0] <= bbox2_left_down[0] or bbox1_left_down[0] >= bbox2_right_up[0] or \
                    bbox1_left_down[1] > bbox2_right_up[1] or bbox1_right_up[1] < bbox2_left_down[1]:
                    return 0
                xmin = max(bbox1_left_down[0], bbox2_left_down[0])
                ymin = max(bbox1_left_down[1], bbox2_left_down[1])
                xmax = min(bbox1_right_up[0], bbox1_right_up[0])
                ymax = min(bbox1_right_up[1], bbox1_right_up[1])

                inter_width = xmax - xmin
                inter_height = ymax - ymin
                inter_size = inter_width * inter_height
                bbox1_size = (bbox1_right_up[0] - bbox1_left_down[0]) * (bbox1_right_up[1] - bbox1_left_down[1])
                bbox2_size = (bbox2_right_up[0] - bbox2_left_down[0]) * (bbox2_right_up[1] - bbox2_left_down[1])

                return inter_size / min(bbox1_size, bbox2_size)


            b_c = box[0]
            b_left_down, _, _, b_right_up = box[1:]
            for bx in bbox:
                #bb_c = bx[0]
                bb_left_down, _, _, bb_right_up = bx[1:]
                if roi(bb_left_down, bb_right_up, b_left_down, b_right_up) > 0.99:
                    return bbox
            bbox.append(box)
            return bbox

        def nmsv3(bbox, box):
            box_contour = numpy.ones((4,1,2), dtype=numpy.int)

            b_c = box[0]
            b_left_down, b_right_down, b_left_up, b_right_up = box[1:]
            box_contour[0][0] = [int(b_left_down[0]), int(b_left_down[1])]
            box_contour[1][0] = [int(b_right_down[0]), int(b_right_down[1])]
            box_contour[2][0] = [int(b_right_up[0]), int(b_right_up[1])]
            box_contour[3][0] = [int(b_left_up[0]), int(b_left_up[1])]
            #print(box_contour.shape)
            for bx in bbox:
                bb_c = bx[0][0]
                bb_left_down, bb_right_down, bb_left_up, bb_right_up = bx[0][1:]
                #print(bb_left_down.shape)
                #print(type(bb_left_down))
                #print(tuple(bb_left_down))
                #print(bb_left_down)
                #print(bx[1])
                if cv2.pointPolygonTest(bx[1], (int(bb_left_down[0]), int(bb_left_down[1])), False) \
                        and cv2.pointPolygonTest(bx[1], (int(bb_left_up[0]), int(bb_left_up[1])), False) \
                        and cv2.pointPolygonTest(bx[1], (int(bb_right_down[0]), int(bb_right_down[1])), False) \
                        and cv2.pointPolygonTest(bx[1], (int(bb_right_up[0]), int(bb_right_up[1])), False):
                    return bbox
            bbox.append([box, box_contour])
            return bbox

        for i in range(0, len(contours)):
            area = cv2.contourArea(contours[i])  # 计算面积
            #print(contours[i])
            #print(type(contours[i]))
            #print(contours[i].shape)
            #print(contours[i].dtype)
            import numpy
            #contour = numpy.ones((4, 1, 2), dtype=numpy.int)
            #contour[0][0] = [0,0]
            #contour[1][0] = [10, 0]
            #contour[2][0] = [10,10]
            #contour[3][0] = [0, 10]
            #print(cv2.pointPolygonTest(contour, (9,9), False))
            #exit(0)
            rect = cv2.minAreaRect(contours[i])  # 最小外接矩,返回值有中心点坐标,矩形宽高,倾斜角度三个参数
            if (area <= 1 * image.shape[0] * image.shape[1]) and (area >= 0.05 * image.shape[0] * image.shape[1]) :
                # 人为设定,身份证正反面框的大小不会超过整张图片大小的0.4,不会小于0.05(这个参数随便设置的)
                countours_res.append(rect)
            else:
                if not init:
                    init = True
                    assert len(countours_res)>= 3
                    bottom = max(countours_res[-1][0][1], countours_res[-2][0][1])
                    for i in range(3):
                        _ = countours_res[len(countours_res) - i -1]
                        box = cv2.boxPoints(_)
                        left_down, right_down, left_up, right_up = point_judge([int(_[0][0]), int(_[0][1])], box)
                        bboxes = nmsv3(bboxes, [_[0], left_down, right_down, left_up, right_up])
                if rect[1][0] < 20 or rect[1][1] < 10 or rect[0][1]>bottom or min(rect[1][0], rect[1][1])>500:
                #if rect[1][0] < 10 or rect[1][1] < 10 or min(rect[1][0], rect[1][1]) > 500:
                #    print(rect)
                    continue
                box = cv2.boxPoints(rect)
                left_down, right_down, left_up, right_up = point_judge([int(rect[0][0]), int(rect[0][1])], box)
                if left_down is None:
                    continue
                src = numpy.float32([left_down, right_down, left_up, right_up])  # 这里注意必须对应

                dst = numpy.float32(
                    [[0, 0], [int(max(rect[1][0], rect[1][1])), 0], [0, int(min(rect[1][0], rect[1][1]))],
                     [int(max(rect[1][0], rect[1][1])),
                      int(min(rect[1][0], rect[1][1]))]])  # rect中的宽高不清楚是个怎么机制,但是对于身份证,肯定是宽大于高,因此加个判定
                m = cv2.getPerspectiveTransform(src, dst)  # 得到投影变换矩阵
                result = cv2.warpPerspective(image, m,
                                             (int(max(rect[1][0], rect[1][1])), int(min(rect[1][0], rect[1][1]))),
                                             flags=cv2.INTER_CUBIC)  # 投影变换
                len_bbox = len(bboxes)
                bboxes = nmsv3(bboxes, [rect[0], left_down, right_down, left_up, right_up])
                if len(bboxes) == len_bbox:
                    continue

                def crop_image(image, threshold=2):
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    binary = (255 - cv2.Canny(gray, 200, 220))/255
                    #print(binary)
                    #ret, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    #cv2.imshow("demo", binary)
                    #cv2.waitKey(0)
                    binary = numpy.ones(binary.shape) - binary
                    binary_map = numpy.sum(binary, axis=1)
                    start = 0
                    stay_upper = False
                    list_images = []
                    for idx in range(len(binary_map)):
                        if binary_map[idx] > threshold:
                            if stay_upper:
                                continue
                            else:
                                stay_upper = True
                                start = idx
                        else:
                            if stay_upper:
                                end = idx
                                stay_upper = False
                                if end - start < 5:
                                    continue
                                else:
                                    start = max(0, start - 3)
                                    end = min(end + 3, len(binary_map))
                                    crop_img = gray[start:end, :]
                                    list_images.append(crop_img)
                            else:
                                continue
                    #print("list length: ", len(list_images))
                    if stay_upper:
                        end = idx
                        stay_upper = False
                        if end - start < 5:
                            pass
                        else:
                            start = max(0, start - 3)
                            end = min(end + 3, len(binary_map))
                            crop_img = gray[start:end, :]
                            list_images.append(crop_img)
                    if len(list_images) ==0:
                        return gray

                    if len(list_images) == 1:
                        return list_images[0]

                    def rescale_img(img, scale_height=32):
                        height, width= img.shape
                        scale = height * 1.0 / scale_height
                        width = int(width / scale)
                        img = cv2.resize(img, (width, 32))
                        return img
                    re_img = rescale_img(list_images[0])
                    for image_ in list_images[1:]:
                        re_img = numpy.hstack((re_img, rescale_img(image_)))
                    return re_img
                import os
                #cv2.imshow("crop_before", result)
                crop_gray = crop_image(result)
                global idx_img
                cv2.imwrite(os.path.join("temp", "{}.jpg".format(idx_img)), crop_gray)
                idx_img = idx_img + 1

                ocr_text = predict(crop_gray)
                if "@" in ocr_text:
                    ocr_text = ocr_text+".com"

                word_result = {}
                word_result.update({"location": {
                    "width": int(right_up[0] - left_down[0]),
                    "top": int(left_down[1]),
                    "left": int(left_down[0]),
                    "height": int(right_up[1] - left_down[1])
                }})
                word_result.update({"words":ocr_text})
                words_result["words_result"].append(word_result)
                words_result_num += 1


                #print(ocr_text)
                #cv2.imshow("crop_result", crop_gray)
                #cv2.waitKey(0)
                #print(left_down)

                img_pil = Image.fromarray(self.show_image)
                draw = ImageDraw.Draw(img_pil)
                draw.text((int(left_down[0]), int(left_down[1])), ocr_text, font=font, fill=(255, 0, 0, 0))
                self.show_image = numpy.array(img_pil)
                #cv2.putText(image, ocr_text, (int(left_down[0]), int(left_down[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255) )
                #print(area, " ", rect)
                #cv2.imshow("crop",image )
                #cv2.waitKey(0)
            cv2.imwrite("result.jpg", self.show_image)
        #print(len(countours_res))
        #print("width: ", rect[1][0], "height: ", rect[1][1])
        #cv2.imshow("coun", self.show_image)
        #cv2.waitKey(0)
        self.words_result = words_result
        self.words_result_num = words_result_num


    def get_json(self):
        results = {}
        results.update({"words_result_num": self.words_result_num})
        results.update(self.words_result)
        return results
progress = Progress(True)

def form_recognition(image, type=0):
    if type == 0:  # 申请表
        progress.main(image)
        return progress.get_json()
    elif type == 1:
        progress.progress_praragraph(image)
        return progress.get_json()

def form_recognition_by_path(image_path, type=0):
    image = cv2.imread(image_path)
    if type == 0:  # 申请表
        progress.main(image)
        return progress.get_json()


def get_show_image():
    return progress.show_image
