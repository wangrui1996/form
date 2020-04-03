import cv2
import numpy

def wrapper_image(image):
    image = image.astype(numpy.float32)# / 255.0
    return image

def gray_and_fliter(img, image_name='1.jpg', save_path='./'):  # 转为灰度图并滤波，后面两个参数调试用
    """
    将图片灰度化，并滤波
    :param img:  输入RGB图片
    :param image_name:  输入图片名称，测试时使用
    :param save_path:   滤波结果保存路径，测试时使用
    :return: 灰度化、滤波后图片
    """
    # img = cv2.imread(image_path + image_name)  # 读取图片
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图片
    # cv2.imwrite(os.path.join(save_path, image_name + '_gray.jpg'), img_gray)  # 保存,方便查看

    #img_blurred = cv2.filter2D(img_gray, -1,
    #    kernel=numpy.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], numpy.float32))  # 对图像进行滤波,是锐化操作
    #img_blurred = cv2.filter2D(img_blurred, -1, kernel=numpy.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], numpy.float32))
    # cv2.imwrite(os.path.join(save_path, img_name + '_blurred.jpg'), img_blurred)  # 锐化, 这里的卷积核可以更改
    img_blurred = img_gray
    return img_blurred

def gradient_and_binary_erode(img_blurred):
    """
        求取梯度，二值化
        :param img_blurred: 滤波后的图片
        :param image_name: 图片名，测试用
        :param save_path: 保存路径，测试用
        :return:  二值化后的图片
        """

    # 这里调整了kernel大小(减小),腐蚀膨胀次数后(增大),出错的概率大幅减小
    img_canny = cv2.Canny(img_blurred, 200, 200)
    # img_closed = cv2.dilate(img_closed, None, iterations=1)  # 腐蚀膨胀
    for _ in range(1):
        img_closed = cv2.dilate(img_canny, None, iterations=3)
        img_closed = cv2.erode(img_closed, None, iterations=5)
        # break
    return img_canny, img_closed

def gradient_and_binary(img_blurred, image_name='1.jpg', save_path='./'):  # 将灰度图二值化，后面两个参数调试用
    """
        求取梯度，二值化
        :param img_blurred: 滤波后的图片
        :param image_name: 图片名，测试用
        :param save_path: 保存路径，测试用
        :return:  二值化后的图片
        """
    # img_closed = cv2.erode(img_closed, None, iterations=1)
    # img_closed = cv2.dilate(img_closed, None, iterations=1)  # 腐蚀膨胀
    # 这里调整了kernel大小(减小),腐蚀膨胀次数后(增大),出错的概率大幅减小
    img_closed = cv2.Canny(img_blurred, 200, 200)
    img_closed = cv2.dilate(img_closed, None, iterations=1)
    return img_closed
