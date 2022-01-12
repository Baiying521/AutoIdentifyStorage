import os
import cv2 as cv

path = "./test/"  # 源文件路径
result_path = {'people': './result/people/result',  # 分类存储路径
               'others': './result/others/result'}


def body_detection(image):
    # 创建一个级联分类器 加载一个.xml分类器文件 它既可以是Haar特征也可以是LBP特征的分类器
    body_detecter = cv.CascadeClassifier(
        r'./haarcascades/haarcascade_fullbody.xml')
    # 多个尺度空间进行人脸检测   返回检测到的人脸区域坐标信息
    body = body_detecter.detectMultiScale(
        image=image, scaleFactor=1.1, minNeighbors=5)

    if body == ():
        print("未检测到身体")
        return 0
    else:
        print('检测身体信息信息如下:\n', body)
        return 1


def eye_detection(image):
    # 创建一个级联分类器 加载一个.xml分类器文件 它既可以是Haar特征也可以是LBP特征的分类器
    eye_detecter = cv.CascadeClassifier(
        r'./haarcascades/haarcascade_eye_tree_eyeglasses.xml')
    # 多个尺度空间进行人脸检测   返回检测到的人脸区域坐标信息
    eye = eye_detecter.detectMultiScale(
        image=image, scaleFactor=1.1, minNeighbors=5)

    if eye == ():
        return body_detection(image)
    else:
        print('检测身体信息信息如下:\n', eye)
        return 1


def face_detection(image):
    # 创建一个级联分类器 加载一个.xml分类器文件 它既可以是Haar特征也可以是LBP特征的分类器
    face_detecter = cv.CascadeClassifier(
        r'./haarcascades/haarcascade_frontalface_default.xml')
    # 多个尺度空间进行人脸检测   返回检测到的人脸区域坐标信息
    faces = face_detecter.detectMultiScale(
        image=image, scaleFactor=1.1, minNeighbors=5)

    if faces == ():
        print('未检测到人脸\n')
        return body_detection(image)
    else:
        print('检测人脸信息如下:\n', faces)
        return 1


def run():
    a = 0
    b = 0
    for file in os.listdir(path):
        image = cv.imread(path + file)
        src = cv.imread(path + file, 0)
        result = face_detection(src)  # 函数调用
        if result == 1:
            cv.imwrite(result_path['people']+str(a)+'.jpg', image)  # 存储图像
            a = a+1

        elif result == 0:
            cv.imwrite(result_path['others']+str(b)+'.jpg', image)  # 存储图像
            b = b+1

    # k = cv.waitKey(0) & 0xFF  # 等待键盘响应（64位计算机）
    # cv.destroyAllWindows()  # 销毁所有窗口


if __name__ == "__main__":
    run()
