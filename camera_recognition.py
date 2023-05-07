import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from component import process_frame



def look_img(img):
    """opencv读入图像为BGR，matplot可视为格式为RGB"""
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.axis('off')
    plt.imshow(img_RGB)
    plt.show()


net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# 获取三个尺度输出层
layersNames = net.getLayerNames()
output_layers_names = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

# 导入coco数据集80个类别
with open("coco.names", "r") as f:
    classes = f.read().splitlines()






if __name__ == '__main__':
    # img = cv2.imread("images/test_img4.jpg")
    # img = process_frame(img)
    # look_img(img)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.open(0)

    while cap.isOpened():
        # 获取画面
        success, frame = cap.read()
        if not success:
            print('Error')
            break
        start_time = time.time()

        # 处理帧函数
        frame = process_frame(net, frame,classes, output_layers_names)

        # 展示处理后的三通道图像
        cv2.imshow("my_window", frame)

        if cv2.waitKey(1) in [ord('q'), 27]:  # q或者esc退出
            break

    cap.release()
    cv2.destroyAllWindows()
