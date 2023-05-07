import cv2
import numpy as np

CONFIDENCE_THRES = 0.2  # 指定置信度阈值，阈值越大，置信度过滤越强
NMS_THRES = 0.4  # 指定NMS阈值，越小，NMS越强


# 处理帧函数
def process_frame(net, img, classes, output_layers_names):
    # 获取图像宽高
    height, width, _ = img.shape
    # pre-process
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    # 前向推断
    prediction = net.forward(output_layers_names)

    # 存放预测框坐标
    boxes = []
    # 存放置信度
    objectness = []
    # 存放类别概率
    class_probs = []
    # 存放预测框类别索引号
    class_ids = []
    # 存放预测框类别名称
    class_names = []

    for scale in prediction:  # 遍历三种尺度
        for bbox in scale:  # 遍历每个预测框
            # 获取该预测框的confidence
            obj = bbox[4]
            # 获取该预测框在COCO数据集80类别的概率
            class_scores = bbox[5:]
            # 获取概率最高概率类别的名称
            class_idx = np.argmax(class_scores)
            class_name = classes[class_idx]
            # 获取最高类的概率数
            class_prob = class_scores[class_idx]

            # 获取预测框中心点坐标、预测框宽高
            center_x = int(bbox[0] * width)
            center_y = int(bbox[1] * height)
            w = int(bbox[2] * width)
            h = int(bbox[3] * height)
            # 计算预测框左上角坐标
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # 将每个预测框的结果存放至列表中
            boxes.append([x, y, w, h])
            objectness.append(float(obj))
            class_ids.append(class_idx)
            class_names.append(class_name)
            class_probs.append(class_prob)

    # 将预测框置信度objectness与各类别置信度class_pred相乘，获得最终该预测框置信度confidence
    confidences = np.array(class_probs) * np.array(objectness)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRES, NMS_THRES)
    # print(indexes.flatten())
    # colors = [[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255],
    #           [255, 0, 255], [192, 192, 192], [128, 128, 128], [128, 0, 0], [128, 128, 0], [0, 128, 0], [128, 0, 128],
    #           [0, 128, 128], [0, 0, 128]]
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    # 遍历留下的每个框，可视化
    for i in indexes:
        # 获取坐标
        x, y, w, h = boxes[i]
        # 获取置信度
        confidence = str(round(confidences[i], 2))
        # 获取颜色，画框
        color = colors[i % len(colors)]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 8)

        # 写类别名称和置信度
        # 图片添加文字，左上角坐标，字体，字体大小，颜色，字体粗细
        string = '{} {}'.format(class_names[i], confidence)
        cv2.putText(img, string, (x, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

    return img
