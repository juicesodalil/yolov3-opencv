import cv2
import time
from tqdm import tqdm
from component import process_frame

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# 获取三个尺度输出层
layersNames = net.getLayerNames()
output_layers_names = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

# 导入coco数据集80个类别
with open("coco.names", "r") as f:
    classes = f.read().splitlines()


def generate_video(input_path):
    file_head = input_path.split('/')[-1]
    output_path = "videos/out-" + file_head

    print('视频开始处理', input_path)

    # 获取视频总帧数
    cap = cv2.VideoCapture(input_path)
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        frame_count += 1
        if not success:
            break
    cap.release()
    print("视频总帧数为：", frame_count)

    cap = cv2.VideoCapture(input_path)
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, fourcc, fps, (int(frame_size[0]), int(frame_size[1])))

    # 进度条绑定视频总帧数
    pbar = tqdm(range(frame_count))
    for i in pbar:
        success, frame = cap.read()

        if not success:
            break
        try:
            frame = process_frame(net, frame, classes, output_layers_names)
        except Exception as e:
            print("error", e)
            pass

        out.write(frame)

        pbar.update(1)

    cv2.destroyAllWindows()
    out.release()
    cap.release()
    print("视频已保存", output_path)


generate_video("videos/test_video.mp4")
