import os
from enum import Enum

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')  # Load an official Pose model


class TaskEnum(str, Enum):
    SERVE = 'serve'
    NOT_SERVE = 'not_serve'


SEQUENCE_SIZE = 32
task = TaskEnum.NOT_SERVE
INVERSE_RATION = 3


def get_person_index(xywh):
    return 0


def choose_rows(arr, bin_size=SEQUENCE_SIZE):
    num_rows = arr.shape[0]
    end = -3
    start = num_rows + end - SEQUENCE_SIZE
    if start < 0:
        return np.array([])

    chosen_rows = arr[start:end, :]

    return chosen_rows.reshape(-1, bin_size, 51)


def choose_row_not_serve(arr, bin_size=SEQUENCE_SIZE):
    num_rows = arr.shape[0] - (200//INVERSE_RATION)
    if num_rows < bin_size:
        return np.array([])

    x = num_rows // bin_size
    total_row_choice = x * bin_size

    start_row = 200//INVERSE_RATION
    chosen_rows = arr[start_row: start_row + total_row_choice, :]

    return chosen_rows.reshape(-1, bin_size, 51)


def get_result(results):
    all_results_nor = []
    for result in results:
        if result.boxes.shape[0] == 0:
            continue

        xywh = result.boxes.xywh.cpu().numpy()
        index = get_person_index(xywh)
        conf = result.keypoints.conf.cpu().numpy()
        xyn = result.keypoints.xyn.cpu().numpy()

        keypoints_nor = np.hstack((xyn[index].reshape(-1, 2), conf[index].reshape(-1, 1))).flatten()

        all_results_nor.append(keypoints_nor)

    # print(all_results_nor)
    all_results_nor = np.array(all_results_nor)

    if task == TaskEnum.SERVE:
        all_results_nor = choose_rows(all_results_nor)
    else:
        all_results_nor = choose_row_not_serve(all_results_nor)

    # print(all_results_nor.shape)

    return all_results_nor


# get_result(results)


folder_path = f'/home/masud.rana/Documents/Learning_Project/Important/Tennis/Dataset_v4/V4_all_data/{task}'

# Get a list of all files in the folder
files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
         os.path.isfile(os.path.join(folder_path, file))]

print(len(files))
# Print the full paths of all files


all_pose_nor = None
save_all_pose_nor = f'data_numpy_v2/all_pose_nor_{task}_real_{SEQUENCE_SIZE}_v2.npy'


#
# def get_modified_frame(frame):
#     # 720*1080
#     resized_image = cv2.resize(frame, (1280, 720))
#     ## (0,0) top left
#     # h*w*c
#     crop_frame = resized_image[100:-50, 250:-250, :]
#     resized_image = cv2.resize(crop_frame, (640, 640))
#     return resized_image


for file in tqdm(files):
    try:
        cap = cv2.VideoCapture(file)
        results = []
        frame_count = -1
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                frame_count += 1
                if frame_count % INVERSE_RATION != 0:
                    continue

                # modified_frame = get_modified_frame(frame)
                result = model.predict(frame, verbose=False)
                results.append(result[0])

            else:
                break

        if len(results) < SEQUENCE_SIZE:
            continue

        all_res_nor = get_result(results)

        if all_pose_nor is not None and all_res_nor.shape[0] != 0:
            all_pose_nor = np.vstack((all_pose_nor, all_res_nor))

        elif all_pose_nor is None and all_res_nor.shape[0] != 0:
            all_pose_nor = all_res_nor

        # print(all_pose_nor.shape)

    except Exception as e:
        print(file)
        raise e

all_pose_nor = np.array(all_pose_nor)

print(all_pose_nor.shape)

np.save(save_all_pose_nor, all_pose_nor)

# data = np.load(save_all_pose)
data2 = np.load(save_all_pose_nor)
print(data2.shape)
