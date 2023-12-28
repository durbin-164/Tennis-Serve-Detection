import numpy as np
import tensorflow as tf
import cv2
import time

from model import TransformerBlock

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     print("We got a GPU")
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# else:
#     print("Sorry, no GPU for you...")


from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n-pose.pt')

# lstm_model = tf.keras.models.load_model('models_v2/lstm_model_24_s2_v1_3.keras')
# lstm_model = tf.keras.models.load_model('models_v2/lstm_model_24_s2_v1_3.keras')
# lstm_model =  tf.keras.models.load_model('models_v2/lstm_model_24_s2_v1_6.keras', custom_objects={'<lambda>': lambda x: x}, compile=False, safe_mode=False)
lstm_model = tf.keras.models.load_model(
    'models_v2/lstm_model_24_s2_v1_15.keras',
    # custom_objects={'TransformerBlock': TransformerBlock},
    # compile=False,
    # safe_mode=False
)

# Open the video file
# video_path = "videos/youtube_2019.mp4"
# video_path = "videos/youtube_2015.mp4"
# video_path = "videos/youtube_2015__5min.mp4"
# video_path = "videos/youtube_2020_5min.mp4"
# video_path = "videos/youtube_2020_1min.mp4"
# video_path = "videos/youtube_2019_1min.mp4"
# video_path = "videos/youtube_2015_1min.mp4"
# video_path = "videos/video_input2.mp4"
video_path = "/home/masud.rana/Documents/Learning_Project/Important/Tennis/Referance videos/Copy of Junior Video Example.m4v"
cap = cv2.VideoCapture(video_path)

# Get the original video's frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# frame_width = 640
# frame_height = 640
frame_rate = int(cap.get(5))

print(f"Frame Rate: {frame_rate}")

# Define the codec and create a VideoWriter object
# output_video_path = "videos/outputs/output_youtube_2019.mp4"
# output_video_path = "videos/outputs/output_youtube_2015.mp4"
# output_video_path = "videos/outputs/youtube_2015__5min.mp4"
# output_video_path = "videos/outputs/youtube_2020_5min.mp4"
# output_video_path = "videos/outputs/youtube_2020_1min.mp4"
# output_video_path = "videos/outputs/youtube_2019_1min.mp4"
# output_video_path = "videos/outputs/youtube_2015_1min.mp4"
# output_video_path = "videos/outputs/output_video_input2.mp4"
output_video_path = "videos/outputs/Junior Video Example_v10.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

# frame_list = []
result_list = []
# new_result_list = []
new_frame_list = []
last_predict_result = None
lstm_bin_size = 24
serve_count = 0
false_frame_count = 0
frame_count = -1
serve_duplicate_count = 0
serve_last_frame = 0
INVERSE_RATION = 3


def get_person_index(boxes):
    index = 0  # Always high confidence age thake.
    return index


def choose_rows_randomly(arr, bin_size=lstm_bin_size):
    return arr.reshape(-1, bin_size, 51)


def get_result(results):
    all_results_nor = []
    for result in results:
        if result.boxes.shape[0] == 0:
            continue

        boxes = result.boxes.cpu().numpy()
        conf = result.keypoints.conf.cpu().numpy()
        xyn = result.keypoints.xyn.cpu().numpy()

        index = get_person_index(boxes)

        keypoints_nor = np.hstack((xyn[index].reshape(-1, 2), conf[index].reshape(-1, 1))).flatten()

        all_results_nor.append(keypoints_nor)

    # print(all_results_nor)
    all_results_nor = np.array(all_results_nor)

    all_results_nor = choose_rows_randomly(all_results_nor)

    return all_results_nor


def get_modified_frame(frame):
    # 720*1080
    resized_image = cv2.resize(frame, (1280, 720))
    ## (0,0) top left
    # h*w*c
    crop_frame = resized_image[200:-50, 250:-250, :]
    resized_image = cv2.resize(crop_frame, (640, 640))
    return resized_image


def add_text_to_frame(input_text, input_frame):
    # Create a copy of the input frame
    result_frame = input_frame.copy()

    # Define the text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    font_color = (0, 0, 255)  # Red color

    # Get the size of the text bounding box
    text_size, _ = cv2.getTextSize(input_text, font, font_scale, font_thickness)

    # Calculate the position to place the text
    text_position = (50, 50)

    # Draw a filled rectangle behind the text
    rectangle_position = (text_position[0], text_position[1] - text_size[1])
    rectangle_size = (text_size[0], text_size[1] + 10)  # Adjust the size as needed
    rectangle_color = (255, 255, 255)
    cv2.rectangle(result_frame, rectangle_position, (rectangle_position[0] + rectangle_size[0], rectangle_position[1] + rectangle_size[1]), rectangle_color, thickness=cv2.FILLED)

    # Draw the text on the result frame
    result_frame = cv2.putText(result_frame, input_text, text_position, font, font_scale, font_color, font_thickness)

    return result_frame


print("STARTING....")
start_time = time.time()
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    frame_count += 1

    if success:
        new_frame_list.append((frame, last_predict_result))
        if frame_count % INVERSE_RATION != 0:
            continue

        # modified_frame = get_modified_frame(frame)
        results = model.predict(frame, verbose=False)
        last_predict_result = results[0]

        if results[0].boxes.shape[0] == 0 or results[0].boxes.shape[0] > 3:
            false_frame_count += 1
            if false_frame_count > 15:
                false_frame_count = 0
                result_list = []
                # new_result_list =[]
            continue
        else:
            false_frame_count = 0

        result_list.append(results[0])
        # new_result_list.append(results[0])

        # Visualize the results on the frame
        # annotated_frame = results[0].plot()

        if len(result_list) >= lstm_bin_size:
            pose_result_norm = get_result(result_list)
            print(pose_result_norm.shape)
            lstm_model_result = lstm_model.predict(pose_result_norm)
            print(lstm_model_result)
            # is_serve = np.argmax(lstm_model_result[0])
            is_serve = lstm_model_result[0][1] >= 0.5
            print(is_serve)
            print(f"Frame progress: {frame_count}")
            print(f"Serve count: {serve_count}")
            is_increase_serve = False

            if is_serve:
                if serve_last_frame == 0 or (frame_count - serve_last_frame >= lstm_bin_size * 5):
                    serve_count += 1
                    serve_last_frame = frame_count
                    is_increase_serve = True

            len_new_frame = len(new_frame_list)
            for f_i, res in enumerate(new_frame_list):
                frame, result = res

                if result is None:
                    continue
                annotated_frame = result.plot()
                if is_increase_serve and (len_new_frame - f_i > lstm_bin_size * 3):
                    in_text = f"Serve Count: {serve_count - 1 }"
                else:
                    in_text = f"Serve Count: {serve_count}"

                # annotated_frame = cv2.putText(annotated_frame, in_text, (50, 50),
                #                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                #                               2)

                annotated_frame = add_text_to_frame(input_text=in_text, input_frame=annotated_frame)
                # Write the annotated frame to the output video
                out.write(annotated_frame)
                # cv2.imshow('serve predictor', annotated_frame)

            result_list = result_list[3:]  # Important.
            # new_result_list = []
            new_frame_list = []

            ###########################

        # Display the annotated frame (optional)
        # cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Introduce a delay to achieve 3 FPS
        # time.sleep(1 / 3)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break
print(f"Total time: {time.time() - start_time}")
# Release the video capture and writer objects
cap.release()
out.release()

# Close any open display windows
cv2.destroyAllWindows()
