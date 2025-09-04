#!/usr/bin/env python3
import os
import cv2
import numpy as np
import torch
from ament_index_python.packages import get_package_share_directory
from person_detection_ros.system.SOD import SOD
import time


show_img = True
use_depth = True


def extract_frame_index(name: str) -> int:
    base = os.path.splitext(name)[0]
    digits = ''.join(ch for ch in base if ch.isdigit())
    return int(digits) if digits else -1

def write_bboxes_lines(fh, frame_id: int, bboxes, track_ids, target_id):
    # normalize track_ids to list of ints
    try:
        track_ids_list = [int(x) for x in np.array(track_ids).reshape(-1).tolist()]
    except Exception:
        track_ids_list = [int(x) for x in track_ids]
    for j, box in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, box)
        tid = track_ids_list[j] if j < len(track_ids_list) else -1
        is_target = 1 if (target_id is not None and tid == int(target_id)) else 0
        fh.write(f"{frame_id},{x1},{y1},{x2},{y2},{is_target}\n")

def evaluation(dataset):

    print("TIME TO EVALUATE")


    bboxes_txt = os.path.join(dataset, "bboxes.txt")
    bbox_f = open(bboxes_txt, "w")
    bbox_f.write("# frame_id,x1,y1,x2,y2,is_target\n")

    ########################################################################################################
    # Initialize Person Detection Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model paths
    pkg_shared_dir = get_package_share_directory('person_detection_ros')

    yolo_models = ['yolo11n-pose.engine','yolo11n-pose.pt']
    yolo_model = None

    for model_file in yolo_models:
        file_path = os.path.join(pkg_shared_dir, "models", model_file)
        if os.path.exists(file_path):
            yolo_model = file_path
            break

    yolo_path = os.path.join(pkg_shared_dir, 'models', yolo_model)
    bytetrack_path = os.path.join(pkg_shared_dir, 'models', 'bytetrack.yaml')
    feature_extracture_cfg_path = os.path.join(pkg_shared_dir, 'models', 'kpr_market_test_in.yaml')



    if use_depth:
        print("USE SORT+DEPTH")
        model = SOD( # SORT+DEPTH
            yolo_model_path = yolo_path, 
            feature_extracture_cfg_path = feature_extracture_cfg_path, 
            tracker_system_path = bytetrack_path,
            yolo_detection_thr = 0.5,
            use_experimental_tracker = True,
            use_mb=True,
            max_age = 2,
            min_hits = 3, 
            iou_threshold = 0.2, 
            mb_threshold = 6.0, 
        )

    model.to(device)

    model.target_id = 1
    
    model.logger.disabled = True

    # Initialize the template
    ########################################################################################################

    rgb_dir = os.path.join(dataset, "rgb")

    # Get all RGB images sorted by numeric order
    rgb_images = sorted(
        [f for f in os.listdir(rgb_dir) if (f.startswith('RGB') or f.startswith('rgb') or f.startswith('left')) or f.startswith('RGB') and (f.endswith('.png') or f.endswith('.jpg'))],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )

    bboxes = []
    times = []

    reid_counter = 0

    # Iterate through each RGB image
    # for i, rgb_img_name in enumerate(rgb_images[3*(len(rgb_images)//8):]):
    for i, rgb_img_name in enumerate(rgb_images):

        # if i == len(rgb_images) //2:
        #     break

        rgb_img_path = os.path.join(rgb_dir, rgb_img_name)

        rgb_img = cv2.imread(rgb_img_path, cv2.IMREAD_COLOR)  # Read RGB as BGR8

        height, width, _ = rgb_img.shape  # Get dimensions

        # Generate a random depth image (grayscale)
        if use_depth:
            depth_path = os.path.join(dataset, "depth", "DEPTH"+rgb_img_name[3:-3]+"npy")
            depth_img = np.load(depth_path)
            camera_intrinsics = [620.84722900, 621.05346680, 325.16311646, 237.45947266]

            # print("DEPTH DATA")
            # print("depth shape", depth_img.shape)
            # print("rgb shape", rgb_img.shape)
            # print("Max value", np.max(depth_img))
            # print("CAMERA INTRINSICS", camera_intrinsics)
            # exit()
        else:
            camera_intrinsics=[1.0, 1.0, 1.0, 1.0]
            depth_img = np.random.randint(0, 256, (height, width), dtype=np.uint8)


        # --- Run detection + tracking
        results = model.detect_and_track(rgb_img, depth_img, camera_intrinsics, detection_class=0)

        if results is None:
            cv2.imshow("Detection", rgb_img)
            cv2.waitKey(67)
            continue

        detections_imgs, detection_kpts, bboxes, poses, track_ids, original_kpts = results

        doing_reid = False

        # If the target box is not found then Re-ID using the ground truth bbox and restart the tracker
        if model.target_id is not None and model.target_id not in track_ids:

            reid_results = model.reidentification_ablation(detections_imgs, detection_kpts, tracked_ids = track_ids, return_mask = True)

            if reid_results is not None:
            
                logits, attentions = reid_results

                probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()

                print("PROBS", probs)

                if np.max(probs) > 0.8:

                    reid_counter += 1

                    if reid_counter > 3:
                        pos_idx = np.argmax(probs)

                        model.target_id = track_ids[pos_idx]
                        reid_counter = 0
                else:
                    reid_counter = 0

                attentions_path = os.path.join(dataset, "attentions", "ATT"+rgb_img_name[3:-3]+"npz")
                doing_reid = True

                np.savez_compressed(
                    attentions_path,
                    attentions=attentions,
                    probs = probs,
                    doing_reid=np.array(doing_reid, dtype=np.bool_)
                )
        else:
            # --- Update ReID
            probs, attentions = model.updating_reid_ablation(detections_imgs, detection_kpts, track_ids, return_mask = True)

            attentions_path = os.path.join(dataset, "attentions", "ATT"+rgb_img_name[3:-3]+"npz")

            np.savez_compressed(
                attentions_path,
                attentions=attentions,
                probs = probs,
                doing_reid=np.array(doing_reid, dtype=np.bool_)
            )

        # print(f"Execution Time: {execution_time_s:.3f} s")

        frame_id = extract_frame_index(rgb_img_name)

        write_bboxes_lines(bbox_f, frame_id, bboxes, track_ids, model.target_id)

        for j in range(len(bboxes)):

            x1, y1, x2, y2 = map(int, bboxes[j])

            if show_img:
                cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red Box
                cv2.putText(rgb_img, f"ID: {int(track_ids[j])}", (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
        if model.target_id in track_ids:

            idx = np.where(track_ids == model.target_id)[0][0]

            id = track_ids[idx]
            
            x1, y1, x2, y2 = map(int, bboxes[idx])

            
            if show_img:
                cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green Box - Target Person


        # Show the image with bounding boxes
        if show_img:
            cv2.imshow("Detection", rgb_img)
            # cv2.waitKey(0)  # Add small delay to update the window
            cv2.waitKey(67)
            # cv2.waitKey(3)

    bbox_f.close()
    print(f"[INFO] Wrote detections to: {bboxes_txt}")


def main():

    evaluation(dataset ="/home/enrique/rosbags/michelbag_imgs")

if __name__ == '__main__':
    main()