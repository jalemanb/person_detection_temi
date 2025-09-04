#!/usr/bin/env python3
import os
import cv2
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from ament_index_python.packages import get_package_share_directory
from person_detection_ros.system.SOD import SOD

# --- GLOBAL FLAGS ---
save_boxes = False
show_img = True
use_depth = True
visualize = True

# --- Ablation configs ---
ABLATION_CONFIGS = {
    # beta represents The probability of using Pseudo Negatives 

    # No Memory Mnaagement At All
    "no_memory": {"sim_thresh": 0.75, "beta": 0.1, "use_mb": True, "use_mem":False, "use_pseudo": False},

    # Proposed Memory Management with different Association values w/o Pseudo Negatives
    "low_thresh_wo_pseudo": {"sim_thresh": 0.25, "beta": 0.1, "use_mb": True, "use_mem":True, "use_pseudo":False},
    "mid_thresh_wo_pseudo": {"sim_thresh": 0.5, "beta": 0.1, "use_mb": True, "use_mem":True, "use_pseudo":False},
    "high_thresh_wo_pseudo": {"sim_thresh": 0.75, "beta": 0.1, "use_mb": True, "use_mem":True, "use_pseudo":False},

    # Proposed Memory Management with different Association values with Pseudo Negatives
    "low_thresh_w_pseudo": {"sim_thresh": 0.25, "beta": 0.1, "use_mb": True, "use_mem":True, "use_pseudo":True},
    "mid_thresh_w_pseudo": {"sim_thresh": 0.5, "beta": 0.1, "use_mb": True, "use_mem":True, "use_pseudo":True},
    "high_thresh_w_pseudo": {"sim_thresh": 0.75, "beta": 0.1, "use_mb": True, "use_mem":True, "use_pseudo":True},
}

def compute_center(box):
    """
    Compute the center (cx, cy) of a bounding box.
    Supports (x, y, w, h) or (x1, y1, x2, y2) formats.
    """
    if len(box) != 4:
        raise ValueError("Box must have 4 values")

    x1, y1, x2, y2 = None, None, None, None

    # format check: if box is (x,y,w,h)
    if box[2] >= 0 and box[3] >= 0 and (box[2] + box[0] > box[0]) and (box[3] + box[1] > box[1]):
        # treat as (x,y,w,h)
        x1, y1 = box[0], box[1]
        x2, y2 = box[0] + box[2], box[1] + box[3]
    else:
        # assume (x1,y1,x2,y2)
        x1, y1, x2, y2 = box

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return cx, cy

def compute_iou(boxA, boxB):
    # box format: (x, y, w, h) or (x1, y1, x2, y2)
    if len(boxA) == 4 and (boxA[2] > boxA[0] and boxA[3] > boxA[1]):  
        # assume x1,y1,x2,y2
        xA1, yA1, xA2, yA2 = boxA
    else:  
        # assume x,y,w,h
        xA1, yA1, wA, hA = boxA
        xA2, yA2 = xA1 + wA, yA1 + hA

    if len(boxB) == 4 and (boxB[2] > boxB[0] and boxB[3] > boxB[1]):
        xB1, yB1, xB2, yB2 = boxB
    else:
        xB1, yB1, wB, hB = boxB
        xB2, yB2 = xB1 + wB, yB1 + hB

    inter_x1 = max(xA1, xB1)
    inter_y1 = max(yA1, yB1)
    inter_x2 = min(xA2, xB2)
    inter_y2 = min(yA2, yB2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    areaA = (xA2 - xA1) * (yA2 - yA1)
    areaB = (xB2 - xB1) * (yB2 - yB1)

    union = float(areaA + areaB - inter_area)
    return inter_area / union if union > 0 else 0.0


# -------------------------------
# Utility: OCL average accuracy
# -------------------------------
def compute_average_accuracy(segment_accs):
    """
    Input: list of per-segment accuracies [ (seg_id, acc_vector), ... ]
           where acc_vector is accuracy over seen segments so far
    Returns: average accuracy curve of length K
    """
    k = len(segment_accs)
    mat = np.zeros((k, k))  # upper triangular acc matrix
    for i, accs in enumerate(segment_accs):
        mat[i, :len(accs)] = accs
    return mat.mean(axis=0)

# -------------------------------
# Read Bounding boxes
# ------------------------------
def load_gt_labels(rgb_dir):
    labels_path = os.path.join(rgb_dir, "labels.txt")
    gt_boxes = {}
    if not os.path.exists(labels_path):
        print(f"[WARN] No labels.txt found in {rgb_dir}")
        return gt_boxes
    with open(labels_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            frame_id, x, y, w, h = map(int, parts)
            gt_boxes[frame_id] = (x, y, w, h)
    return gt_boxes


# -------------------------------
# Core Evaluation
# -------------------------------
def evaluation(dataset, cfg, ocl_dataset_path, crowd_dataset_path, robot_dataset_path):
    print(f"\n▶ Evaluating {dataset} with config {cfg}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pkg_shared_dir = get_package_share_directory('person_detection_ros')

    # Resolve paths
    yolo_model = None
    for m in ["yolo11n-pose.engine","yolo11n-pose.pt"]:
        if os.path.exists(os.path.join(pkg_shared_dir, "models", m)):
            yolo_model = os.path.join(pkg_shared_dir, "models", m)
            break

    yolo_path = yolo_model
    bytetrack_path = os.path.join(pkg_shared_dir, 'models', 'bytetrack.yaml')
    feat_cfg = os.path.join(pkg_shared_dir, 'models', 'kpr_market_test_in.yaml')

    ocl_datasets = ["corridor1", "corridor2", "room", "lab_corridor", "ocl_demo", "ocl_demo2"]
    robot_datasets = ["corridor_corners", "hallway_2", "sidewalk", "walking_outdoor"]
    crowd_datasets = ["crowd1", "crowd2", "crowd3", "crowd4"]

    if dataset in ocl_datasets:
        rgb_dir = f"{ocl_dataset_path}/{dataset}/"
    elif dataset in robot_datasets:
        rgb_dir = f"{robot_dataset_path}/{dataset}/left/"
    elif dataset in crowd_datasets:
        rgb_dir = f"{crowd_dataset_path}/{dataset}/"
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    
    gt_boxes = load_gt_labels(rgb_dir)

    # Load focal lengths
    focal_dict = {}
    with open(os.path.join(rgb_dir, "focal_lengths.txt"), "r") as f:
        for line in f:
            name, focal = line.strip().split()
            focal_dict[name] = float(focal)

    # --- Setup model with ablation params
    model = SOD(
        yolo_model_path=yolo_path,
        feature_extracture_cfg_path=feat_cfg,
        tracker_system_path=bytetrack_path,
        use_experimental_tracker=True,
        use_mb=cfg["use_mb"],
        sim_thresh=cfg["sim_thresh"],
        beta=cfg["beta"],
        use_memory = cfg["use_mem"],
        use_pseudo = cfg["use_pseudo"],
        yolo_detection_thr=0.5,
        min_hits=1,
        max_age=1,
        iou_threshold=0.5,
        mb_threshold=6.0,
        kpr_kpt_conf=0.3,
        reid_count_thr=1,
        class_prediction_thr=0.8,
    )
    model.logger.disabled = True
    model.to(device)
    model.target_id = 1

    # Collect results
    rgb_images = sorted(
        [f for f in os.listdir(rgb_dir) if f.lower().endswith((".png", ".jpg")) and f.startswith("frame_")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )


    k = 8
    seg_size = max(1, len(rgb_images) // k)

    acc_matrix = np.zeros((k, k))
    acc_std_matrix = np.zeros((k, k))

    for seg_id in range(k):
        seg_imgs = rgb_images[seg_id*seg_size : (seg_id+1)*seg_size]

        train_on_segment(seg_id, model, seg_imgs, rgb_dir, focal_dict, gt_boxes)

        for eval_id in range(seg_id + 1):
            eval_imgs = rgb_images[eval_id*seg_size : (eval_id+1)*seg_size]
            acc, acc_std = evaluate_on_segment(seg_id, model, eval_imgs, rgb_dir, focal_dict, gt_boxes)
            acc_matrix[seg_id, eval_id] = acc
            acc_std_matrix[seg_id, eval_id] = acc_std


    return acc_matrix, acc_std_matrix

# -------------------------------
# INTERFACES TO FILL
# -------------------------------
def train_on_segment(seg_id, model, seg_imgs, rgb_dir, focal_dict, gt_boxes=None):
    """
    Train/update the model with all images in this segment and visualize detections.
    """
    for img_name in seg_imgs:
        print("SEG_ID", seg_id)
        frame_id = int(img_name.split("_")[1].split(".")[0])  # e.g. frame_0403.png -> 403
        gt_box = gt_boxes.get(frame_id, (0,0,0,0)) if gt_boxes else (0,0,0,0)

        rgb_path = os.path.join(rgb_dir, img_name)
        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_img is None:
            continue
        h, w, _ = rgb_img.shape

        # Depth handling
        if use_depth:
            depth_path = os.path.join(rgb_dir, "depth", img_name[:-4]+".npy")
            if not os.path.exists(depth_path):
                continue
            depth_img = np.load(depth_path)
            fx = fy = focal_dict[img_name]
            intr = [fx, fy, w/2.0, h/2.0]
        else:
            intr = [1,1,1,1]
            depth_img = np.zeros((h, w), dtype=np.uint16)

        # --- Run detection + tracking
        results = model.detect_and_track(rgb_img, depth_img, intr, detection_class=0)
        if results is None:
            continue

        detections_imgs, detection_kpts, bboxes, poses, track_ids, original_kpts = results
        
        # Gound Truth Box
        gx, gy, gw, gh = gt_box

        if (gx, gy, gw, gh) == (0,0,0,0):
            continue

        # --- Find closest detection to GT by IoU
        gt_cx = gx + 0.5 * gw
        gt_cy = gy + 0.5 * gh

        centers = np.empty((len(bboxes), 2), dtype=np.float32)
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            centers[i, 0] = 0.5 * (float(x1) + float(x2))
            centers[i, 1] = 0.5 * (float(y1) + float(y2))

        dists = np.sqrt((centers[:, 0] - gt_cx) ** 2 + (centers[:, 1] - gt_cy) ** 2)
        if dists.size == 0:
            continue

        pos_idx = int(np.argmin(dists))  # exactly one positive: closest center
        labels = np.zeros(len(bboxes), dtype=bool)
        labels[pos_idx] = True

        # (Optional) assign target_id to the closest detection
        if np.min(dists) <= 50:
            if track_ids is not None and len(track_ids) == len(bboxes):
                model.target_id = track_ids[pos_idx]

        # If the target box is not found then Re-ID using the ground truth bbox and restart the tracker
        if model.target_id is not None and model.target_id not in track_ids:
            continue
        else:
            # --- Update ReID
            model.updating_reid_ablation(detections_imgs, detection_kpts, track_ids)

        target_id = model.target_id

        if visualize:
            # --- Visualization
            vis_img = rgb_img.copy()

            # Draw GT bounding box (blue)
            if (gx, gy, gw, gh) != (0,0,0,0):
                cv2.rectangle(vis_img, (gx, gy), (gx+gw, gy+gh), (255, 0, 0), 2)
                cv2.putText(vis_img, "GT", (gx, max(0, gy-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

            # Draw detections
            if bboxes is not None and len(bboxes) > 0:
                for i, box in enumerate(bboxes):
                    x1, y1, x2, y2 = map(int, box)

                    if target_id is not None and track_ids[i] == target_id:
                        # Green overlay for target
                        overlay = vis_img.copy()
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                        cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)
                        cv2.putText(vis_img, f"Target ID: {track_ids[i]}",
                                    (x1, max(0, y1-10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    else:
                        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(vis_img, f"ID: {track_ids[i]}",
                                    (x1, max(0, y1-10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # Show
            cv2.imshow(f"Train Segment Visualization", vis_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break


def evaluate_on_segment(seg_id, model, seg_imgs, rgb_dir, focal_dict, gt_boxes=None):
    """
    Evaluate the model on all images of this segment.
    Return average classification accuracy (0..1).
    """
    per_frame_acc = []

    TP = TN = FP = FN = 0

    for img_name in seg_imgs:
        frame_id = int(img_name.split("_")[1].split(".")[0])  # parse frame number
        gt_box = gt_boxes.get(frame_id, (0,0,0,0)) if gt_boxes else (0,0,0,0)

        rgb_path = os.path.join(rgb_dir, img_name)
        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_img is None:
            continue
        h, w, _ = rgb_img.shape

        # Depth handling
        if use_depth:
            depth_path = os.path.join(rgb_dir, "depth", img_name[:-4]+".npy")
            if not os.path.exists(depth_path):
                continue
            depth_img = np.load(depth_path)
            fx = fy = focal_dict[img_name]
            intr = [fx, fy, w/2.0, h/2.0]
        else:
            intr = [1,1,1,1]
            depth_img = np.zeros((h, w), dtype=np.uint16)

        # --- Run detection + tracking
        results = model.detect_and_track(rgb_img, depth_img, intr, detection_class=0)
        if results is None:
            continue

        detections_imgs, detection_kpts, bboxes, poses, track_ids, original_kpts = results

        # --- Ground Truth Box
        gx, gy, gw, gh = gt_box

        if (gx, gy, gw, gh) == (0,0,0,0):
            continue


        # --- Build binary labels via closest center to GT center ---
        gt_cx = gx + 0.5 * gw
        gt_cy = gy + 0.5 * gh

        centers = np.empty((len(bboxes), 2), dtype=np.float32)
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            centers[i, 0] = 0.5 * (float(x1) + float(x2))
            centers[i, 1] = 0.5 * (float(y1) + float(y2))

        dists = np.sqrt((centers[:, 0] - gt_cx) ** 2 + (centers[:, 1] - gt_cy) ** 2)
        if dists.size == 0:
            continue

        pos_idx = int(np.argmin(dists))  # exactly one positive: closest center
        labels = np.zeros(len(bboxes), dtype=bool)
        labels[pos_idx] = True

        # (Optional) assign target_id to the closest detection
        if np.min(dists) <= 50:
            if track_ids is not None and len(track_ids) == len(bboxes):
                model.target_id = track_ids[pos_idx]

        # --- If target_id not visible and GT exists
        if model.target_id is not None and model.target_id not in track_ids:
            continue
        else:
            # --- Update ReID
            target_id_idx = np.where(track_ids == model.target_id)[0]

            logits = model.reidentification_ablation(detections_imgs, detection_kpts)

            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            print("FRAME_ID", frame_id, "SEG_ID", seg_id)
            print("LABELS", labels)
            print("PROBS", probs)
            preds_pos = probs >= 0.8  # boolean predictions

            # --- Tally confusion ---
            tp = int(np.logical_and(preds_pos,  labels).sum())
            tn = int(np.logical_and(~preds_pos, ~labels).sum())
            fp = int(np.logical_and(preds_pos,  ~labels).sum())
            fn = int(np.logical_and(~preds_pos,  labels).sum())

            TP += tp; TN += tn; FP += fp; FN += fn

            denom = tp + tn + fp + fn
            if denom > 0:
                per_frame_acc_val = (tp + tn) / denom
                per_frame_acc.append(per_frame_acc_val)
                print("per_frame_acc_val", per_frame_acc_val)
                print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
                print("denom", denom)


        target_id = model.target_id

        if visualize:

            # --- Visualization
            vis_img = rgb_img.copy()

            # Draw GT bounding box (blue)
            if (gx, gy, gw, gh) != (0,0,0,0):
                cv2.rectangle(vis_img, (gx, gy), (gx+gw, gy+gh), (255, 0, 0), 2)
                cv2.putText(vis_img, "GT", (gx, max(0, gy-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

            # Draw detections
            if bboxes is not None and len(bboxes) > 0:
                for i, box in enumerate(bboxes):
                    x1, y1, x2, y2 = map(int, box)
                    if target_id is not None and track_ids[i] == target_id:
                        # Green overlay for target
                        overlay = vis_img.copy()
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                        cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)
                        cv2.putText(vis_img, f"Target ID: {track_ids[i]}",
                                    (x1, max(0, y1-10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    else:
                        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(vis_img, f"ID: {track_ids[i]}",
                                    (x1, max(0, y1-10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # Show visualization
            cv2.imshow(f"Eval Segment Visualization", vis_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break


    total = TP + TN + FP + FN
    acc = float((TP + TN) / total) if total > 0 else 0.0
    std = float(np.std(per_frame_acc)) if per_frame_acc else 0.0
    return acc, std

# -------------------------------
# Ablation Runner
# -------------------------------
def run_ablation(datasets, ocl_dataset_path, crowd_dataset_path, robot_dataset_path):
    results = {}
    for d in datasets:
        
        for name, cfg in ABLATION_CONFIGS.items():
            print(f"\n=== Running {name} ===")
            all_curves = []

            acc_mtx, acc_std_mtx = evaluation(d, cfg, ocl_dataset_path, crowd_dataset_path, robot_dataset_path)
            save_results(d, name, acc_mtx, acc_std_mtx)

# def run_ablation(datasets, ocl_dataset_path, crowd_dataset_path, robot_dataset_path):
#     results = {}
#     for name, cfg in ABLATION_CONFIGS.items():
#         print(f"\n=== Running {name} ===")
#         all_curves = []

#         for d in datasets:
#             acc_mtx, acc_std_mtx = evaluation(d, cfg, ocl_dataset_path, crowd_dataset_path, robot_dataset_path)
#             save_results(d, name, acc_mtx, acc_std_mtx)

def save_results(dataset_name, ablation_name, acc_matrix, acc_std_matrix, out_dir="results"):
    """
    Save accuracy matrices to .npz file named after the dataset and ablation.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_name}_{ablation_name}_results.npz")
    np.savez(out_path, acc_matrix=acc_matrix, acc_std_matrix=acc_std_matrix)
    print(f"[INFO] Saved results for {dataset_name} ({ablation_name}) -> {out_path}")

# -------------------------------
# Main
# -------------------------------
def main():
    datasets = ["corridor2", "corridor1", "lab_corridor"]  # extend as needed
    run_ablation(
        datasets,
        ocl_dataset_path="/media/enrique/Extreme SSD/ocl",
        crowd_dataset_path="/home/enrique/Videos/crowds",
        robot_dataset_path="/media/enrique/Extreme SSD/jtl-stereo-tracking-dataset/icvs2017_dataset/zed"
    )

if __name__ == "__main__":
    main()