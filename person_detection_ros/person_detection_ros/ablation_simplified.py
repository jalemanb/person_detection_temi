#!/usr/bin/env python3
import os
import cv2
import numpy as np
import torch
from ament_index_python.packages import get_package_share_directory
from person_detection_ros.system.SOD import SOD

# --- GLOBAL FLAGS ---
use_depth = True
visualize = True

# --- Ablation configs ---
ABLATION_CONFIGS = {
    "no_memory": {"sim_thresh": 0.75, "beta": 0.1, "use_mb": True, "use_mem": False, "use_pseudo": False},
    "low_thresh_wo_pseudo": {"sim_thresh": 0.25, "beta": 0.1, "use_mb": True, "use_mem": True, "use_pseudo": False},
    "mid_thresh_wo_pseudo": {"sim_thresh": 0.5, "beta": 0.1, "use_mb": True, "use_mem": True, "use_pseudo": False},
    "high_thresh_wo_pseudo": {"sim_thresh": 0.75, "beta": 0.1, "use_mb": True, "use_mem": True, "use_pseudo": False},
    "low_thresh_w_pseudo": {"sim_thresh": 0.25, "beta": 0.1, "use_mb": True, "use_mem": True, "use_pseudo": True},
    "mid_thresh_w_pseudo": {"sim_thresh": 0.5, "beta": 0.1, "use_mb": True, "use_mem": True, "use_pseudo": True},
    "high_thresh_w_pseudo": {"sim_thresh": 0.75, "beta": 0.1, "use_mb": True, "use_mem": True, "use_pseudo": True},
}

# -------------------------------
# Utility functions
# -------------------------------
def compute_iou(boxA, boxB):
    xA1, yA1, xA2, yA2 = (
        (boxA[0], boxA[1], boxA[2], boxA[3])
        if (boxA[2] > boxA[0] and boxA[3] > boxA[1])
        else (boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3])
    )

    xB1, yB1, xB2, yB2 = (
        (boxB[0], boxB[1], boxB[2], boxB[3])
        if (boxB[2] > boxB[0] and boxB[3] > boxB[1])
        else (boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3])
    )

    inter_x1 = max(xA1, xB1)
    inter_y1 = max(yA1, yB1)
    inter_x2 = min(xA2, xB2)
    inter_y2 = min(yA2, yB2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    areaA = (xA2 - xA1) * (yA2 - yA1)
    areaB = (xB2 - xB1) * (yB2 - yB1)

    union = float(areaA + areaB - inter_area)
    return inter_area / union if union > 0 else 0.0


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pkg_shared_dir = get_package_share_directory("person_detection_ros")

    # Resolve paths
    yolo_model = None
    for m in ["yolo11n-pose.engine", "yolo11n-pose.pt"]:
        if os.path.exists(os.path.join(pkg_shared_dir, "models", m)):
            yolo_model = os.path.join(pkg_shared_dir, "models", m)
            break

    yolo_path = yolo_model
    bytetrack_path = os.path.join(pkg_shared_dir, "models", "bytetrack.yaml")
    feat_cfg = os.path.join(pkg_shared_dir, "models", "kpr_market_test_in.yaml")

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

    # --- Setup model
    model = SOD(
        yolo_model_path=yolo_path,
        feature_extracture_cfg_path=feat_cfg,
        tracker_system_path=bytetrack_path,
        use_experimental_tracker=True,
        use_mb=cfg["use_mb"],
        sim_thresh=cfg["sim_thresh"],
        beta=cfg["beta"],
        use_memory=cfg["use_mem"],
        use_pseudo=cfg["use_pseudo"],
        yolo_detection_thr=0.5,
        min_hits=1,
        max_age=1,
        iou_threshold=0.5,
        mb_threshold=6.0,
        kpr_kpt_conf=0.3,
        reid_count_thr=1,
        class_prediction_thr=0.8,
    )
    model.to(device)
    model.target_id = 1

    # Collect frames
    rgb_images = sorted(
        [f for f in os.listdir(rgb_dir) if f.lower().endswith((".png", ".jpg")) and f.startswith("frame_")],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )

    k = 8
    seg_size = max(1, len(rgb_images) // k)

    # -------------------
    # (1) Train ONCE on full dataset with GT logic
    # -------------------
    train_on_segment("all", model, rgb_images, rgb_dir, focal_dict, gt_boxes)

    # -------------------
    # (2) Evaluate segment by segment
    # -------------------
    acc_vector = np.zeros(k)
    acc_std_vector = np.zeros(k)

    for seg_id in range(k):
        seg_imgs = rgb_images[seg_id * seg_size : (seg_id + 1) * seg_size]
        acc, acc_std = evaluate_on_segment(model, seg_imgs, rgb_dir, focal_dict, gt_boxes)
        acc_vector[seg_id] = acc
        acc_std_vector[seg_id] = acc_std

    return acc_vector, acc_std_vector


# -------------------------------
# Training / Evaluation
# -------------------------------
def train_on_segment(seg_id, model, seg_imgs, rgb_dir, focal_dict, gt_boxes=None):
    for img_name in seg_imgs:
        frame_id = int(img_name.split("_")[1].split(".")[0])
        gt_box = gt_boxes.get(frame_id, (0, 0, 0, 0)) if gt_boxes else (0, 0, 0, 0)

        rgb_path = os.path.join(rgb_dir, img_name)
        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_img is None:
            continue
        h, w, _ = rgb_img.shape

        if use_depth:
            depth_path = os.path.join(rgb_dir, "depth", img_name[:-4] + ".npy")
            if not os.path.exists(depth_path):
                continue
            depth_img = np.load(depth_path)
            fx = fy = focal_dict[img_name]
            intr = [fx, fy, w / 2.0, h / 2.0]
        else:
            intr = [1, 1, 1, 1]
            depth_img = np.zeros((h, w), dtype=np.uint16)

        results = model.detect_and_track(rgb_img, depth_img, intr, detection_class=0)
        if results is None:
            continue

        detections_imgs, detection_kpts, bboxes, poses, track_ids, original_kpts = results

        # --- Assign target_id by IoU with GT
        gx, gy, gw, gh = gt_box
        if (gx, gy, gw, gh) != (0, 0, 0, 0):
            gt_rect = (gx, gy, gx + gw, gy + gh)
            best_iou, best_id = 0.0, None
            for i, box in enumerate(bboxes):
                x1, y1, x2, y2 = map(int, box)
                iou = compute_iou((x1, y1, x2, y2), gt_rect)
                if iou > best_iou:
                    best_iou, best_id = iou, track_ids[i]
            if best_iou > 0.5:
                model.target_id = best_id

        # If the target box is not found then Re-ID using the ground truth bbox and restart the tracker
        if model.target_id is not None and model.target_id not in track_ids:

            if (gx, gy, gw, gh) == (0, 0, 0, 0):
                continue
        else:
            # --- Update ReID
            model.updating_reid_ablation(detections_imgs, detection_kpts, track_ids)

def evaluate_on_segment(model, seg_imgs, rgb_dir, focal_dict, gt_boxes=None):
    preds = []
    for img_name in seg_imgs:
        frame_id = int(img_name.split("_")[1].split(".")[0])
        gt_box = gt_boxes.get(frame_id, (0, 0, 0, 0)) if gt_boxes else (0, 0, 0, 0)

        rgb_path = os.path.join(rgb_dir, img_name)
        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_img is None:
            continue
        h, w, _ = rgb_img.shape

        if use_depth:
            depth_path = os.path.join(rgb_dir, "depth", img_name[:-4] + ".npy")
            if not os.path.exists(depth_path):
                continue
            depth_img = np.load(depth_path)
            fx = fy = focal_dict[img_name]
            intr = [fx, fy, w / 2.0, h / 2.0]
        else:
            intr = [1, 1, 1, 1]
            depth_img = np.zeros((h, w), dtype=np.uint16)

        results = model.detect_and_track(rgb_img, depth_img, intr, detection_class=0)
        if results is None:
            continue

        detections_imgs, detection_kpts, bboxes, poses, track_ids, original_kpts = results

        gx, gy, gw, gh = gt_box
        if (gx, gy, gw, gh) != (0, 0, 0, 0):
            gt_rect = (gx, gy, gx + gw, gy + gh)
            best_iou, best_id = 0.0, None
            for i, box in enumerate(bboxes):
                x1, y1, x2, y2 = map(int, box)
                iou = compute_iou((x1, y1, x2, y2), gt_rect)
                if iou > best_iou:
                    best_iou, best_id = iou, track_ids[i]
            if best_iou > 0.5:
                model.target_id = best_id

        # --- If target_id not visible and GT exists
        if model.target_id is not None and model.target_id not in track_ids:
            if (gx, gy, gw, gh) == (0,0,0,0):
                continue
            else: 
                continue
        else:
            # --- Update ReID
            target_id_idx = np.where(track_ids == model.target_id)[0]

            reid_inference_result = model.reidentification_ablation(detections_imgs[target_id_idx], detection_kpts[target_id_idx])

            if reid_inference_result > 0.0:
                preds.append(reid_inference_result)

    return np.mean(preds) if preds else 0.0, np.std(preds) if preds else 0.0


# -------------------------------
# Ablation Runner
# -------------------------------
def run_ablation(datasets, ocl_dataset_path, crowd_dataset_path, robot_dataset_path):
    for name, cfg in ABLATION_CONFIGS.items():
        print(f"\n=== Running {name} ===")
        for d in datasets:
            acc_vec, acc_std_vec = evaluation(d, cfg, ocl_dataset_path, crowd_dataset_path, robot_dataset_path)
            save_results(d, name, acc_vec, acc_std_vec)


def save_results(dataset_name, ablation_name, acc_vector, acc_std_vector, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_name}_{ablation_name}_results.npz")
    np.savez(out_path, acc_vector=acc_vector, acc_std_vector=acc_std_vector)
    print(f"[INFO] Saved results for {dataset_name} ({ablation_name}) -> {out_path}")


# -------------------------------
# Main
# -------------------------------
def main():
    datasets = ["corridor2"]
    run_ablation(
        datasets,
        ocl_dataset_path="/media/enrique/Extreme SSD/ocl",
        crowd_dataset_path="/home/enrique/Videos/crowds",
        robot_dataset_path="/media/enrique/Extreme SSD/jtl-stereo-tracking-dataset/icvs2017_dataset/zed",
    )


if __name__ == "__main__":
    main()
