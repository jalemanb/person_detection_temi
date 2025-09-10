# person_detection_ros/pose_estimation_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import Pose, TransformStamped, PoseArray, PoseStamped
from person_detection_msgs.msg import Candidate, CandidateArray
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch 
from person_detection_ros.system.SOD import SOD
from person_detection_ros.system.img_msg_tools import compressed_imgmsg_to_cv2
import os
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation as R
import time
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data
from std_srvs.srv import Trigger

class HumanPoseEstimationNode(Node):
    def __init__(self):
        super().__init__('person_detection_node')

        sensor_fast_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # no retries (lower latency)
            history=HistoryPolicy.KEEP_LAST,
            depth=1,                                    # only the newest
            durability=DurabilityPolicy.VOLATILE        # do not latch
        )

        # Create publishers
        self.human_image_detection_pub = self.create_publisher(
            CandidateArray, 
            'image_detections', 
            1
        )
        self.human_cartesian_detection_pub = self.create_publisher(
            PoseArray, 
            'cartesian_detections_local', 
            1
        )
        self.target_human_cartesian_detection_pub = self.create_publisher(
            PoseArray, 
            'target_cartesian_detection_local', 
            1
        )
        self.publisher_debug_detection_image_compressed = self.create_publisher(
            CompressedImage, 
            'human_detection/img_compressed', 
            1
        )
        self.publisher_debug_detection_image = self.create_publisher(
            Image, 
            'human_detection/img_raw', 
            1
        )

        self.set_target_id_srv = self.create_service(Trigger, 'set_target_person', self.set_target_id_callback)
        self.unset_target_id_srv = self.create_service(Trigger, 'unset_target_person', self.unset_target_id_callback)

        self.temp_counter_ = 0
 
        # Create a TransformBroadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Bridge to convert ROS messages to OpenCV 
        self.cv_bridge = CvBridge()

        # Single Person Detection model
        # Setting up Available CUDA device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Setting up model paths (YOLO for object detection and segmentation, and orientation estimation model)
        pkg_shared_dir = get_package_share_directory('person_detection_ros')

        yolo_models = ['yolo11n-pose.engine','yolo11n-pose.pt']
        yolo_model = None

        for model_file in yolo_models:
            file_path = os.path.join(pkg_shared_dir, "models", model_file)
            if os.path.exists(file_path):
                yolo_model = file_path
                break

        yolo_path = os.path.join(pkg_shared_dir, 'models', yolo_model)
        # feature_extracture_cfg_path = os.path.join(pkg_shared_dir, 'models', 'kpr_market_test_in.yaml') kpr_market_test_solider
        feature_extracture_cfg_path = os.path.join(pkg_shared_dir, 'models', 'kpr_market_test_in.yaml')

        bytetrack_path = os.path.join(pkg_shared_dir, 'models', 'bytetrack.yaml')

        # Setting up Detection Pipeline
        self.model = SOD(
            yolo_model_path = yolo_path, 
            feature_extracture_cfg_path = feature_extracture_cfg_path, 
            tracker_system_path=bytetrack_path,
            yolo_detection_thr = 0.3,
            use_experimental_tracker = True,
            use_mb=True,
            max_age = 2,
            min_hits = 3, 
            iou_threshold = 0.2, 
            mb_threshold = 6.0, 
        )

        self.model.to(device)
        self.get_logger().info('Deep Learning Model Armed')

        # Warmup inference (GPU can be slow in the first inference)
        self.model.detect(
            img_rgb = np.ones((480, 640, 3), dtype=np.uint8), 
            img_depth = np.ones((480, 640), dtype=np.uint16))
        self.get_logger().info('Warmup Inference Executed')
        
        # Frame ID from where the human is being detected
        self.frame_id = None
        self.header = None
        self.target_frame = 'temi/base_link'

        # Quaternion for 90 degrees rotation around the x-axis
        rot_x = R.from_euler('x', 90, degrees=True)

        # Quaternion for -90 degrees rotation around the z-axis
        rot_z = R.from_euler('z', -90, degrees=True)

        # Combine the rotations
        self.combined_rotation = rot_x * rot_z
        
        # Subscribers using message_filters
        self.depth_sub = Subscriber(
            self, 
            CompressedImage, 
            '/temi/camera/aligned_depth_to_color/image_raw/compressed',
            qos_profile=sensor_fast_qos)
        self.rgb_sub = Subscriber(
            self, 
            CompressedImage, 
            '/temi/camera/color/image_raw/compressed',
            qos_profile=sensor_fast_qos)
        self.info_sub = Subscriber(
            self, 
            CameraInfo, 
            '/temi/camera/color/camera_info',
            qos_profile=sensor_fast_qos)

        # ApproximateTimeSynchronizer allows small timestamp mismatch
        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.info_sub],
            queue_size=1,
            slop=0.1  # seconds
        )
        self.ts.registerCallback(self.image_callback)

    def set_target_id_callback(self, request, response):
        self.get_logger().info('Setting target Person received!')

        self.model.set_target_id()

        response.success = True
        response.message = 'Action completed successfully.'
        return response

    def unset_target_id_callback(self, request, response):
        self.get_logger().info('Unsetting Target Person!')

        self.model.unset_target_id()
        response.success = True
        response.message = 'Action completed successfully.'
        return response

    def image_callback(self, rgb_msg, depth_msg, info_msg):

        self.frame_id = rgb_msg.header.frame_id
        self.header = rgb_msg.header

        # Convert ROS Image messages to OpenCV images
        depth_image = compressed_imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')

        rgb_image = self.cv_bridge.compressed_imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')

        # Check if RGB and depth images are the same size
        if depth_image.shape != rgb_image.shape[:2]:
            self.get_logger().warning('Depth and RGB images are not the same size. Skipping this pair.')
            return

        fx = info_msg.k[0]
        fy = info_msg.k[4]
        cx = info_msg.k[2]
        cy = info_msg.k[5]

        # Process the images and estimate pose
        self.process_images(rgb_image, depth_image, [fx, fy, cx, cy])

    def process_images(self, rgb_image, depth_image, camera_params):
        # Your pose estimation logic here
        # For demonstration, let's assume we get the pose from some function
        # if self.temp_counter_  == 5:
        #     self.model.set_target_id()
        self.temp_counter_ +=1

        start_time = time.time()
        ############################
        results = self.model.detect(rgb_image, depth_image, camera_params)
        ############################
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000
            
        # self.get_logger().info(f"Model Inference Time: {execution_time} ms")

        person_poses = []
        bbox = []
        kpts = []
        tracked_ids = []
        target_id = None

        if results is not None:

            person_poses, bbox, _, tracked_ids, kpts = results
            target_id = self.model.target_id

            if person_poses is not None:

                image_detection_array_msg, pose_array_msg = self.get_human_msgs(person_poses, kpts, tracked_ids, self.frame_id)

                # Generate and publish the point cloud
                if self.human_image_detection_pub.get_subscription_count() > 0:
                    # Publish the Image Detections Array msg
                    self.human_image_detection_pub.publish(image_detection_array_msg)

                # Generate and publish the point cloud
                if self.human_cartesian_detection_pub.get_subscription_count() > 0:
                    # Publish the Pose Detections Array msg
                    self.human_cartesian_detection_pub.publish(pose_array_msg)

                # Publish the pose of the target person in case a cartesian tracker system needs it
                if (target_id is not None) and \
                    (target_id in tracked_ids) and \
                    (self.target_human_cartesian_detection_pub.get_subscription_count() > 0):
                        target_human_msg = self.get_target_human_msg(person_poses, target_id, tracked_ids, self.frame_id, empty=False)
                        self.target_human_cartesian_detection_pub.publish(target_human_msg)
                else:
                    # IF tracks available but none is the target person then publish empty msg
                    target_human_msg = self.get_target_human_msg(None, None, None, self.frame_id, empty=True)
                    self.target_human_cartesian_detection_pub.publish(target_human_msg)
        else:
            # If No Target Available in IMG Publish Empty Msg 
            target_human_msg = self.get_target_human_msg(None, None, None, self.frame_id, empty=True)
            self.target_human_cartesian_detection_pub.publish(target_human_msg)

        # Publish CompressedImage with detection Bounding Box for Visualizing the proper detection of the desired target person
        if self.publisher_debug_detection_image_compressed.get_subscription_count() > 0:
            # self.get_logger().info('Publishing Compressed Images with Detections for Debugging Purposes')
            self.publish_debug_img(rgb_image, bbox, kpts = kpts, tracked_ids = tracked_ids, target_id = target_id,  compressed=True)

        # Publish Image with detection Bounding Box for Visualizing the proper detection of the desired target person
        if self.publisher_debug_detection_image.get_subscription_count() > 0:
            # self.get_logger().info('Publishing Images with Detections for Debugging Purposes')
            self.publish_debug_img(rgb_image, bbox, kpts = kpts, tracked_ids = tracked_ids, target_id = target_id, compressed=False)

    def publish_debug_img(self, rgb_img, boxes, kpts, tracked_ids, target_id = None,  compressed = True):
        color_kpts = (255, 0, 0) 
        radius_kpts = 10
        thickness = 2

        if len(boxes) > 0 and len(kpts) > 0:

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = map(int, box)

                # Invalid detections go red
                cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (0, 0, 255), thickness)

                print("TARGET ID:", target_id)

                # overlay the Target person, this is to be considered later on
                if target_id is not None and tracked_ids[i] == target_id:
                    alpha = 0.2
                    overlay = rgb_img.copy()
                    cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (0, 255, 0), -1)
                    cv2.addWeighted(overlay, alpha, rgb_img, 1 - alpha, 0, rgb_img)

                # Just for debugging
                cv2.putText(rgb_img, f"ID: {tracked_ids[i]}" , (x1 + int((x2-x1)/2), y1 + int((y2-y1)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Draw the keypoints they have to be with respect the original image dimensions
                kpt = kpts[i]
                for j in range(kpt.shape[0]):
                    if kpt[j, 2] > 0.5:
                        u = int(kpt[j, 0])
                        v = int(kpt[j, 1])
                        cv2.circle(rgb_img, (u, v), radius_kpts, color_kpts, thickness)

        if compressed:
            self.publisher_debug_detection_image_compressed.publish(self.cv_bridge.cv2_to_compressed_imgmsg(rgb_img))
        else:
            self.publisher_debug_detection_image.publish(self.cv_bridge.cv2_to_imgmsg(rgb_img, encoding='bgr8'))


    def get_target_human_msg(self, poses, target_id, tracked_ids, frame_id, empty = False):
        pose_arrray_msg = PoseArray()
        pose_arrray_msg.header.stamp = self.get_clock().now().to_msg()
        pose_arrray_msg.header.frame_id = frame_id
        # pose_arrray_msg.header = self.header

        if empty:

            return pose_arrray_msg
        
        else:

            target_idx = np.where(tracked_ids == target_id)[0]

            pose_msg = Pose()

            pose_msg.position.x = float(poses[target_idx, 0])
            pose_msg.position.y = float(poses[target_idx, 1])
            pose_msg.position.z = float(poses[target_idx, 2])
            pose_msg.orientation.x = 0.
            pose_msg.orientation.y = 0.
            pose_msg.orientation.z = 0.
            pose_msg.orientation.w = 1.

            pose_arrray_msg.poses.append(pose_msg)

            return pose_arrray_msg


    def get_human_msgs(self, poses, kpts, tracked_ids, frame_id):
        # Prepare the Image Detections Array msg
        image_detection_array_msg = CandidateArray()
        image_detection_array_msg.header.stamp = self.get_clock().now().to_msg()
        image_detection_array_msg.header.frame_id = frame_id

        # Prepare the Pose Detections Array msg
        pose_array_msg = PoseArray()
        pose_array_msg.header.stamp = self.get_clock().now().to_msg()
        pose_array_msg.header.frame_id = frame_id

        for idx, (pose, kpt, tracked_id) in enumerate(zip(poses, kpts, tracked_ids)):

            # Initialize individual msgs
            image_detection_msg = Candidate()
            pose_msg = Pose()

            # Bounding Box Tracking ID
            image_detection_msg.id = int(tracked_id)

            # Target image position\
            # The indices 0 to 5 include all the keypoints belonging to the head, nose, ears, etc.
            # The one with the largest confidence is selected
            best_body_part_idx = np.argmax(kpt[0:5, 2])

            image_detection_msg.u = int(kpt[best_body_part_idx, 0].item())
            image_detection_msg.v = int(kpt[best_body_part_idx, 1].item())
            image_detection_msg.conf = kpt[best_body_part_idx, 2].item()
            image_detection_msg.dist = np.linalg.norm(pose)

            # Set the rotation using the composed quaternion
            pose_msg.position.x = pose[0]
            pose_msg.position.y = pose[1]
            pose_msg.position.z = pose[2]
            # Set the rotation using the composed quaternion
            pose_msg.orientation.x = 0.
            pose_msg.orientation.y = 0.
            pose_msg.orientation.z = 0.
            pose_msg.orientation.w = 1.

            # Append to the Corresponding Array msgs
            image_detection_array_msg.candidates.append(image_detection_msg)
            pose_array_msg.poses.append(pose_msg)

        return image_detection_array_msg, pose_array_msg
    
    def broadcast_human_pose(self, pose, orientation):
        # Broadcast the transform
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.frame_id
        transform.child_frame_id = "target_human"
        transform.transform.translation.x = pose[0]
        transform.transform.translation.y = pose[1]
        transform.transform.translation.z = pose[2]
        # Set the rotation using the composed quaternion
        transform.transform.rotation.x = orientation[0]
        transform.transform.rotation.y = orientation[1]
        transform.transform.rotation.z = orientation[2]
        transform.transform.rotation.w = orientation[3]
        self.tf_broadcaster.sendTransform(transform)
