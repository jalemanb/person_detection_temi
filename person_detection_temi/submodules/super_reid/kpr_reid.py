from .keypoint_promptable_reidentification.torchreid.scripts.builder import build_config, build_model
from .keypoint_promptable_reidentification.torchreid.metrics.distance import compute_distance_matrix_using_bp_features
from .keypoint_promptable_reidentification.torchreid.data.datasets.keypoints_to_masks import KeypointsToMasks
from .keypoint_promptable_reidentification.torchreid.data.transforms import build_transforms
from .keypoint_promptable_reidentification.torchreid.data import ImageDataset
from .keypoint_promptable_reidentification.torchreid.utils.constants import *
import torch
from torch.nn import functional as F
import numpy as np

class KPR(object):
    def __init__(self, cfg, kpt_conf = 0.8, device = 'cpu') -> None:
        self.cfg = cfg
        self.kpt_conf = kpt_conf
        self.device = device

        self.model = build_model(cfg)
        self.model.eval()

        _, self.preprocess, self.target_preprocess, self.prompt_preprocess = build_transforms(
                                                                                cfg.data.height,
                                                                                cfg.data.width,
                                                                                cfg,
                                                                                transforms=None,
                                                                                norm_mean=cfg.data.norm_mean,
                                                                                norm_std=cfg.data.norm_std,
                                                                                masks_preprocess=cfg.model.kpr.masks.preprocess,
                                                                                softmax_weight=cfg.model.kpr.masks.softmax_weight,
                                                                                background_computation_strategy=cfg.model.kpr.masks.background_computation_strategy,
                                                                                mask_filtering_threshold=cfg.model.kpr.masks.mask_filtering_threshold,
                                                                            )
        
        self.keypoints_to_prompt_masks = KeypointsToMasks(mode=cfg.model.kpr.keypoints.prompt_masks,
                                                                vis_thresh=kpt_conf,
                                                                vis_continous=cfg.model.kpr.keypoints.vis_continous,
                                                                )
        
        self.keypoints_to_target_masks = KeypointsToMasks(mode=cfg.model.kpr.keypoints.target_masks,
                                                            vis_thresh=kpt_conf,
                                                            vis_continous=False,
                                                            )   


    def extract_test_embeddings(self, model_output, test_embeddings):
        embeddings, visibility_scores, id_cls_scores, pixels_cls_scores, spatial_features, parts_masks = model_output
        embeddings_list = []
        visibility_scores_list = []
        embeddings_masks_list = []

        for test_emb in test_embeddings:
            embds = embeddings[test_emb]
            embeddings_list.append(embds if len(embds.shape) == 3 else embds.unsqueeze(1))
            if test_emb in bn_correspondants:
                test_emb = bn_correspondants[test_emb]
            vis_scores = visibility_scores[test_emb]
            visibility_scores_list.append(vis_scores if len(vis_scores.shape) == 2 else vis_scores.unsqueeze(1))
            pt_masks = parts_masks[test_emb]
            embeddings_masks_list.append(pt_masks if len(pt_masks.shape) == 4 else pt_masks.unsqueeze(1))

        assert len(embeddings) != 0

        embeddings = torch.cat(embeddings_list, dim=1)  # [N, P+2, D]
        visibility_scores = torch.cat(visibility_scores_list, dim=1)  # [N, P+2]
        embeddings_masks = torch.cat(embeddings_masks_list, dim=1)  # [N, P+2, Hf, Wf]

        return embeddings, visibility_scores, embeddings_masks, pixels_cls_scores
    
    def normalize(self, features):
        return F.normalize(features, p=2, dim=-1)
    
    def clamp_kpts(self, kpts, width, height):
        kpts = kpts.clone()
        kpts[:, :, 0] = torch.clamp(kpts[:, :, 0], 0, width - 1)  # Clip x
        kpts[:, :, 1] = torch.clamp(kpts[:, :, 1], 0, height - 1) 
        return kpts
    
    def extract(self, imgs, kpts, return_heatmaps = False): 
        # Input imgs are tensors of shape [Batch, C, W, H]
        # Input kpts are tensors of shape [Batch, 17, 3]

        imgs_list = []
        prompts_list = []
        kpts_list = []
        for i in range(imgs.shape[0]):
            # kpts = self.clamp_kpts(kpts, imgs.shape[3], imgs.shape[2])
            sample = {"image":imgs[i, :, :, :].permute(1, 2, 0).cpu().numpy(), "keypoints_xyc":kpts[i, :, :].cpu().numpy(), "negative_kps":[]}
            preprocessed_sample = ImageDataset.getitem(
                            sample,
                            self.cfg,
                            self.keypoints_to_prompt_masks,
                            self.prompt_preprocess,
                            self.keypoints_to_target_masks,
                            self.target_preprocess,
                            self.preprocess,
                            load_masks=True,
                        )
            imgs_list.append(preprocessed_sample["image"])
            prompts_list.append(preprocessed_sample["prompt_masks"])
            kpts_list.append(torch.Tensor(preprocessed_sample["keypoints_xyc"]))
        
        # Preprocessed images and Keypoint Prompts
        ready_imgs = torch.stack(imgs_list, dim = 0)
        ready_prompts = torch.stack(prompts_list, dim = 0)
        ready_kpts = torch.stack(kpts_list, dim = 0)

        output = self.model(images = ready_imgs, prompt_masks = ready_prompts, keypoints_xyc = ready_kpts)
        
        features = self.extract_test_embeddings(output,  self.cfg.model.kpr.test_embeddings)

        # The first Feature is the foreground and the rest are the K parts
        # For this inference model K = 5 which consist on 
        # k_five = {
        #     'head': ['nose', 'head_bottom', 'head_top', 'left_ear', 'right_ear'],
        #     'torso': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'],
        #     'arms': ['left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'],
        #     'legs': ['left_knee', 'right_knee'],
        #     'feet': ['left_ankle', 'right_ankle'],
        # }

        f_, v_, _, _ = features


        if self.cfg.test.normalize_feature:
            f_ = self.normalize(f_)

        if return_heatmaps:
            return f_, v_, ready_prompts

        return f_, v_

    def compare(self, fq, fg, vq, vg): 
        # Comparing Query Feature (Target Person) against Gallery features (Detected People)
        return compute_distance_matrix_using_bp_features(fq,
                                                         fg,
                                                         vq,
                                                         vg,
                                                         self.cfg.test.part_based.dist_combine_strat,
                                                         self.cfg.test.batch_size_pairwise_dist_matrix,
                                                         use_gpu = self.cfg.use_gpu,
                                                         metric = self.cfg.test.dist_metric,)
    