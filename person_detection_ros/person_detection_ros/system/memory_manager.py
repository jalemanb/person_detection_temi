import torch
import torch.nn.functional as F
import numpy as np
import random

class MemoryManager:
    def __init__(self, 
                 max_samples=1000, 
                 sim_thresh=0.8, 
                 feature_dim=512, 
                 num_parts=6,
                 pseudo_std = 0.001,
                 beta = 0.1,
                 use_pseudo = True):
        
        # Memory Manager Parameters
        self.max_samples = max_samples
        self.sim_thresh = sim_thresh
        self.feature_dim = feature_dim
        self.num_parts = num_parts
        self.pseudo_std = pseudo_std
        self.use_pseudo = use_pseudo
        self.beta = beta # percentage of time to use pseudonegative samples
        self.device = 'cpu'

        # Storage variables
        self.pos_feats = torch.zeros((max_samples, num_parts, feature_dim), dtype=torch.float32, device='cpu')
        self.neg_feats = torch.zeros((max_samples, num_parts, feature_dim), dtype=torch.float32, device='cpu')
        self.pos_vis = torch.zeros((max_samples, num_parts), dtype=torch.bool, device='cpu')
        self.neg_vis = torch.zeros((max_samples, num_parts), dtype=torch.bool, device='cpu')
        self.pos_counts = torch.zeros(max_samples, dtype=torch.int32, device='cpu')
        self.neg_counts = torch.zeros(max_samples, dtype=torch.int32, device='cpu')

        # Counters
        self.n_pos = 0
        self.n_neg = 0
        self._pos_sampled = set()
        self._neg_sampled = set()

        self.scaling_factor = torch.ones(1, 6, 1)

    def reset(self):
        # Storage variables
        self.pos_feats = torch.zeros((self.max_samples, self.num_parts, self.feature_dim), dtype=torch.float32, device='cpu')
        self.neg_feats = torch.zeros((self.max_samples, self.num_parts, self.feature_dim), dtype=torch.float32, device='cpu')
        self.pos_vis = torch.zeros((self.max_samples, self.num_parts), dtype=torch.bool, device='cpu')
        self.neg_vis = torch.zeros((self.max_samples, self.num_parts), dtype=torch.bool, device='cpu')
        self.pos_counts = torch.zeros(self.max_samples, dtype=torch.int32, device='cpu')
        self.neg_counts = torch.zeros(self.max_samples, dtype=torch.int32, device='cpu')

        # Counters
        self.n_pos = 0
        self.n_neg = 0
        self._pos_sampled = set()
        self._neg_sampled = set()

        self.scaling_factor = torch.ones(1, 6, 1)


    def _avg_cos_sim(self, f, v, mem_f, mem_v): # UNDERSTOOD
        f = F.normalize(f, dim=-1)
        mem_f = F.normalize(mem_f, dim=-1)
        dot = (f.unsqueeze(0) * mem_f).sum(dim=-1)
        mask = v.unsqueeze(0) & mem_v
        dot = dot * mask
        score = dot.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return score

    # This function ensures diversity in sampling by returning a unique index each time — i.e., no repetitions 
    # until all possible indices are used once. Once every index has been sampled, it resets and starts over.
    def _get_unique_index(self, total, sampled_set): # UNDERSTOOD
        available = list(set(range(total)) - sampled_set)
        if not available:
            sampled_set.clear()
            available = list(range(total))
        idx = random.choice(available)
        sampled_set.add(idx)
        return idx

    def _insert(self, feats, vis, mem_f, mem_v, counts, n_total): # UNDERSTOOD
        for i in range(feats.shape[0]):
            f, v = feats[i], vis[i]

            # If the storage has some features, do the comparison and find wether
            # to update the current prototype feature representation via averaging 
            # or in case the feature doesnt correspond then add it as a new one
            if n_total > 0:
                sims = self._avg_cos_sim(f, v, mem_f[:n_total], mem_v[:n_total])
                max_sim, max_idx = sims.max(0)
                if max_sim >= self.sim_thresh:
                    mem_f[max_idx] = f 
                    mem_v[max_idx] = v

                    counts[max_idx] += 1
                    continue

            # This is only called to initialize the storage when its empty
            # And add new features if the similarity is less than a threshold
            if n_total < self.max_samples: 
                mem_f[n_total] = f
                mem_v[n_total] = v
                counts[n_total] = 1
                n_total += 1

            # This is called to remove the least considered 
            else: 
                min_idx = counts[:n_total].argmin()
                mem_f[min_idx] = f
                mem_v[min_idx] = v
                counts[min_idx] = 1

        return n_total

    def insert_positive(self, feats_vis): # UNDERSTOOD
        feats = feats_vis[0].detach().cpu()
        vis = feats_vis[1].detach().cpu()
        self.n_pos = self._insert(feats, vis, self.pos_feats, self.pos_vis, self.pos_counts, self.n_pos)

    def insert_negative(self, feats_vis): # UNDERSTOOD
        feats = feats_vis[0].detach().cpu()
        vis = feats_vis[1].detach().cpu()
        self.n_neg = self._insert(feats, vis, self.neg_feats, self.neg_vis, self.neg_counts, self.n_neg)


    def get_sample(self): # UNDERSTOOD
        if self.n_pos == 0:
            return None


        # Get positive sample
        pos_idx = self._get_unique_index(self.n_pos, self._pos_sampled)
        pos_feat = self.pos_feats[pos_idx].unsqueeze(0) #+ torch.randn(1, self.num_parts, self.feature_dim)*0.001
        pos_vis = self.pos_vis[pos_idx].unsqueeze(0)
        is_pseudo = False

        # self.scaling_factor = torch.norm(pos_feat, p=2, dim=2, keepdim=True) 

        # Get negative sample (real or pseudo)
        if self.n_neg > 0 and (not self.use_pseudo or random.random() > self.beta):

            neg_idx = self._get_unique_index(self.n_neg, self._neg_sampled)
            neg_feat = self.neg_feats[neg_idx].unsqueeze(0) #+ torch.randn(1, self.num_parts, self.feature_dim)*0.001
            neg_vis = self.neg_vis[neg_idx].unsqueeze(0) 
        elif self.use_pseudo:
            # Get a Pseudo negative
            neg_feat = torch.randn(1, self.num_parts, self.feature_dim)*self.pseudo_std
            neg_vis = torch.ones(1, self.num_parts, dtype=torch.bool)
            is_pseudo = True


        # If there are Negative Samples Available or there are pseudo negatives allowed use them
        # Otherwise only use positive features
        if self.n_neg > 0 or self.use_pseudo:
            feats = torch.cat([neg_feat, pos_feat], dim=0)          # [2, 6, 512]
            vis = torch.cat([neg_vis, pos_vis], dim=0)              # [2, 6]
            labels = torch.tensor([[0], [1]], dtype=torch.float32)  # [2, 1]
        else:
            feats = pos_feat                                        # [1, 6, 512]
            vis = pos_vis                                           # [1, 6]
            labels =  torch.tensor([[1]], dtype=torch.float32)      # [1, 1]

        return feats, vis, labels, is_pseudo

    def save(self, path): # UNDERSTOOD
        np.savez_compressed(path,
            pos_feats=self.pos_feats[:self.n_pos].numpy(),
            pos_vis=self.pos_vis[:self.n_pos].numpy(),
            pos_counts=self.pos_counts[:self.n_pos].numpy(),
            neg_feats=self.neg_feats[:self.n_neg].numpy(),
            neg_vis=self.neg_vis[:self.n_neg].numpy(),
            neg_counts=self.neg_counts[:self.n_neg].numpy(),
            n_pos=self.n_pos,
            n_neg=self.n_neg)

    def load(self, path): # UNDERSTOOD
        data = np.load(path)
        self.n_pos = int(data['n_pos'])
        self.n_neg = int(data['n_neg'])

        self.pos_feats[:self.n_pos] = torch.from_numpy(data['pos_feats']).to(self.device)
        self.pos_vis[:self.n_pos] = torch.from_numpy(data['pos_vis']).to(self.device)
        self.pos_counts[:self.n_pos] = torch.from_numpy(data['pos_counts']).to(self.device)

        self.neg_feats[:self.n_neg] = torch.from_numpy(data['neg_feats']).to(self.device)
        self.neg_vis[:self.n_neg] = torch.from_numpy(data['neg_vis']).to(self.device)
        self.neg_counts[:self.n_neg] = torch.from_numpy(data['neg_counts']).to(self.device)