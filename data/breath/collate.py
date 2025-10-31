import torch
import numpy as np

class NumpyCollator:
    def __init__(self, transform=None):
        """
        Args:
            transform: Callable that takes (x, mz) tuple and returns dict with 'intensity' key
        """
        self.transform = transform

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    @staticmethod
    def minmax_0_1(x: np.ndarray) -> np.ndarray:
        x_min, x_max = x.min(), x.max()
        if x_max == x_min:
            return np.zeros_like(x, dtype=np.float32)
        return (x - x_min) / (x_max - x_min)

    def apply_transform(self, x: np.ndarray, mz=None):
        if self.transform is not None:
            return self.transform((x, mz))
        # Default transform
        intensity = self.relu(x)
        intensity = self.minmax_0_1(intensity)
        return {'intensity': torch.from_numpy(intensity)}

    def __call__(self, batch):
        """
        Args:
            batch: list of tuples (x, (class_label, id_label, cluster_label))
        Returns:
            dict: {
                'intensity': tensor [B, L],
                'class': tensor [B],
                'id': tensor [B],
                'cluster': tensor [B]
            }
        """
        all_intensity = []
        classes, ids, clusters = [], [], []

        for x, (class_label, id_label, cluster_label) in batch:
            # Ensure x is numpy array
            if isinstance(x, torch.Tensor):
                x = x.numpy()
                #class_label=class_label.numpy()
                #id_label = id_label.numpy()
                #cluster_label=cluster_label.numpy()
            out = self.apply_transform(x)
            all_intensity.append(out['intensity'])
            classes.append(class_label)
            ids.append(id_label)
            clusters.append(cluster_label)

        return {
            'data': torch.stack(all_intensity).unsqueeze(1),  # [B, L]
            'class': torch.tensor(classes),
            'id': torch.tensor(ids),
            'cluster': torch.tensor(clusters)
        }
