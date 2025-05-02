import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Literal

import h5py
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sam2.sam2_image_predictor import SAM2ImagePredictor
from skimage.measure import regionprops
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    # Dinov2Config,
    # Dinov2Model,
    AutoModel,
    HieraConfig,
    HieraModel,
    SamModel,
    SamProcessor,
)

from trackastra.data import CTCData
from trackastra.utils import add_timepoints_to_coords, masks2properties

MICRO_SAM_AVAILABLE = False
try:
    from micro_sam.util import get_sam_model as get_microsam_model
    MICRO_SAM_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# forward declarations for indexing
HieraFeatures = None
DinoV2Features = None
SAMFeatures = None
SAM2Features = None
MicroSAMFeatures = None

# Updated with actual class after each definition
# See register_backbone decorator
AVAILABLE_PRETRAINED_BACKBONES = {}

PretrainedFeatsExtractionMode = Literal[
    # "exact_patch",  # Uses the image patch centered on the detection for embedding
    "nearest_patch",  # Runs on whole image, then finds the nearest patch to the detection in the embedding
    "mean_patches",  # Runs on whole image, then averages the embeddings of all patches that intersect with the detection
    "max_patches"  # Runs on whole image, then takes the maximum for each feature dimension of all patches that intersect with the detection
]

PretrainedBackboneType = Literal[  # cannot unpack this directly in python < 3.11 so it has to be copied
    "facebook/hiera-tiny-224-hf",  # 768
    "facebook/dinov2-base",  # 768
    "facebook/sam-vit-base",  # 256
    "facebook/sam2-hiera-large",  # 256
    "facebook/sam2.1-hiera-base-plus",  # 256
    "microsam/vit_b_lm",
    "microsam/vit_l_lm",
    "random"
]


def register_backbone(model_name, feat_dim):
    def decorator(cls):
        AVAILABLE_PRETRAINED_BACKBONES[model_name] = {
            "class": cls,
            "feat_dim": feat_dim,
        }
        return cls
    return decorator


# Feature extraction from pretrained models
# Currently meant to wrap any transformers model
import time
from functools import wraps


def average_time_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not hasattr(wrapper, 'total_time'):
            wrapper.total_time = 0
            wrapper.call_count = 0
        
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        wrapper.total_time += elapsed_time
        wrapper.call_count += 1
        average_time = wrapper.total_time / wrapper.call_count
        
        print(f"Average time taken by {func.__name__}: {average_time:.6f} seconds over {wrapper.call_count} calls")
        
        return result
    
    return wrapper


def percentile_norm(b):
    for i, im in enumerate(b):
        p1, p99 = np.percentile(im, (1, 99.8))
        b[i] = (im - p1) / (p99 - p1)
        b[i] = np.clip(b[i], 0, 1)
    return b

# Augmentations for pre-trained models


from torchvision import tv_tensors
from torchvision.transforms import v2 as transforms


class BaseAugmentation(ABC):
    """Base class for windowed region augmentations."""
    def __init__(self, p: float = 0.5, rng_seed=None):
        self._p = p
        self._rng = np.random.RandomState(rng_seed)
        self.applied_record = {}

    def __call__(self, images: torch.Tensor, masks: tv_tensors.Mask):
        aug = self._get_aug()
        if self._p is None or self._rng.rand() < self._p:
            return aug(images, masks)
        return images, masks
    
    @abstractmethod
    def _get_aug(self) -> transforms.Compose:
        raise NotImplementedError()
    

class FlipAugment(BaseAugmentation):
    def __init__(self, p_horizontal: float = 0.5, p_vertical: float = 0.5, rng_seed=None):
        super().__init__(p=None, rng_seed=rng_seed)
        self._p_horizontal = p_horizontal
        self._p_vertical = p_vertical
    
    def __call__(self, images: torch.Tensor, masks: tv_tensors.Mask):
        if self._rng.rand() < self._p_horizontal:
            images = transforms.functional.hflip(images)
            masks = transforms.functional.hflip(masks)
            self.applied_record["hflip"] = True
        else:
            self.applied_record["hflip"] = False
        if self._rng.rand() < self._p_vertical:
            images = transforms.functional.vflip(images)
            masks = transforms.functional.vflip(masks)
            self.applied_record["vflip"] = True
        else:
            self.applied_record["vflip"] = False
        return images, masks
    
    def _get_aug(self) -> transforms.Compose:
        raise NotImplementedError("Use __call__ instead.")
        

class RotAugment(BaseAugmentation):
    
    def __init__(self, p: float = 0.5, degrees: int = 90, rng_seed=None):
        super().__init__(p, rng_seed=rng_seed)
        self.degrees = degrees
    
    def __call__(self, images, masks):
        aug = self._get_aug()
        if self._rng.rand() > self._p:
            return images, masks
        images = aug(images)
        masks = aug(masks)
        return images, masks
    
    def _get_aug(self):
        angle = self._rng.randint(1, 4) * self.degrees  # sample from 90, 180, 270
        self.applied_record["rotation"] = angle
        return partial(transforms.functional.rotate, angle=angle, expand=True)
    

class BrightnessJitter(BaseAugmentation):
    
    def __init__(self, bright_shift: float = 0.5, contrast_shift: float = 0.5, rng_seed=None):
        super().__init__(p=None, rng_seed=rng_seed)
        self._b_shift = bright_shift
        self._c_shift = contrast_shift
    
    def _get_aug(self):
        if self._b_shift is not None:
            bright = self._rng.uniform(0, self._b_shift)
        else:
            bright = None
        self.applied_record["brightness_jitter"] = bright
        if self._c_shift is not None:
            contrast = self._rng.uniform(0, self._c_shift)
        else:
            contrast = None
        self.applied_record["contrast_jitter"] = contrast
        return transforms.ColorJitter(brightness=bright, contrast=contrast)


class PretrainedAugmentations:
    """Augmentation pipeline to get augmented copies of model embeddings."""
    def __init__(self, rng_seed=None, normalize=True):
        self.aug_record = {}
        self.aug_list = [
            BrightnessJitter(bright_shift=0.75, contrast_shift=0.75, rng_seed=rng_seed),
            FlipAugment(p_horizontal=0.5, p_vertical=0.5, rng_seed=rng_seed),
            RotAugment(p=0.5, degrees=90, rng_seed=rng_seed),
        ]
        self._aug = transforms.Compose(
            self.aug_list
        )
        self.normalize = normalize

    def __call__(self, images: torch.Tensor, masks: tv_tensors.Mask, normalize=True) -> tuple[torch.Tensor, tv_tensors.Mask, dict]:
        """Applies the augmentations to the images."""
        images, masks = self.preprocess(images, masks, normalize=normalize)
        
        images = torch.unsqueeze(images, dim=1)  # add channel dimension (T, C, H, W) for augmentation
        masks = torch.unsqueeze(masks, dim=1)  # add channel dimension (T, C, H, W) for augmentation
        
        images, masks = self._aug(images, masks)
        # NOTE : most models do require 3 channels, but this will be done in FeatureExtractor, so the output is squeezed
        return images.squeeze(), masks.squeeze(), self.gather_records()
    
    def preprocess(self, images, masks, normalize=None):
        if not len(images.shape) == 3:
            raise ValueError(f"Images must be tensor of shape (T, H, W), got {len(images.shape)}D tensor.")
        if not len(masks.shape) == 3:
            raise ValueError(f"Masks must be tensor of shape (T, H, W), got {len(masks.shape)}D tensor.")
        
        if not isinstance(images, torch.Tensor):
            try:
                images = torch.tensor(images, dtype=torch.float32)
            except Exception as e:
                raise ValueError(f"Failed to convert images to tensor: {e}")
        if not isinstance(masks, tv_tensors.Mask):
            try:
                masks = tv_tensors.Mask(masks, dtype=torch.int64)
            except Exception as e:
                raise ValueError(f"Failed to convert masks to tensor: {e}")
        
        normalize = normalize if normalize is not None else self.normalize
        if normalize:
            images = percentile_norm(images)
            
        return images, masks
    
    def gather_records(self):
        """Gathers the applied augmentation records."""
        self.aug_record = {}
        for aug in self.aug_list:
            self.aug_record.update(aug.applied_record)
        return self.aug_record


@dataclass
class PretrainedFeatureExtractorConfig:
    """model_name (str):
        Specify the pretrained backbone to use.
    save_path (str | Path):
        Specify the path to save the embeddings.
    batch_size (int):
        Specify the batch size to use for the model.
    mode (str):
        Specify the mode to use for the model.
        Currently available modes are "nearest_patch", "mean_patches", and "max_patches".
    device (str):
        Specify the device to use for the model.
        If not set and "pretrained_feats" is used, the device is automatically set by default to "cuda", "mps" or "cpu" as available.
    n_augmented_copies (int):
        How many augmented copies of the embeddings to create. If 0, only the original embeddings are saved. Creates n+1 embeddings entries total. 
    additional_features (str):
        Specify any additional features (from regionprops) to include in the extraction process. See WRFeat documentation for available features. Unused if None.
    pca_components (int):
        Specify the number of PCA components to use for dimensionality reduction of the features. Unused if None.
    pca_preprocessor_path (str | Path):
        Specify the path to the pickled PCA preprocessor. This is used to transform the features to a reduced PCA feature space.
    """
    model_name: PretrainedBackboneType
    save_path: str | Path = None
    batch_size: int = 4
    mode: PretrainedFeatsExtractionMode = "nearest_patch"
    device: str | None = None
    feat_dim: int = None
    additional_features: str | None = None  # for regionprops features
    pca_components: int = None  # for PCA reduction of the features
    pca_preprocessor_path: str | Path = None  # for PCA preprocessor path
    
    def __post_init__(self):
        self._guess_device()
        self._check_model_availability()
        
    def _check_model_availability(self):
        if self.model_name not in AVAILABLE_PRETRAINED_BACKBONES.keys():
            raise ValueError(f"Model {self.model_name} is not available for feature extraction.")
        self.feat_dim = AVAILABLE_PRETRAINED_BACKBONES[self.model_name]["feat_dim"]
        if self.additional_features is not None:
            self.feat_dim += CTCData.FEATURES_DIMENSIONS[self.additional_features][2] + 1  # TODO if this ever accepts 3D data this will be incorrect
            # TODO also figure out why the dimensions require +1 to be correct
            if self.additional_features not in CTCData.FEATURES_DIMENSIONS:
                raise ValueError(f"Additional feature {self.additional_features} is not valid.")
        if self.pca_components is not None:
            self.feat_dim = self.pca_components

    def _guess_device(self):
        if self.device is None:
            should_use_mps = (
                torch.backends.mps.is_available()
                and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") is not None
                and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "0"
            )
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else (
                    "mps"
                    if should_use_mps and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK")
                    else "cpu"
                )
            )
        
        try:
            torch.device(self.device)  # check if device is valid
        except Exception as e:
            raise ValueError(f"Invalid device: {self.device}") from e

    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)
    

class FeatureExtractor(ABC):
    model_name = None
    _available_backbones = None

    def __init__(
        self, 
        image_size: tuple[int, int],
        save_path: str | Path,
        batch_size: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mode: PretrainedFeatsExtractionMode = "nearest_patch",
        ):
        # Image processor extra args
        self.im_proc_kwargs = {
            "do_rescale": False,
            "do_normalize": False,
            "do_resize": True,
            "return_tensors": "pt",
            "do_center_crop": False,
        }
        # Model specs
        self.model = None
        self._input_size: int = None
        self._final_grid_size: int = None
        self.n_channels: int = None
        self.hidden_state_size: int = None
        self.model_patch_size: int = None
        # Data specs
        self.orig_image_size = image_size
        self.orig_n_channels = 1
        # Batch options and preprocessing
        self.percentile_norm = True
        self.rescale_batches = False
        self.channel_first = True
        self.batch_return_type: Literal["list[np.ndarray]", "np.ndarray"] = "np.ndarray"
        # Parameters for embedding extraction
        self.batch_size = batch_size
        self.device = device
        self.mode = mode
        self.additional_features = None
        # Saving parameters
        self.save_path: str | Path = save_path
        self.do_save = True
        self.force_recompute = False
        self.embeddings = None
        
        if not isinstance(self.save_path, Path):
            self.save_path = Path(self.save_path)
        
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def input_size(self):
        return self._input_size
    
    @input_size.setter
    def input_size(self, value: int):
        if value is None:
            raise ValueError("Input size cannot be None.")
        self._input_size = value
        self._set_model_patch_size()
        
    @property
    def final_grid_size(self):
        return self._final_grid_size
    
    @final_grid_size.setter
    def final_grid_size(self, value: int):
        self._final_grid_size = value
        self._set_model_patch_size()
        
    @property
    def model_name_path(self):
        return self.model_name.replace("/", "-")
    
    @classmethod
    def from_model_name(cls, 
                        model_name: PretrainedBackboneType, 
                        image_shape: tuple[int, int], 
                        save_path: str | Path, 
                        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
                        mode="nearest_patch",
                        additional_features=None,
                        # mode="mean_patches"
                        ):
        cls._available_backbones = AVAILABLE_PRETRAINED_BACKBONES
        if model_name not in cls._available_backbones:
            raise ValueError(f"Model {model_name} is not available for feature extraction.")
        logger.info(f"Using model {model_name} with mode {mode} for pretrained feature extraction.")
        backbone = cls._available_backbones[model_name]["class"]
        backbone.model_name = model_name
        backbone.additional_features = additional_features
        return backbone(image_shape, save_path, device=device, mode=mode)

    @classmethod
    def from_config(cls, config: PretrainedFeatureExtractorConfig, image_shape: tuple[int, int], save_path: str | Path):
        cls._available_backbones = AVAILABLE_PRETRAINED_BACKBONES
        if config.model_name not in cls._available_backbones:
            raise ValueError(f"Model {config.model_name} is not available for feature extraction.")
        logger.info(f"Using model {config.model_name} with mode {config.mode} for pretrained feature extraction.")
        backbone = cls._available_backbones[config.model_name]["class"]
        backbone.model_name = config.model_name
        backbone.additional_features = config.additional_features
        
        return backbone(
            image_shape, 
            save_path, 
            batch_size=config.batch_size, 
            device=config.device, 
            mode=config.mode,
            n_augmented_copies=config.n_augmented_copies,
            aug_pipeline=PretrainedAugmentations() if config.n_augmented_copies > 0 else None,
        )
        
    def clear_model(self):
        """Clears the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            logger.info("Model cleared from memory.")
        else:
            logger.warning("No model to clear from memory.")
    
    def _set_model_patch_size(self):
        if self.final_grid_size is not None and self.input_size is not None:
            self.model_patch_size = self.input_size // self.final_grid_size
            # if not self.input_size % self.final_grid_size == 0:
            # raise ValueError("The input size must be divisible by the final grid size.")

    def compute_region_features(
            self, 
            coords,
            masks=None, 
            timepoints=None,
            labels=None,
        ) -> torch.Tensor:  # (n_regions, embedding_size)
        """Extracts embeddings from the model.
        
        Args:
        - coords (np.ndarray): Coordinates of the detections.
        - masks (np.ndarray): Masks where each region has a unique label for each timepoint.
        - timepoints (np.ndarray): For each region, contains the corresponding timepoint.
        - labels (np.ndarray): Unique labels of the regions.
        """
        feats = torch.zeros(len(coords), self.hidden_state_size, device=self.device)
        
        if self.mode == "nearest_patch":
            feats = self._nearest_patches(coords, masks)  
        elif self.mode == "mean_patches":
            if masks is None or labels is None or timepoints is None:
                raise ValueError("Masks and labels must be provided for mean patch mode.")
            feats = self._agg_patches(masks, timepoints, labels)
        elif self.mode == "max_patches":
            if masks is None or labels is None or timepoints is None:
                raise ValueError("Masks and labels must be provided for max patch mode.")
            feats = self._agg_patches(masks, timepoints, labels, agg=torch.max)
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented.")
        
        assert feats.shape == (len(coords), self.hidden_state_size)
        return feats  # (n_regions, embedding_size)
    
    def precompute_image_embeddings(self, images):  # , windows, window_size):
        """Precomputes embeddings for all images."""
        missing = self._check_missing_embeddings()
        all_embeddings = torch.zeros(len(images), self.final_grid_size**2, self.hidden_state_size, device=self.device)
        if missing:
            for ts, batches in tqdm(self._prepare_batches(images), total=len(images) // self.batch_size, desc="Computing embeddings"):
                embeddings = self._run_model(batches)
                if torch.any(embeddings.isnan()):
                    raise RuntimeError("NaN values found in features.")
                # logger.debug(f"Embeddings shape: {embeddings.shape}")
                all_embeddings[ts] = embeddings
            self.embeddings = all_embeddings
            self._save_features(all_embeddings)
        return self.embeddings

    def _extract_region_embeddings(self, all_frames_embeddings, window, start_index, remaining=None):
        window_coords = window["coords"]
        window_timepoints = window["timepoints"]
        window_masks = window["mask"]
        window_labels = window["labels"]
        
        n_regions_per_frame, features = self.extract_embedding(window_masks, window_timepoints, window_labels, window_coords)
        
        for i in range(remaining or len(n_regions_per_frame)):
            # if computing remaining frames' embeddings, start from the end
            obj_per_frame = n_regions_per_frame[-i - 1] if remaining else n_regions_per_frame[i]
            frame_index = start_index + i if not remaining else np.max(window_timepoints) - i
            # logger.debug(f"Frame {frame_index} has {obj_per_frame} objects.")
            all_frames_embeddings[frame_index] = features[:obj_per_frame]
            features = features[obj_per_frame:]
    
    def extract_embedding(self, masks, timepoints, labels, coords):
        if masks.shape[-2:] != self.orig_image_size:
            # This should not be occur since each folder is loaded as a separate CTCData
            logger.debug(f"Input shape change detected: {masks.shape[-2:]} from {self.orig_image_size}.")
            self.orig_image_size = masks.shape[-2:]
        n_regions_per_frame = np.unique(timepoints, return_counts=True)[1]
        tot_regions = n_regions_per_frame.sum()
        coords_txy = np.concatenate((timepoints[:, None], coords), axis=-1)
        if coords_txy.shape[0] != tot_regions:
            raise RuntimeError(f"Number of coords ({coords_txy.shape[0]}) does not match the number of coordinates ({timepoints.shape[0]}).")
        features = self.compute_region_features(
            masks=masks,
            coords=coords_txy,
            timepoints=timepoints,
            labels=labels,
        )
        if torch.isnan(features).any():
            raise RuntimeError("NaN values found in features.")
        if tot_regions != features.shape[0]:
            raise RuntimeError(f"Number of regions ({n_regions_per_frame}) does not match the number of embeddings ({features.shape[0]}).")
        return n_regions_per_frame, features
    
    @abstractmethod
    def _run_model(self, images) -> torch.Tensor:  # must return (B, grid_size**2, hidden_state_size)
        """Extracts embeddings from the model."""
        pass
    
    def _normalize_batch(self, b):
        b = percentile_norm(b)
        if self.rescale_batches:
            b = b * 255.0
        return b
    
    def _prepare_batches(self, images):
        """Prepares batches of images for embedding extraction."""
        for i in range(0, len(images), self.batch_size):
            end = i + self.batch_size
            end = len(images) if end > len(images) else end
            batch = np.expand_dims(images[i:end], axis=1)  # (B, C, H, W)
            
            # required by AutoImageProcessor (PIL Image needs [0, 1] range)
            if self.percentile_norm:
                batch = self._normalize_batch(batch)
            timepoints = range(i, end)
            if self.n_channels > 1:  # repeat channels if needed
                if self.orig_n_channels > 1 and self.orig_n_channels != self.n_channels:
                    raise ValueError("When more than one original channel is provided, the number of channels in the model must match the number of channels in the input.")
                batch = np.repeat(batch, self.n_channels, axis=1)
            if not self.channel_first:
                batch = np.moveaxis(batch, 1, -1)
            if self.batch_return_type == "list[np.ndarray]":
                batch = list([im for im in batch])
            yield timepoints, batch
    
    def _map_coords_to_model_grid(self, coords):
        scale_x = self.input_size / self.orig_image_size[0]
        scale_y = self.input_size / self.orig_image_size[1]
        coords = np.array(coords)
        patch_x = (coords[:, 1] * scale_x).astype(int)
        patch_y = (coords[:, 2] * scale_y).astype(int)
        patch_coords = np.column_stack((coords[:, 0], patch_x, patch_y))
        return patch_coords
    
    def _find_nearest_cell(self, patch_coords):
        x_idxs = patch_coords[:, 1] // self.model_patch_size
        y_idxs = patch_coords[:, 2] // self.model_patch_size
        patch_idxs = np.column_stack((patch_coords[:, 0], x_idxs, y_idxs)).astype(int)
        return patch_idxs
    
    def _find_bbox_cells(self, regions: dict, cell_height: int, cell_width: int):
        """Finds the cells in a grid that a bounding box belongs to.
        
        Args:
        - regions (dict): Dictionary from regionprops. Must contain bbox.
        - cell_height (int): Height of a cell in the grid.
        - cell_width (int): Width of a cell in the grid.
    
        Returns:
        - tuple: A tuple containing the grid cell indices that the bounding box intersects.
        """
        mask_patches = {}
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            patches = self._find_region_cells(minr, minc, maxr, maxc, cell_height, cell_width)
            mask_patches[region.label] = patches
            
        return mask_patches
    
    @staticmethod
    def _find_region_cells(minr, minc, maxr, maxc, cell_height, cell_width):
        start_patch_y = minr // cell_height
        end_patch_y = (maxr - 1) // cell_height
        start_patch_x = minc // cell_width
        end_patch_x = (maxc - 1) // cell_width
        patches = np.array([(i, j) for i in range(start_patch_y, end_patch_y + 1) for j in range(start_patch_x, end_patch_x + 1)])
        return patches
    
    def _find_patches_for_masks(self, image_mask: np.ndarray) -> dict:
        """Find which patches in a grid each mask belongs to using regionprops.
        
        Args:
        - image_masks (np.ndarray): Masks where each region has a unique label.
                
        Returns:
        - mask_patches (dict): Dictionary with region labels as keys and lists of patch indices as values.
        """
        patch_height, patch_width = image_mask.shape[0] // self.final_grid_size, image_mask.shape[1] // self.final_grid_size
        regions = regionprops(image_mask)
        return self._find_bbox_cells(regions, patch_height, patch_width)
    
    def _debug_show_patches(self, embeddings, masks, coords, patch_idxs):
        import napari
        v = napari.Viewer()
        # v.add_labels(masks)
        e = embeddings.detach().cpu().numpy().swapaxes(1, 2).reshape(-1, self.hidden_state_size, self.final_grid_size, self.final_grid_size).swapaxes(0, 1)
        v.add_image(
            e,
            name="Embeddings",
        )
        # add red points at patch indices for the relevant frame
        points = np.zeros((len(patch_idxs) * self.hidden_state_size, 3))
        for i, (t, y, x) in enumerate(patch_idxs):
            point = np.array([t, y, x])
            points[i * self.hidden_state_size:(i + 1) * self.hidden_state_size] = np.tile(point, (self.hidden_state_size, 1))
        
        v.add_points(points, size=1, face_color='red', name='Patch Indices')

        from skimage.transform import resize
        masks_resized = resize(masks[0], (self.final_grid_size, self.final_grid_size), anti_aliasing=False, order=0, preserve_range=True)
        v.add_labels(masks_resized)
        
        napari.run()
            
    def _nearest_patches(self, coords, masks=None):
        """Finds the nearest patches to the detections in the embedding."""
        # find coordinate patches from detections
        patch_coords = self._map_coords_to_model_grid(coords)
        patch_idxs = self._find_nearest_cell(patch_coords)
        # logger.debug(f"Patch indices: {patch_idxs}")

        # load the embeddings and extract the relevant ones
        feats = torch.zeros(len(coords), self.hidden_state_size, device=self.device)
        indices = [y * self.final_grid_size + x for _, y, x in patch_idxs]
        
        unique_timepoints = list(set(t for t, _, _ in patch_idxs))
        # logger.debug(f"Unique timepoints: {unique_timepoints}")
        embeddings = self._load_features()
        
        # t = coords[0][0]
        # if t > 80:
        #    self._debug_show_patches(embeddings, masks, coords, patch_idxs)

        # logger.debug(f"Embeddings shape: {embeddings.shape}")
        embeddings_dict = {t: embeddings[t] for t in unique_timepoints}
        try:
            for i, (t, _, _) in enumerate(patch_idxs):
                feats[i] = embeddings_dict[t][indices[i]]
        except KeyError as e:
            logger.error(f"KeyError: {e} - Check if the timepoint exists in embeddings_dict.")
        except IndexError as e:
            # TODO improve handling of this error. Maybe check shape earlier
            logger.error(f"IndexError: {e} - Embeddings exist but do not have the correct shape. Did the model input size change ? If so, please delete saved embeddings and recompute.")
            
        return feats
    
    # @average_time_decorator
    def _agg_patches(self, masks, timepoints, labels, agg=torch.mean):
        """Averages the embeddings of all patches that intersect with the detection.
        
        Args:
            - masks (np.ndarray): Masks where each region has a unique label (t x H x W).
            - timepoints (np.ndarray): For each region, contains the corresponding timepoint. (n_regions)
            - labels (np.ndarray): Unique labels of the regions. (n_regions)
            - agg (callable): Aggregation function to use for averaging the embeddings.
        """
        try:
            n_regions = len(timepoints)
            timepoints_shifted = timepoints - timepoints.min()
        except ValueError:
            logger.error("Error: issue computing shifted timepoints.")
            logger.error(f"Regions: {len(timepoints)}")
            logger.error(f"Timepoints: {timepoints}")
            return torch.zeros(n_regions, self.hidden_state_size, device=self.device)

        feats = torch.zeros(n_regions, self.hidden_state_size, device=self.device)
        patches = []
        times = np.unique(timepoints_shifted)
        patches_res = joblib.Parallel(n_jobs=8, backend="threading")(
            joblib.delayed(self._find_patches_for_masks)(masks[t]) for t in times
        )   
        patches = {t: patch for t, patch in zip(times, patches_res)}
        # logger.debug(f"Patches : {patches}")
            
        embeddings = self._load_features()    

        def process_region(i, t):
            patches_feats = []
            for patch in patches[t][labels[i]]:
                patches_feats.append(embeddings[t][patch[1] * self.final_grid_size + patch[0]])
            aggregated = agg(torch.stack(patches_feats), dim=0)
            # If agg is torch.max, extract only the values
            if isinstance(aggregated, torch.return_types.max):
                aggregated = aggregated.values
            return aggregated
        
        res = joblib.Parallel(n_jobs=8, backend="threading")(
            joblib.delayed(process_region)(i, t) for i, t in enumerate(timepoints_shifted)
        )

        for i, r in enumerate(res):
            feats[i] = r

        return feats

    def _exact_patch(self, imgs, masks, coords):
        """Uses the image patch centered on the detection for embedding."""
        raise NotImplementedError()
    
    def _save_features(self, features):  # , timepoint):
        """Saves the features to disk."""
        # save_path = self.save_path / f"{timepoint}_{self.model_name_path}_features.npy"
        self.embeddings = features
        if not self.do_save:
            return
        save_path = self.save_path / f"{self.model_name_path}_features.npy"
        np.save(save_path, features.cpu().numpy())
        assert save_path.exists(), f"Failed to save features to {save_path}"
    
    def _load_features(self):  # , timepoint):
        """Loads the features from disk."""
        # load_path = self.save_path / f"{timepoint}_{self.model_name_path}_features.npy"
        if self.embeddings is None:
            load_path = self.save_path / f"{self.model_name_path}_features.npy"
            if load_path.exists():
                features = np.load(load_path)
                assert features is not None, f"Failed to load features from {load_path}"
                if np.any(np.isnan(features)): 
                    raise RuntimeError(f"NaN values found in features loaded from {load_path}.")
                # check feature shape consistency
                if features.shape[1] != self.final_grid_size**2 or features.shape[2] != self.hidden_state_size:
                    logger.error(f"Saved embeddings found, but shape {features.shape} does not match expected shape {('n_frames', self.final_grid_size**2, self.hidden_state_size)}.")
                    logger.error("Embeddings will be recomputed.")
                    return None
                self.embeddings = torch.tensor(features).to(self.device)
                logger.info("Saved embeddings loaded.")
                return self.embeddings
            else:
                logger.info(f"No saved embeddings found at {load_path}. Features will be computed.")
                return None
        else:
            return self.embeddings
   
    def _check_missing_embeddings(self):
        """Checks if embeddings for the model already exist or are missing.
        
        Returns whether the embeddings need to be recomputed.
        """
        if self.force_recompute:
            return True
        try:
            features = self._load_features()
        except FileNotFoundError:
            return True
        if features is None:
            return True
        else:
            logger.info(f"Embeddings for {self.model_name} already exist. Skipping embedding computation.")
        return False


class FeatureExtractorAugWrapper:
    """Wrapper for the FeatureExtractor class to apply augmentations."""
    def __init__(
            self,
            extractor: FeatureExtractor,
            augmenter: PretrainedAugmentations, 
            n_aug: int = 1
        ):
        self.extractor = extractor
        self.n_aug = n_aug
        self.aug_pipeline = augmenter
        self.all_aug_features = {}  # n_aug -> {aug_id: {applied_augs, data}}
        # data -> {t: {lab: {"coords": coords, "features": features}}}
        
        self.extractor.force_recompute = True
        self.extractor.do_save = False  # do not save intermediate features (augmented image embeddings)
        # instead, we will save the augmented features + coordinates on a per-object basis in an HDF5 file

        self.save_path = None
        self.force_recompute = False
        
        self._debug_view = None
        
    def get_save_path(self):
        root_path = self.extractor.save_path / "aug"
        if not root_path.exists():
            root_path.mkdir(parents=True, exist_ok=True)
        self.save_path = root_path / f"{self.extractor.model_name_path}_aug.h5"
        return self.save_path
        
    def _check_existing(self):
        """Checks if an h5 file already exists, and which augmentations are already saved."""
        self.get_save_path()
        if not self.save_path.exists() or self.force_recompute:
            return False, None, None
        logger.info(f"Loading existing features from {self.save_path}...")
        with h5py.File(self.save_path, "r") as f:
            existing_augs = list(f.keys())
            # remove all keys that are not integers
            existing_augs = [aug for aug in existing_augs if aug.isdigit()]
            features_dict = self.load_all_features()
        logger.info("Done.")
        return True, existing_augs, features_dict
    
    def _compute(self, images, masks):
        """Computes the features for the images and masks."""
        embs = self.extractor.precompute_image_embeddings(images)
        if self._debug_view is not None:
            embs = embs.cpu().numpy()
            logger.debug(f"Embeddings shape: {embs.shape}")
            embs = embs.reshape(-1, self.extractor.final_grid_size, self.extractor.final_grid_size, self.extractor.hidden_state_size)
            embs = np.moveaxis(embs, 3, 0)
            self._debug_view.add_image(embs, name="Embeddings", colormap="inferno")
            self._debug_view.add_image(images.cpu().numpy(), name="Images", colormap="viridis")
            self._debug_view.add_labels(masks.cpu().numpy(), name="Masks")
        labels, ts, coords = masks2properties(images.cpu().numpy(), masks.cpu().numpy())
        coords = add_timepoints_to_coords(coords, ts)
        feats = self.extractor.compute_region_features(coords, masks, ts, labels)
        feat_dict = self._create_feat_dict(labels, ts, coords, feats)
        return feat_dict
    
    def _compute_original(self, images, masks):
        """Computes the original features for the images and masks."""
        images, masks = self.aug_pipeline.preprocess(images, masks)
        orig_feat_dict = self._compute(images, masks)
        return orig_feat_dict
    
    def _compute_augmented(self, images, masks):
        images, masks = self.aug_pipeline.preprocess(images, masks)
        aug_images, aug_masks, aug_record = self.aug_pipeline(images, masks)
        
        im_shape, masks_shape = aug_images.shape, aug_masks.shape
        assert im_shape == masks_shape, f"Augmented images shape {im_shape} does not match augmented masks shape {masks_shape}."
        if im_shape[-2:] != self.extractor.orig_image_size:
            self.extractor.orig_image_size = im_shape[-2:]
            
        aug_feat_dict = self._compute(aug_images, aug_masks)
        return aug_feat_dict, aug_record
        
    def compute_all_features(self, images, masks) -> dict:
        """Augments the images and masks, computes the embeddings, and saves features incrementally."""
        # check existing features
        present, existing_augs, existing_features_dict = self._check_existing()
        if present:
            logger.debug(f"Saved features found at {self.save_path}.")
            if len(existing_augs) == self.n_aug + 1:
                logger.info(f"All {self.n_aug} augmentations + original already exist. Loading existing features.")
                self.all_aug_features = existing_features_dict
                return self.all_aug_features
            else:
                logger.info("No existing augmentations found.")
        if existing_augs is None:
            existing_aug_ids = []
        else:
            existing_aug_ids = [int(aug.split("_")[-1]) for aug in existing_augs]        
        
        if 0 not in existing_aug_ids:
            orig_feat_dict = self._compute_original(images, masks)
        else:
            logger.info("Original features already exist. Skipping computation for original features.")
            orig_feat_dict = existing_features_dict[0]["data"]
            
        self.all_aug_features = {
            0: {
                "data": orig_feat_dict,
            }
        }
        if 0 not in existing_aug_ids:
            self._save_features(0, self.all_aug_features[0])
        
        self.aug_pipeline.normalize = False  # do not re-normalize the images
        for n in range(self.n_aug):
            if n + 1 in existing_aug_ids:
                logger.info(f"Augmentation {n + 1} already exists. Skipping computation.")
                aug_feat_dict = existing_features_dict[n + 1]["data"]
                aug_record = existing_features_dict[n + 1]["applied_augs"]
            else:
                aug_feat_dict, aug_record = self._compute_augmented(images, masks)
            
            self.all_aug_features[n + 1] = {
                "applied_augs": aug_record,
                "data": aug_feat_dict,
            }
            if n + 1 not in existing_aug_ids:
                self._save_features(n + 1, self.all_aug_features[n + 1])

        return self.all_aug_features

    def _create_feat_dict(self, labels, ts, coords, features):
        """Creates a dictionary with the augmented features."""
        aug_feat_dict = {}
        features = features.cpu().numpy()
        for i, (t, lab) in enumerate(zip(ts, labels)):
            t = int(t)
            lab = int(lab)
            if t not in aug_feat_dict:
                aug_feat_dict[t] = {}
            aug_feat_dict[t][lab] = {
                "coords": coords[i],
                "features": features[i],
            }
        return aug_feat_dict

    def _save_features(self, aug_id: int, aug_data: dict):
        """Saves the features for a specific augmentation to disk as HDF5."""
        root_path = self.extractor.save_path / "aug"
        if not root_path.exists():
            root_path.mkdir(parents=True, exist_ok=True)
        self.save_path = root_path / f"{self.extractor.model_name_path}_aug.h5"

        with h5py.File(self.save_path, "a") as f:
            # Check if the group already exists and delete it if necessary
            group_name = f"augmentation_{aug_id}"
            if group_name in f:
                del f[group_name]
            
            group = f.create_group(group_name)
            # Store applied augmentations as attributes
            try:
                group.attrs["applied_augs"] = json.dumps(aug_data["applied_augs"])
            except KeyError:
                pass

            # Save data for each timepoint and label
            for t, data in aug_data["data"].items():
                t_group = group.create_group(f"timepoint_{t}")
                for lab, lab_data in data.items():
                    lab_group = t_group.create_group(f"label_{lab}")
                    lab_group.create_dataset("coords", data=lab_data["coords"], compression="gzip")
                    lab_group.create_dataset("features", data=lab_data["features"], compression="gzip")

        logger.info(f"Augmented features for augmentation {aug_id} saved to {self.save_path}.")

    def load_all_features(self) -> dict:
        """Loads all features from disk."""
        self.get_save_path()
        if not self.save_path.exists():
            raise FileNotFoundError(f"Path {self.save_path} does not exist.")
        
        features = FeatureExtractorAugWrapper.load_features(self.save_path)
        self.all_aug_features = features
        return features

    @staticmethod
    def load_features(path: str | Path) -> dict:
        """Loads the features for a specific augmentation from disk."""
        if not isinstance(path, Path):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")
        
        features = {}
        with h5py.File(path, "r") as f:
            for aug_id, group in f.items():
                if not aug_id.isdigit():
                    continue
                aug_id = int(aug_id.split("_")[-1])
                try:
                    applied_augs = json.loads(group.attrs["applied_augs"])
                except KeyError:
                    applied_augs = None
                data = {}
                for t, t_group in group.items():
                    if isinstance(t, str):
                        continue
                    t = int(t.split("_")[-1])
                    data[t] = {}
                    for lab, lab_group in t_group.items():
                        lab = int(lab.split("_")[-1])
                        data[t][lab] = {
                            "coords": np.array(lab_group["coords"]),
                            "features": np.array(lab_group["features"]),
                        }
                features[aug_id] = {
                    "applied_augs": applied_augs,
                    "data": data,
                }
        return features
       

##############
@register_backbone("facebook/hiera-tiny-224-hf", 768)
class HieraFeatures(FeatureExtractor):
    model_name = "facebook/hiera-tiny-224-hf"

    def __init__(
        self, 
        image_size: tuple[int, int],
        save_path: str | Path,
        batch_size: int = 16,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mode: PretrainedFeatsExtractionMode = "nearest_patch",
        ):
        super().__init__(image_size, save_path, batch_size, device, mode)
        # self.input_size = 224
        self.input_mul = 3
        self.input_size = self.input_mul * 224
        self.final_grid_size = 7 * self.input_mul  # 14x14 grid
        self.n_channels = 3
        self.hidden_state_size = 768
        self.rescale_batches = False

        ##
        self.im_proc_kwargs["size"] = (self.input_size, self.input_size)
        ##
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
        config = HieraConfig.from_pretrained(self.model_name)
        config.image_size = [self.input_size, self.input_size]
        # logger.debug(f"Config: {config}")
        # self.model = HieraModel.from_pretrained(self.model_name)
        # self.model.config.image_size = [self.input_size, self.input_size]
        self.model = HieraModel(config)
        self.model.to(self.device)
        # self.model.embeddings.patch_embeddings.num_patches = (self.input_size // self.model.config.patch_size[0]) ** 2
        # self.model.embeddings.position_embeddings = torch.nn.Parameter(
        #     torch.zeros(1, self.model.embeddings.patch_embeddings.num_patches + 1, self.hidden_state_size)
        # )
        # self.model.embeddings.position_ids = torch.arange(0, self.model.embeddings.patch_embeddings.num_patches + 1).unsqueeze(0)

    def _run_model(self, images) -> torch.Tensor:
        """Extracts embeddings from the model."""
        # images = self._normalize_batch(images)
        inputs = self.image_processor(images, **self.im_proc_kwargs).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return outputs.last_hidden_state
    

@register_backbone("facebook/dinov2-base", 768)
class DinoV2Features(FeatureExtractor):
    model_name = "facebook/dinov2-base"

    def __init__(
        self, 
        image_size: tuple[int, int],
        save_path: str | Path,
        batch_size: int = 16,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mode: PretrainedFeatsExtractionMode = "nearest_patch",
        ):
        super().__init__(image_size, save_path, batch_size, device, mode)
        self.input_size = 224
        self.final_grid_size = 16  # 16x16 grid
        self.n_channels = 3  # expects RGB images
        self.hidden_state_size = 768
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
        ##
        self.im_proc_kwargs["size"] = (self.input_size, self.input_size)
        ##
        self.model = AutoModel.from_pretrained(self.model_name)
        self.rescale_batches = False
        # config = Dinov2Config.from_pretrained(self.model_name)
        # config.image_size = self.input_size
        
        # self.model = Dinov2Model(config)
        # logger.info(f"Model from config: {self.model.config}")
        # logger.info(f"Pretrained model : {Dinov2Model.from_pretrained(self.model_name).config}")
        self.model.to(self.device)
        
    def _run_model(self, images) -> torch.Tensor:
        """Extracts embeddings from the model."""
        inputs = self.image_processor(images, **self.im_proc_kwargs).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # ignore the CLS token (not classifying)
        # this way we get only the patch embeddings
        # which are compatible with finding the relevant patches directly
        # in the rest of the code
        return outputs.last_hidden_state[:, 1:, :]
    

@register_backbone("facebook/sam-vit-base", 256)
class SAMFeatures(FeatureExtractor):
    model_name = "facebook/sam-vit-base"

    def __init__(
        self, 
        image_size: tuple[int, int],
        save_path: str | Path,
        batch_size: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mode: PretrainedFeatsExtractionMode = "nearest_patch",
        ):
        super().__init__(image_size, save_path, batch_size, device, mode)
        self.input_size = 1024
        self.final_grid_size = 64  # 64x64 grid
        self.n_channels = 3
        self.hidden_state_size = 256
        self.image_processor = SamProcessor.from_pretrained(self.model_name)
        self.model = SamModel.from_pretrained(self.model_name)
        self.rescale_batches = False
        
        self.model.to(self.device)
        
    def _run_model(self, images) -> torch.Tensor:
        """Extracts embeddings from the model."""
        inputs = self.image_processor(images, **self.im_proc_kwargs).to(self.device)
        outputs = self.model.get_image_embeddings(inputs['pixel_values'])
        B, N, H, W = outputs.shape
        return outputs.permute(0, 2, 3, 1).reshape(B, H * W, N)  # (B, grid_size**2, hidden_state_size)
        

@register_backbone("facebook/sam2-hiera-large", 256)
@register_backbone("facebook/sam2.1-hiera-base-plus", 256)
class SAM2Features(FeatureExtractor):
    model_name = "facebook/sam2.1-hiera-base-plus"
    
    def __init__(
        self, 
        image_size: tuple[int, int],
        save_path: str | Path,
        batch_size: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mode: PretrainedFeatsExtractionMode = "nearest_patch",
        ):
        super().__init__(image_size, save_path, batch_size, device, mode)
        self.input_size = 1024
        self.final_grid_size = 64  # 64x64 grid
        self.n_channels = 3   
        self.hidden_state_size = 256        
        self.model = SAM2ImagePredictor.from_pretrained(self.model_name, device=self.device).model
        
        self.batch_return_type = "list[np.ndarray]"
        self.channel_first = True
        self.rescale_batches = False
        
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]
        if self.rescale_batches:
            print("Rescaling batches to [0, 255] range.")
        
    @torch.no_grad()
    def _run_model(self, images: list[np.ndarray]) -> torch.Tensor:
        """Extracts embeddings from the model."""
        with torch.autocast(device_type=self.device), torch.inference_mode():
            images_ten = torch.stack([torch.tensor(image) for image in images]).to(self.device)
            # logger.debug(f"Image dtype: {images_ten.dtype}")
            # logger.debug(f"Image shape: {images_ten.shape}")
            # logger.debug(f"Image min :  {images_ten.min()}, max: {images_ten.max()}")
            images_ten = F.interpolate(images_ten, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
            # from torchvision.transforms.functional import resize
            # images_ten = resize(images_ten, size=(self.input_size, self.input_size))
            # images_ten = normalize(images_ten, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            out = self.model.image_encoder(images_ten)
            _, vision_feats, _, _ = self.model._prepare_backbone_features(out)
            # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
            if self.model.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

            feats = [
                feat.permute(1, 2, 0).view(feat.shape[1], -1, *feat_size)
                for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
            ][::-1]
            features = feats[-1]
        # features = self.model.set_image_batch(images)
        # features = self.model._features['image_embed']
        B, N, H, W = features.shape
        return features.permute(0, 2, 3, 1).reshape(B, H * W, N)  # (B, grid_size**2, hidden_state_size)
    
    
@register_backbone("microsam/vit_b_lm", 256)
@register_backbone("microsam/vit_l_lm", 256)
class MicroSAMFeatures(FeatureExtractor):
    model_name = "microsam/vit_b_lm"
    
    def __init__(
        self, 
        image_size: tuple[int, int],
        save_path: str | Path,
        batch_size: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mode: PretrainedFeatsExtractionMode = "nearest_patch",
        ):
        if not MICRO_SAM_AVAILABLE:
            raise ImportError("microSAM is not available. Please install it following the instructions in the documentation.")
        super().__init__(image_size, save_path, batch_size, device, mode)
        self.input_size = 1024
        self.final_grid_size = 64
        self.n_channels = 3
        self.hidden_state_size = 256
        model_name = self.model_name.split("/")[-1]
        self.model = get_microsam_model(model_name, device=self.device)
        
        self.batch_return_type = "list[np.ndarray]"
        self.channel_first = False
        self.rescale_batches = False
        
    def _run_model(self, images: list[np.ndarray]) -> torch.Tensor:
        """Extracts embeddings from the model."""
        embeddings = []
        for image in images:
            # logger.debug(f"Image shape: {image.shape}")
            self.model.set_image(image)
            embedding = self.model.get_image_embedding()  # (1, hidden_state_size, grid_size, grid_size)
            B, N, H, W = embedding.shape
            embedding = embedding.permute(0, 2, 3, 1).reshape(B, H * W, N)
            embeddings.append(embedding)
        out = torch.stack(embeddings).squeeze()  # (B, grid_size**2, hidden_state_size)
        if len(out.shape) == 2:
            out = out.unsqueeze(0)
        return out


@register_backbone("random", 256)
class RandomFeatures(FeatureExtractor):
    model_name = "random"

    def __init__(
        self, 
        image_size: tuple[int, int],
        save_path: str | Path,
        batch_size: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mode: PretrainedFeatsExtractionMode = "nearest_patch",
        ):
        super().__init__(image_size, save_path, batch_size, device, mode)
        self.input_size = 1024
        self.final_grid_size = 64
        self.n_channels = 3
        self.hidden_state_size = 256
        
        self._seed = 42
        self._generator = torch.Generator(device=device).manual_seed(self._seed)
        
    def _run_model(self, images) -> torch.Tensor:
        """Extracts embeddings from the model."""
        # Normal distribution
        # return torch.randn(len(images), self.final_grid_size**2, self.hidden_state_size, generator=self._generator).to(self.device)
        # Uniform distribution
        feats = torch.rand(len(images), self.final_grid_size**2, self.hidden_state_size, generator=self._generator).to(self.device)
        feats = feats * 4 - 2  # [-2, 2]
        return feats


FeatureExtractor._available_backbones = AVAILABLE_PRETRAINED_BACKBONES


# Embeddings post-processing

import pickle

from sklearn.decomposition import PCA


class EmbeddingsPCACompression:
    
    def __init__(self, original_model_name: str, n_components: int = 15, save_path: str | Path | None = None):
        self.original_model_name = original_model_name
        self.n_components = n_components
        self.save_path = save_path / "pca_model.pkl" if save_path is not None else None
        self.pca = PCA(n_components=n_components)
        self.max_frames = 1500
        self.random_state = 42
        self.pca.random_state = self.random_state
        self.generator = np.random.default_rng(self.random_state)
    
    @classmethod
    def from_pretrained_cfg(cls, cfg: PretrainedFeatureExtractorConfig):
        return cls(
            original_model_name=cfg.model_name.replace("/", "-"),
            n_components=cfg.pca_components,
            save_path=cfg.pca_preprocessor_path
        )
    
    def fit(self, X: np.ndarray):
        """Fits the PCA model to the embeddings."""
        self.pca.fit(X)
        
        if self.save_path is not None:
            if not self.save_path.parents[0].exists():
                self.save_path.parents[0].mkdir(parents=True, exist_ok=True)
            with open(self.save_path, 'wb') as f:
                pickle.dump(self.pca, f)
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms the embeddings using the fitted PCA model."""
        return self.pca.transform(X)
    
    def load_from_file(self, path: str | Path):
        """Loads the PCA model from a file."""
        if isinstance(path, str):
            path = Path(path)
        path = path / "pca_model.pkl" if path.suffix != ".pkl" else path
        with open(path, 'rb') as f:
            self.pca = pickle.load(f)
        logger.info(f"Loaded PCA model from {path}.")
    
    def fit_on_embeddings(self, embeddings_source_folders: list[str | Path]):
        """Fits the PCA model to the embeddings loaded from a file."""
        embeddings = []
        N_samples = 0
        
        embeddings_paths = []
        for folder in embeddings_source_folders:
            for file in Path(folder).rglob("*.npy"):
                if self.original_model_name in file.name:
                    embeddings_paths.append(file)
        
        if len(embeddings_paths) == 0:
            return
        
        embeddings_paths = self.generator.permutation(embeddings_paths)
        logger.info(f"Fitting PCA model on {len(embeddings_paths)} embeddings files.")
        logger.info("Files :")
        for p in embeddings_paths:
            logger.info(f" - {p}")
        logger.info("*" * 50)
        
        for path in embeddings_paths:
            if isinstance(path, str):
                path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"File {path} does not exist.")
            emb = np.load(path)
            N_samples += emb.shape[0]
            if N_samples > self.max_frames:
                logger.info(f"Amount of loaded frames exceeds {self.max_frames} limit for PCA computation.")
                break
            else:
                embeddings.append(emb)
        
        embeddings = np.concatenate(embeddings, axis=0)
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        self.fit(embeddings)
        logger.info(f"Fitted PCA model with {self.n_components} components on {N_samples} frames.")
        