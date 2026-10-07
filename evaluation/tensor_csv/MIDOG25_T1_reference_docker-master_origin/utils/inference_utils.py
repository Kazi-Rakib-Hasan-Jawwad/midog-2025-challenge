from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import logging
import SimpleITK as sitk
import yaml

from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.ops.boxes import nms as torch_nms
from tqdm.autonotebook import tqdm

from utils.model import make_fcos_model


Coords = Tuple[int, int]
ImageType = Union[np.ndarray, torch.Tensor]


@dataclass(kw_only=True)
class Config:
    @classmethod
    def load(cls, filepath: str):
        """Load configuration from a YAML file.

        Args:
            filepath (str): Path to the configuration file

        Returns:
            Config: Loaded configuration object
        """
        with open(filepath, 'r') as file:
            config_dict = yaml.load(file, Loader=yaml.SafeLoader)
        return cls(**config_dict)

@dataclass
class PatchConfig(Config):
    """Configuration for patch extraction parameters.

    Args:
        size (int): Size of patches (assumed square)
        overlap (float): Overlap between adjacent patches (0-1)
        level (int): Pyramid level for WSI
    """
    size: int = 1024
    overlap: float = 0.3
    level: int = 0

@dataclass
class InferenceConfig(Config):
    """Configuration for inference parameters.

    Args:
        batch_size (int): Number of patches to process simultaneously
        num_workers (int): Number of worker processes for data loading
        device: Device to run inference on ('cuda' or 'cpu')
        nms_thresh: Non-maximum suppression threshold
        score_thresh: Minimum confidence score for detections
    """
    batch_size: int = 8
    num_workers: int = 4
    device: str = 'cuda'
    nms_thresh: float = 0.3
    score_thresh: float = 0.5

@dataclass
class ModelConfig(Config):
    """Configuration for the model parameters.

    Args:
        backbone (str): ResNet backbone of the detector
        checkpoint (str): Path to model checkpoint
        det_thresh (float): Detection threshold
        patch_size (int): Size of patches (assumed square)
    """
    backbone: str 
    checkpoint: str 
    det_thresh: float
    patch_size: int 


class BaseInferenceDataset(Dataset, ABC):
    """Base class for inference datasets handling patch-based processing.

    Args:
        patch_config: Configuration for patch extraction
        transforms: Optional transforms to apply to patches
    """
    def __init__(
        self,
        patch_config: PatchConfig,
        transforms: Optional[Union[List[Callable], Callable]] = None,
    ) -> None:
        self.config = patch_config
        self.transforms = transforms

        # To be set by child classes
        self.coords: List[Coords] = []
        self.image_size: Tuple[int, int] = (0, 0)


    @abstractmethod
    def _load_image(self) -> None:
        """Load the image/slide and set necessary attributes."""
        pass

    @abstractmethod
    def _get_patch(self, coords: Coords) -> ImageType:
        """Extract a patch from the image at given coordinates."""
        pass

    def _normalize_patch(self, patch: ImageType) -> torch.Tensor:
        """Normalize patch and convert to tensor."""
        if isinstance(patch, np.ndarray):
            patch = torch.from_numpy(patch / 255.).permute(2, 0, 1).float()
        return patch

    def _get_coords(self) -> List[Coords]:
        """Generate patch coordinates based on image size and overlap."""
        width, height = self.image_size
        stride = int(self.config.size * (1 - self.config.overlap))

        coords = []
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                # Adjust coordinates to prevent going out of bounds
                x_adj = min(x, width - self.config.size)
                y_adj = min(y, height - self.config.size)
                coords.append((x_adj, y_adj))

        return coords

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        x, y = self.coords[idx]
        patch = self._get_patch((x, y))

        if self.transforms is not None:
            patch = self.transforms(patch)

        patch = self._normalize_patch(patch)
        return patch, x, y

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, int, int]]) -> Tuple[List[torch.Tensor], List[int], List[int]]:
        """Custom collate function for batching."""
        patches, x_coords, y_coords = zip(*batch)
        return list(patches), list(x_coords), list(y_coords)


class ROI_InferenceDataset(BaseInferenceDataset):
    """Dataset for regular image inference."""

    def __init__(
        self,
        image: sitk.Image,
        patch_config: Optional[PatchConfig] = None,
        transforms: Optional[Union[List[Callable], Callable]] = None
    ) -> None:
        self.image = image 
        patch_config = patch_config or PatchConfig()
        super().__init__(patch_config, transforms)

        self._load_image()
        self.coords = self._get_coords()

    def _load_image(self) -> None:
        """Load image and set image size."""
        # Convert to numpy array
        self.image_array = sitk.GetArrayFromImage(self.image)

        # Set image size
        self.image_size = (self.image_array.shape[1], self.image_array.shape[0])

    def _get_patch(self, coords: Coords) -> np.ndarray:
        """Extract patch from numpy array."""
        x, y = coords
        patch = self.image_array[y:y + self.config.size, x:x + self.config.size, :3]
        return patch
    



    
class Strategy(ABC):
    """Abstract base class defining the interface for inference strategies.

    This class serves as a template for implementing different inference strategies
    for processing images with deep learning models.
    """

    @abstractmethod
    def process_image(self, model: nn.Module, image: str, **kwargs) -> Dict[str, np.ndarray]:
        """Process an image using the specified model.

        Args:
            model (nn.Module): The neural network model to use for inference
            image (str): Path to the image file
            **kwargs: Additional keyword arguments for processing

        Returns:
            Dict[str, np.ndarray]: Dictionary containing inference results for boxes, labels, scores.
        """
        pass


class Torchvision_Inference(Strategy):
    """Inference strategy for Torchvision-based object detection models.

    This class handles patch-based inference for histopathology crop out regions,
    with support for various detection models (Faster R-CNN, Mask R-CNN, FCOS, etc.).

    Args:
        model: The detection model to use
        config: Inference configuration parameters
        logger: Optional logger instance
    """
    def __init__(
        self,
        model: nn.Module,
        inference_config: Optional[InferenceConfig] = None,
        patch_config: Optional[PatchConfig] = None,
        logger: Optional[logging.Logger] = None
    ) -> None:
        self.model = model
        self.inference_config = inference_config or InferenceConfig()
        self.patch_config = patch_config or PatchConfig()
        self.logger = logger or self._setup_logger()
        self.device = self._setup_device()

    def _setup_logger(self) -> logging.Logger:
        """Initialize logger with appropriate configuration."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _setup_device(self) -> torch.device:
        """Set up and validate the processing device."""
        if self.inference_config.device == 'cuda' and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available. Using CPU instead.")
            return torch.device('cpu')
        return torch.device(self.inference_config.device)

    def _create_dataloader(
        self,
        image: sitk.Image,
        transforms: Optional[Union[List[Callable], Callable]] = None
    ) -> DataLoader:
        """Creates the dataloader."""
        try:
            dataset = ROI_InferenceDataset(
                image=image,
                patch_config=self.patch_config,
                transforms=transforms
            )
        except Exception as e:
            self.logger.error(f"Failed to create dataset: {str(e)}")
            raise

        return DataLoader(
            dataset,
            batch_size=self.inference_config.batch_size,
            num_workers=self.inference_config.num_workers,
            collate_fn=dataset.collate_fn,
        )

    @torch.no_grad()
    def _process_batch(
        self,
        batch: List[torch.Tensor]
    ) -> List[Dict[str, torch.Tensor]]:
        """Process a batch of patches."""
        images = [img.to(self.device) for img in batch]
        try:
            predictions = self.model(images)
            return predictions
        except RuntimeError as e:
            self.logger.error(f"Error during forward pass: {str(e)}")
            raise

    def _post_process_predictions(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        coords: List[Coords]
    ) -> Dict[str, torch.Tensor]:
        """Post-process predictions including coordinate adjustment and NMS."""
        boxes_list = []
        scores_list = []
        labels_list = []

        for pred, (x_orig, y_orig) in zip(predictions, coords):
            if len(pred['boxes']) > 0:
                # Adjust coordinates to original image space
                boxes = pred['boxes'] + torch.tensor(
                    [x_orig, y_orig, x_orig, y_orig],
                    device=pred['boxes'].device
                )               

                boxes_list.append(boxes)
                scores_list.append(pred['scores'])
                labels_list.append(pred['labels'])

        if not boxes_list:
            return {
                'boxes': torch.empty((0, 4), device=self.device),
                'scores': torch.empty(0, device=self.device),
                'labels': torch.empty(0, device=self.device)
            }

        # Concatenate all predictions
        boxes = torch.cat(boxes_list)
        scores = torch.cat(scores_list)
        labels = torch.cat(labels_list)

        # Apply NMS per class
        final_boxes = []
        final_scores = []
        final_labels = []

        for label in labels.unique():
            mask = labels == label
            class_boxes = boxes[mask]
            class_scores = scores[mask]

            keep = torch_nms(class_boxes, class_scores, self.inference_config.nms_thresh)

            final_boxes.append(class_boxes[keep])
            final_scores.append(class_scores[keep])
            final_labels.append(labels[mask][keep])

        # Concatenate 
        final_boxes = torch.cat(final_boxes)
        final_scores = torch.cat(final_scores)
        final_labels = torch.cat(final_labels)

        return {
            'boxes': final_boxes,
            'scores': final_scores,
            'labels': final_labels
        }

    def process_image(
        self,
        image: sitk.Image,
    ) -> Dict[str, np.ndarray]:
        """Process an image using patch-based inference.

        Args:
            image: SITK Image

        Returns:
            Dict containing 'boxes', 'scores', and 'labels' as numpy arrays
        """
        # Prepare model
        self.model.eval()
        self.model.to(self.device)

        # Create dataloader
        dataloader = self._create_dataloader(image)

        # Initialize results storage
        all_predictions = []
        all_coords = []

        # Process batches
        with tqdm(dataloader, desc="Processing batches") as pbar:
            for batch_images, batch_x, batch_y in pbar:
                predictions = self._process_batch(batch_images)
                all_predictions.extend(predictions)
                all_coords.extend(zip(batch_x, batch_y))

        # Post-process results
        results = self._post_process_predictions(all_predictions, all_coords)

        # Convert to numpy arrays
        return {
            'boxes': results['boxes'].cpu().numpy(),
            'scores': results['scores'].cpu().numpy(),
            'labels': results['labels'].cpu().numpy()
        }
    

class DetectionAlgorithm:
    def __init__(
            self,
            model_config: str,
            patch_config: str, 
            inference_config: str,
            logger: Optional[logging.Logger] = None
            ):
        
        # Setup logger first
        self.logger = logger or self._setup_logger()
        
        self.logger.info("Initializing DetectionAlgorithm...")
        
        # Validate config file paths
        self._validate_config_paths(model_config, patch_config, inference_config)
        
        # Read config files 
        try:
            self.model_config = ModelConfig.load(model_config)
            self.patch_config = PatchConfig.load(patch_config)
            self.inference_config = InferenceConfig.load(inference_config)
            
            self.logger.info("All configuration files loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load configuration files: {str(e)}")
            raise

        # Validate model checkpoint exists
        self._validate_checkpoint()

        # Load model 
        self.logger.info("Loading detection model...")
        self.model = self.load_model()

        # Load inference strategy 
        self.logger.info("Initializing inference strategy...")
        self.strategy = Torchvision_Inference(
            self.model, 
            self.inference_config, 
            self.patch_config,
            self.logger
        )
        
        self.logger.info("DetectionAlgorithm initialization completed successfully")

    def _setup_logger(self) -> logging.Logger:
        """Initialize logger with appropriate configuration."""
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _validate_config_paths(self, model_config: str, patch_config: str, inference_config: str) -> None:
        """Validate that all config file paths exist."""
        config_files = {
            'model_config': model_config,
            'patch_config': patch_config,
            'inference_config': inference_config
        }
        
        for config_name, config_path in config_files.items():
            if not Path(config_path).exists():
                error_msg = f"{config_name} file not found: {config_path}"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)

    def _validate_checkpoint(self) -> None:
        """Validate that the model checkpoint exists."""
        checkpoint_path = Path(self.model_config.checkpoint)
        if not checkpoint_path.exists():
            error_msg = f"Model checkpoint not found: {checkpoint_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        self.logger.info(f"Model checkpoint found: {checkpoint_path}")

    def load_model(self):
        """Initialize detection model and load checkpoint."""
        try:
            self.logger.info(f"Initializing FCOS model with backbone: {self.model_config.backbone}")
            
            # Initialize model 
            model = make_fcos_model(
                backbone=self.model_config.backbone,
                det_thresh=0.2,  # we set low threshold for AP calculation
                patch_size=self.model_config.patch_size,
                weights=None
            )
            
            self.logger.info("Model architecture created successfully")
            
            # Load checkpoint (we extract weights from a lightning checkpoint)
            self.logger.info(f"Loading checkpoint from: {self.model_config.checkpoint}")
            checkpoint = torch.load(self.model_config.checkpoint, map_location=self.inference_config.device)
            model_state_dict = checkpoint['state_dict']
            cleaned_state_dict = {}
            for key, value in model_state_dict.items():
                if key.startswith('model.'):
                    cleaned_state_dict[key[6:]] = value  # remove 'model.' prefix
                else:
                    cleaned_state_dict[key] = value

            # Load into the model
            model.load_state_dict(cleaned_state_dict, strict=False)
            self.logger.info("Model weights loaded successfully")
        
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def predict(self, image: sitk.Image):
        """Wraps the process_image call of the inference strategy."""
        try:            
            self.logger.info("Running inference...")
            
            # Get predictions
            results = self.strategy.process_image(image)
            boxes = results['boxes']
            scores = results['scores']
            labels = results['labels']
            
            self.logger.info(f"Inference completed - Found {len(boxes)} detections")

            # Initialize output
            candidates = []

            # Classnames for saving results 
            classnames = ['non-mitotic figure', 'mitotic figure']

            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                # Boxes are returned in x1, y1, x2, y2 format 
                # we need to transform them back to center coordinates
                x1, y1, x2, y2 = box
                coord = ((x1 + x2) / 2, (y1 + y2) / 2)

                # For the test set, we expect the coordinates in millimeters - this transformation ensures that the pixel
                # coordinates are transformed to mm - if resolution information is available in the .tiff image. If not,
                # pixel coordinates are returned.
                try:
                    world_coords = image.TransformContinuousIndexToPhysicalPoint(
                        [c for c in coord]
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to transform coordinates to physical space: {e}")
                    world_coords = coord

                # Expected syntax from evaluation container is:
                # x-coordinate(centroid), y-coordinate(centroid), 0, label, score
                # where label should be 1 if score is above threshold and 0 else
                candidates.append([
                    *tuple(world_coords), 0,
                    int(score > self.model_config.det_thresh), 
                    score
                ])

            # Create points and convert to native types
            # If there are no detections in an image, points should be an empty list!
            points = [
                {
                    "name": classnames[int(c[3])],
                    "point": [float(c[0]), float(c[1]), float(c[2])],
                    "probability": float(c[4])
                } for c in candidates
            ]

            # Create final output format
            output_mitotic_figures = {
                "name": "Points of interest",
                "type": "Multiple points",
                "points": points,
                "version": {"major": 1, "minor": 0},
            }

            self.logger.info("Prediction completed successfully")
            return output_mitotic_figures
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise