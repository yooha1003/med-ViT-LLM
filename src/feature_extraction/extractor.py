"""
Feature Extractor for Medical Images
Extracts volumetric, morphological, and intensity-based features from segmented regions
"""

import numpy as np
import SimpleITK as sitk
from typing import Dict, Union, Optional, Tuple
from pathlib import Path
import logging
import nibabel as nib
from scipy import ndimage

try:
    from radiomics import featureextractor
    RADIOMICS_AVAILABLE = True
except ImportError:
    RADIOMICS_AVAILABLE = False
    logging.warning("PyRadiomics not available. Advanced texture features will be disabled.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract quantitative features from segmented MRI regions

    Features include:
    - Volumetric: volume, surface area
    - Morphological: sphericity, elongation, flatness, compactness
    - Intensity-based: mean, std, min, max, entropy
    - Texture: (if PyRadiomics is available) GLCM, GLRLM, etc.
    """

    def __init__(self, use_radiomics: bool = True, radiomics_params: Optional[Dict] = None):
        """
        Initialize Feature Extractor

        Args:
            use_radiomics: Whether to use PyRadiomics for texture features
            radiomics_params: Custom parameters for PyRadiomics
        """
        self.use_radiomics = use_radiomics and RADIOMICS_AVAILABLE

        if self.use_radiomics:
            # Initialize PyRadiomics feature extractor
            if radiomics_params is None:
                radiomics_params = {
                    'binWidth': 25,
                    'resampledPixelSpacing': None,
                    'interpolator': sitk.sitkBSpline,
                    'verbose': False
                }
            self.radiomics_extractor = featureextractor.RadiomicsFeatureExtractor(**radiomics_params)
            logger.info("PyRadiomics feature extractor initialized")
        else:
            self.radiomics_extractor = None

    def extract_volumetric_features(
        self,
        mask: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> Dict[str, float]:
        """
        Extract volumetric features

        Args:
            mask: Binary segmentation mask
            spacing: Voxel spacing (x, y, z) in mm

        Returns:
            Dictionary of volumetric features
        """
        features = {}

        # Calculate voxel volume
        voxel_volume = np.prod(spacing)

        # Volume
        num_voxels = np.sum(mask > 0)
        volume_mm3 = num_voxels * voxel_volume
        features['volume_mm3'] = float(volume_mm3)
        features['volume_cm3'] = float(volume_mm3 / 1000)

        # Surface area (approximate using edge detection)
        edges = ndimage.sobel(mask.astype(float))
        surface_voxels = np.sum(edges > 0)
        voxel_surface = 2 * (spacing[0] * spacing[1] + spacing[1] * spacing[2] + spacing[0] * spacing[2])
        features['surface_area_mm2'] = float(surface_voxels * voxel_surface)

        logger.info(f"Volume: {features['volume_cm3']:.2f} cm³")

        return features

    def extract_morphological_features(
        self,
        mask: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> Dict[str, float]:
        """
        Extract morphological features

        Args:
            mask: Binary segmentation mask
            spacing: Voxel spacing (x, y, z) in mm

        Returns:
            Dictionary of morphological features
        """
        features = {}

        # Get volumetric features first
        vol_features = self.extract_volumetric_features(mask, spacing)
        volume = vol_features['volume_mm3']
        surface_area = vol_features['surface_area_mm2']

        # Sphericity: how sphere-like the shape is
        # Sphericity = (π^(1/3) * (6*Volume)^(2/3)) / Surface_Area
        if surface_area > 0:
            sphericity = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / surface_area
            features['sphericity'] = float(min(sphericity, 1.0))  # Cap at 1.0
        else:
            features['sphericity'] = 0.0

        # Compactness: Volume / (Surface_Area^(3/2))
        if surface_area > 0:
            features['compactness'] = float(volume / (surface_area ** 1.5))
        else:
            features['compactness'] = 0.0

        # Calculate moments for orientation and elongation
        coords = np.argwhere(mask > 0)
        if len(coords) > 0:
            # Center of mass
            center = coords.mean(axis=0)
            features['center_x'] = float(center[0] * spacing[0])
            features['center_y'] = float(center[1] * spacing[1])
            features['center_z'] = float(center[2] * spacing[2])

            # Covariance matrix for principal axes
            coords_centered = coords - center
            coords_scaled = coords_centered * spacing
            cov_matrix = np.cov(coords_scaled.T)

            # Eigenvalues represent the variance along principal axes
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending

            if eigenvalues[0] > 0:
                # Elongation: ratio of second to first principal axis
                features['elongation'] = float(np.sqrt(eigenvalues[1] / eigenvalues[0]))

                # Flatness: ratio of third to first principal axis
                features['flatness'] = float(np.sqrt(eigenvalues[2] / eigenvalues[0]))
            else:
                features['elongation'] = 0.0
                features['flatness'] = 0.0

            # Bounding box dimensions
            min_coords = coords.min(axis=0) * spacing
            max_coords = coords.max(axis=0) * spacing
            bbox_dims = max_coords - min_coords
            features['bbox_x'] = float(bbox_dims[0])
            features['bbox_y'] = float(bbox_dims[1])
            features['bbox_z'] = float(bbox_dims[2])

        return features

    def extract_intensity_features(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract intensity-based features from the region

        Args:
            image: Original MRI image
            mask: Binary segmentation mask

        Returns:
            Dictionary of intensity features
        """
        features = {}

        # Get intensities within the mask
        intensities = image[mask > 0]

        if len(intensities) > 0:
            features['intensity_mean'] = float(np.mean(intensities))
            features['intensity_std'] = float(np.std(intensities))
            features['intensity_min'] = float(np.min(intensities))
            features['intensity_max'] = float(np.max(intensities))
            features['intensity_median'] = float(np.median(intensities))

            # Percentiles
            features['intensity_p10'] = float(np.percentile(intensities, 10))
            features['intensity_p90'] = float(np.percentile(intensities, 90))

            # Entropy
            hist, _ = np.histogram(intensities, bins=50)
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            features['intensity_entropy'] = float(-np.sum(hist * np.log2(hist)))

            # Coefficient of variation
            if features['intensity_mean'] > 0:
                features['intensity_cv'] = float(features['intensity_std'] / features['intensity_mean'])
            else:
                features['intensity_cv'] = 0.0

        return features

    def extract_radiomics_features(
        self,
        image_path: Union[str, Path],
        mask_path: Union[str, Path]
    ) -> Dict[str, float]:
        """
        Extract texture features using PyRadiomics

        Args:
            image_path: Path to original MRI image
            mask_path: Path to segmentation mask

        Returns:
            Dictionary of radiomics features
        """
        if not self.use_radiomics:
            logger.warning("PyRadiomics not available or disabled")
            return {}

        features = {}

        try:
            # Extract features using PyRadiomics
            result = self.radiomics_extractor.execute(str(image_path), str(mask_path))

            # Filter and convert to regular dict
            for key, value in result.items():
                if not key.startswith('diagnostics_'):
                    try:
                        features[key] = float(value)
                    except (ValueError, TypeError):
                        pass

            logger.info(f"Extracted {len(features)} radiomics features")

        except Exception as e:
            logger.error(f"Error extracting radiomics features: {e}")

        return features

    def extract_all_features(
        self,
        image: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        image_path: Optional[Union[str, Path]] = None,
        mask_path: Optional[Union[str, Path]] = None,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> Dict[str, float]:
        """
        Extract all available features

        Args:
            image: Original MRI image array (optional)
            mask: Segmentation mask array (optional)
            image_path: Path to MRI image file (optional)
            mask_path: Path to mask file (optional)
            spacing: Voxel spacing

        Returns:
            Dictionary containing all extracted features
        """
        all_features = {}

        # Load from files if paths provided
        if image_path is not None and image is None:
            nifti_img = nib.load(str(image_path))
            image = nifti_img.get_fdata()
            # Get spacing from NIfTI header
            spacing = tuple(nifti_img.header.get_zooms()[:3])

        if mask_path is not None and mask is None:
            nifti_mask = nib.load(str(mask_path))
            mask = nifti_mask.get_fdata()

        if mask is None:
            raise ValueError("Either mask or mask_path must be provided")

        # Ensure binary mask
        mask = (mask > 0).astype(np.uint8)

        # Extract volumetric and morphological features
        logger.info("Extracting volumetric features...")
        vol_features = self.extract_volumetric_features(mask, spacing)
        all_features.update(vol_features)

        logger.info("Extracting morphological features...")
        morph_features = self.extract_morphological_features(mask, spacing)
        all_features.update(morph_features)

        # Extract intensity features if image provided
        if image is not None:
            logger.info("Extracting intensity features...")
            intensity_features = self.extract_intensity_features(image, mask)
            all_features.update(intensity_features)

        # Extract radiomics features if paths provided
        if self.use_radiomics and image_path is not None and mask_path is not None:
            logger.info("Extracting radiomics texture features...")
            radiomics_features = self.extract_radiomics_features(image_path, mask_path)
            all_features.update(radiomics_features)

        logger.info(f"Total features extracted: {len(all_features)}")

        return all_features

    def get_feature_summary(self, features: Dict[str, float]) -> str:
        """
        Generate human-readable summary of key features

        Args:
            features: Dictionary of extracted features

        Returns:
            Formatted string summary
        """
        summary = "=== Feature Summary ===\n\n"

        # Volumetric
        if 'volume_cm3' in features:
            summary += f"Volume: {features['volume_cm3']:.2f} cm³\n"
            summary += f"Surface Area: {features.get('surface_area_mm2', 0):.2f} mm²\n\n"

        # Morphological
        summary += "Morphological Features:\n"
        if 'sphericity' in features:
            summary += f"  Sphericity: {features['sphericity']:.3f}\n"
        if 'elongation' in features:
            summary += f"  Elongation: {features['elongation']:.3f}\n"
        if 'flatness' in features:
            summary += f"  Flatness: {features['flatness']:.3f}\n"
        summary += "\n"

        # Intensity
        if 'intensity_mean' in features:
            summary += "Intensity Features:\n"
            summary += f"  Mean: {features['intensity_mean']:.2f}\n"
            summary += f"  Std: {features['intensity_std']:.2f}\n"
            summary += f"  Range: [{features['intensity_min']:.2f}, {features['intensity_max']:.2f}]\n"

        return summary
