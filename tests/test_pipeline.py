"""
Unit tests for MRI diagnosis pipeline
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from segmentation.unet_3d import UNet3D
from feature_extraction.extractor import FeatureExtractor
from interpretation.llm_interpreter import SimpleLLMInterpreter


class TestSegmentation:
    """Test segmentation module"""

    def test_unet3d_initialization(self):
        """Test UNet3D model initialization"""
        model = UNet3D(in_channels=1, num_classes=2, base_features=32)
        assert model is not None
        assert model.in_channels == 1
        assert model.num_classes == 2

    def test_unet3d_forward(self):
        """Test UNet3D forward pass"""
        model = UNet3D(in_channels=1, num_classes=2)
        model.eval()

        # Create dummy input
        x = np.random.randn(1, 1, 64, 64, 64).astype(np.float32)
        import torch
        x_tensor = torch.from_numpy(x)

        # Forward pass
        output = model(x_tensor)

        assert output.shape[0] == 1  # batch size
        assert output.shape[1] == 2  # num classes
        assert output.shape[2:] == (64, 64, 64)  # spatial dims


class TestFeatureExtraction:
    """Test feature extraction module"""

    def test_feature_extractor_initialization(self):
        """Test FeatureExtractor initialization"""
        extractor = FeatureExtractor(use_radiomics=False)
        assert extractor is not None

    def test_volumetric_features(self):
        """Test volumetric feature extraction"""
        extractor = FeatureExtractor(use_radiomics=False)

        # Create dummy mask (sphere)
        mask = np.zeros((64, 64, 64), dtype=np.uint8)
        center = 32
        radius = 10
        for i in range(64):
            for j in range(64):
                for k in range(64):
                    if (i-center)**2 + (j-center)**2 + (k-center)**2 <= radius**2:
                        mask[i, j, k] = 1

        # Extract features
        features = extractor.extract_volumetric_features(mask, spacing=(1.0, 1.0, 1.0))

        assert 'volume_mm3' in features
        assert 'volume_cm3' in features
        assert 'surface_area_mm2' in features
        assert features['volume_mm3'] > 0

    def test_morphological_features(self):
        """Test morphological feature extraction"""
        extractor = FeatureExtractor(use_radiomics=False)

        # Create dummy mask
        mask = np.zeros((64, 64, 64), dtype=np.uint8)
        mask[20:40, 20:40, 20:40] = 1

        # Extract features
        features = extractor.extract_morphological_features(mask)

        assert 'sphericity' in features
        assert 'elongation' in features
        assert 'flatness' in features
        assert 0 <= features['sphericity'] <= 1

    def test_intensity_features(self):
        """Test intensity feature extraction"""
        extractor = FeatureExtractor(use_radiomics=False)

        # Create dummy image and mask
        image = np.random.randn(64, 64, 64) * 100 + 500
        mask = np.zeros((64, 64, 64), dtype=np.uint8)
        mask[20:40, 20:40, 20:40] = 1

        # Extract features
        features = extractor.extract_intensity_features(image, mask)

        assert 'intensity_mean' in features
        assert 'intensity_std' in features
        assert 'intensity_min' in features
        assert 'intensity_max' in features
        assert 'intensity_entropy' in features


class TestInterpretation:
    """Test interpretation module"""

    def test_simple_interpreter_initialization(self):
        """Test SimpleLLMInterpreter initialization"""
        interpreter = SimpleLLMInterpreter()
        assert interpreter is not None

    def test_generate_interpretation(self):
        """Test interpretation generation"""
        interpreter = SimpleLLMInterpreter()

        # Dummy features
        features = {
            'volume_cm3': 2.5,
            'sphericity': 0.75,
            'intensity_mean': 145.3
        }

        # Generate interpretation
        interpretation = interpreter.generate_interpretation(
            features=features,
            structure_name="Hippocampus",
            patient_info={'age': 65, 'sex': 'Male'}
        )

        assert interpretation is not None
        assert len(interpretation) > 0
        assert isinstance(interpretation, str)

    def test_generate_report(self):
        """Test report generation"""
        interpreter = SimpleLLMInterpreter()

        features = {
            'volume_cm3': 2.5,
            'sphericity': 0.75
        }

        report = interpreter.generate_report(
            features=features,
            structure_name="Hippocampus"
        )

        assert 'patient_info' in report
        assert 'structure' in report
        assert 'interpretation' in report

    def test_format_report(self):
        """Test report formatting"""
        interpreter = SimpleLLMInterpreter()

        report = {
            'patient_info': {'age': 65},
            'structure': 'Hippocampus',
            'interpretation': 'Test interpretation'
        }

        formatted = interpreter.format_report(report)

        assert isinstance(formatted, str)
        assert len(formatted) > 0
        assert 'Hippocampus' in formatted


class TestUtilities:
    """Test utility functions"""

    def test_config_loader(self):
        """Test configuration loading"""
        from utils.config_loader import load_config

        # Load default config
        config = load_config()

        assert config is not None
        assert 'segmentation' in config
        assert 'feature_extraction' in config
        assert 'interpretation' in config


def test_end_to_end_dummy():
    """Test end-to-end pipeline with dummy data"""
    # This is a placeholder for integration testing
    # In practice, you would use real or synthetic MRI data

    # Create dummy volume
    volume = np.random.randn(64, 64, 64).astype(np.float32)

    # Create dummy mask
    mask = np.zeros((64, 64, 64), dtype=np.uint8)
    mask[20:40, 20:40, 20:40] = 1

    # Extract features
    extractor = FeatureExtractor(use_radiomics=False)
    features = extractor.extract_morphological_features(mask)

    # Generate interpretation
    interpreter = SimpleLLMInterpreter()
    interpretation = interpreter.generate_interpretation(
        features=features,
        structure_name="Test Structure"
    )

    # Verify results
    assert features is not None
    assert interpretation is not None
    assert len(features) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
