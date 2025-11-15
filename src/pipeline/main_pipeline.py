"""
End-to-End MRI Diagnosis Pipeline
Integrates segmentation, feature extraction, and LLM-based interpretation
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union, Tuple
import numpy as np
import nibabel as nib
import json
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from segmentation.model import SegmentationModel
from feature_extraction.extractor import FeatureExtractor
from interpretation.llm_interpreter import LLMInterpreter, SimpleLLMInterpreter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MRIDiagnosisPipeline:
    """
    Complete MRI diagnosis pipeline: Image → Segmentation → Features → Interpretation

    This pipeline:
    1. Loads MRI image
    2. Performs automated segmentation
    3. Extracts quantitative features
    4. Generates clinical interpretation using LLM
    """

    def __init__(
        self,
        segmentation_model_type: str = "unet3d",
        segmentation_checkpoint: Optional[str] = None,
        llm_model_name: str = "microsoft/phi-2",
        use_gpu: bool = True,
        use_radiomics: bool = True,
        use_llm: bool = True,
        device: str = None
    ):
        """
        Initialize the diagnosis pipeline

        Args:
            segmentation_model_type: Type of segmentation model
            segmentation_checkpoint: Path to segmentation model weights
            llm_model_name: Name of LLM for interpretation
            use_gpu: Whether to use GPU
            use_radiomics: Whether to use PyRadiomics for texture features
            use_llm: Whether to use LLM (otherwise use rule-based interpretation)
            device: Specific device to use
        """
        logger.info("Initializing MRI Diagnosis Pipeline...")

        # Initialize segmentation model
        logger.info("Loading segmentation model...")
        self.segmentation_model = SegmentationModel(
            model_type=segmentation_model_type,
            checkpoint_path=segmentation_checkpoint,
            device=device
        )

        # Initialize feature extractor
        logger.info("Initializing feature extractor...")
        self.feature_extractor = FeatureExtractor(use_radiomics=use_radiomics)

        # Initialize interpreter
        logger.info("Initializing clinical interpreter...")
        if use_llm:
            try:
                self.interpreter = LLMInterpreter(
                    model_name=llm_model_name,
                    use_gpu=use_gpu,
                    device=device
                )
            except Exception as e:
                logger.warning(f"Failed to load LLM: {e}. Using rule-based interpreter instead.")
                self.interpreter = SimpleLLMInterpreter()
        else:
            self.interpreter = SimpleLLMInterpreter()

        logger.info("Pipeline initialization complete!")

    def run(
        self,
        image_path: Union[str, Path],
        structure_name: str = "brain region",
        patient_info: Optional[Dict] = None,
        clinical_context: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        save_segmentation: bool = True,
        save_features: bool = True,
        save_report: bool = True
    ) -> Dict:
        """
        Run complete diagnosis pipeline on an MRI image

        Args:
            image_path: Path to MRI image (NIfTI format)
            structure_name: Name of anatomical structure being analyzed
            patient_info: Patient metadata (age, sex, symptoms, etc.)
            clinical_context: Additional clinical context
            output_dir: Directory to save outputs
            save_segmentation: Whether to save segmentation mask
            save_features: Whether to save extracted features
            save_report: Whether to save clinical report

        Returns:
            Dictionary containing:
                - segmentation_mask: numpy array
                - features: dictionary of features
                - interpretation: clinical interpretation text
                - report: structured report
        """
        logger.info(f"Processing: {image_path}")
        logger.info(f"Structure: {structure_name}")

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Create output directory if specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory: {output_dir}")

        results = {}

        # Step 1: Segmentation
        logger.info("=" * 60)
        logger.info("Step 1: Performing segmentation...")
        logger.info("=" * 60)

        mask, probabilities = self.segmentation_model.segment(image_path=image_path)
        results['segmentation_mask'] = mask
        results['segmentation_probabilities'] = probabilities

        logger.info(f"Segmentation completed. Mask shape: {mask.shape}")
        logger.info(f"Unique labels: {np.unique(mask)}")

        # Save segmentation if requested
        if save_segmentation and output_dir:
            mask_path = output_dir / f"{image_path.stem}_segmentation.nii.gz"
            nifti_img = nib.load(str(image_path))
            mask_nifti = nib.Nifti1Image(mask, nifti_img.affine, nifti_img.header)
            nib.save(mask_nifti, str(mask_path))
            logger.info(f"Segmentation saved to: {mask_path}")
            results['segmentation_path'] = str(mask_path)

        # Step 2: Feature Extraction
        logger.info("\n" + "=" * 60)
        logger.info("Step 2: Extracting features...")
        logger.info("=" * 60)

        # Determine which label to analyze (assume label 1 is the target, or use largest region)
        if len(np.unique(mask)) > 2:
            # Multiple labels - use largest non-background region
            labels, counts = np.unique(mask, return_counts=True)
            # Remove background (label 0)
            labels = labels[labels > 0]
            counts = counts[len(counts) - len(labels):]
            target_label = labels[np.argmax(counts)]
            logger.info(f"Multiple labels detected. Analyzing largest region (label {target_label})")
        else:
            target_label = 1

        # Create binary mask for target region
        binary_mask = (mask == target_label).astype(np.uint8)

        # Save temporary mask for radiomics
        temp_mask_path = None
        if output_dir:
            temp_mask_path = output_dir / "temp_mask.nii.gz"
            nifti_img = nib.load(str(image_path))
            mask_nifti = nib.Nifti1Image(binary_mask, nifti_img.affine, nifti_img.header)
            nib.save(mask_nifti, str(temp_mask_path))

        # Extract features
        features = self.feature_extractor.extract_all_features(
            image_path=image_path,
            mask=binary_mask,
            mask_path=temp_mask_path
        )

        results['features'] = features

        # Print feature summary
        feature_summary = self.feature_extractor.get_feature_summary(features)
        logger.info("\n" + feature_summary)

        # Save features if requested
        if save_features and output_dir:
            features_path = output_dir / f"{image_path.stem}_features.json"
            with open(features_path, 'w') as f:
                json.dump(features, f, indent=2)
            logger.info(f"Features saved to: {features_path}")
            results['features_path'] = str(features_path)

        # Clean up temporary mask
        if temp_mask_path and temp_mask_path.exists():
            temp_mask_path.unlink()

        # Step 3: Clinical Interpretation
        logger.info("\n" + "=" * 60)
        logger.info("Step 3: Generating clinical interpretation...")
        logger.info("=" * 60)

        interpretation = self.interpreter.generate_interpretation(
            features=features,
            structure_name=structure_name,
            patient_info=patient_info,
            clinical_context=clinical_context
        )

        results['interpretation'] = interpretation

        # Generate structured report
        report = self.interpreter.generate_report(
            features=features,
            structure_name=structure_name,
            patient_info=patient_info,
            clinical_context=clinical_context
        )

        results['report'] = report

        # Format and display report
        formatted_report = self.interpreter.format_report(report)
        logger.info("\n" + formatted_report)

        # Save report if requested
        if save_report and output_dir:
            report_path = output_dir / f"{image_path.stem}_report.txt"
            with open(report_path, 'w') as f:
                f.write(formatted_report)
            logger.info(f"Report saved to: {report_path}")

            # Also save JSON version
            report_json_path = output_dir / f"{image_path.stem}_report.json"
            report_with_meta = {
                'timestamp': datetime.now().isoformat(),
                'image_path': str(image_path),
                'structure': structure_name,
                **report
            }
            with open(report_json_path, 'w') as f:
                json.dump(report_with_meta, f, indent=2)

            results['report_path'] = str(report_path)
            results['report_json_path'] = str(report_json_path)

        logger.info("\n" + "=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)

        return results

    def batch_process(
        self,
        image_paths: list,
        structure_name: str = "brain region",
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> list:
        """
        Process multiple images in batch

        Args:
            image_paths: List of paths to MRI images
            structure_name: Name of anatomical structure
            output_dir: Base output directory
            **kwargs: Additional arguments for run()

        Returns:
            List of result dictionaries
        """
        logger.info(f"Starting batch processing of {len(image_paths)} images...")

        results_list = []

        for i, image_path in enumerate(image_paths):
            logger.info(f"\n\nProcessing image {i+1}/{len(image_paths)}: {image_path}")

            # Create individual output directory
            if output_dir:
                output_dir = Path(output_dir)
                image_output_dir = output_dir / Path(image_path).stem
            else:
                image_output_dir = None

            try:
                results = self.run(
                    image_path=image_path,
                    structure_name=structure_name,
                    output_dir=image_output_dir,
                    **kwargs
                )
                results['success'] = True
                results['error'] = None

            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results = {
                    'success': False,
                    'error': str(e),
                    'image_path': str(image_path)
                }

            results_list.append(results)

        logger.info(f"\nBatch processing completed!")
        logger.info(f"Successful: {sum(r['success'] for r in results_list)}/{len(image_paths)}")

        return results_list


def main():
    """Example usage of the pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description='MRI Diagnosis Pipeline')
    parser.add_argument('--image', type=str, required=True, help='Path to MRI image')
    parser.add_argument('--structure', type=str, default='brain region', help='Anatomical structure name')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Segmentation model checkpoint')
    parser.add_argument('--llm', type=str, default='microsoft/phi-2', help='LLM model name')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU')
    parser.add_argument('--no-llm', action='store_true', help='Use rule-based interpretation')

    args = parser.parse_args()

    # Patient info example
    patient_info = {
        'age': 65,
        'sex': 'M',
        'chief_complaint': 'Memory loss'
    }

    # Initialize pipeline
    pipeline = MRIDiagnosisPipeline(
        segmentation_checkpoint=args.checkpoint,
        llm_model_name=args.llm,
        use_gpu=not args.no_gpu,
        use_llm=not args.no_llm
    )

    # Run pipeline
    results = pipeline.run(
        image_path=args.image,
        structure_name=args.structure,
        patient_info=patient_info,
        output_dir=args.output
    )

    print("\nPipeline completed successfully!")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
