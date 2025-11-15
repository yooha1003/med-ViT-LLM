"""
Example script for running MRI diagnosis pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pipeline.main_pipeline import MRIDiagnosisPipeline


def example_basic_usage():
    """Basic usage example"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # Initialize pipeline
    pipeline = MRIDiagnosisPipeline(
        segmentation_model_type="unet3d",
        llm_model_name="microsoft/phi-2",
        use_gpu=True,
        use_llm=False  # Set to True to use actual LLM (requires more resources)
    )

    # Patient information
    patient_info = {
        'age': 65,
        'sex': 'Male',
        'chief_complaint': 'Memory loss and confusion',
        'clinical_history': 'Progressive cognitive decline over 2 years'
    }

    # Run pipeline on a single image
    # Note: Replace with actual image path
    image_path = "data/raw/example_mri.nii.gz"

    results = pipeline.run(
        image_path=image_path,
        structure_name="Hippocampus",
        patient_info=patient_info,
        clinical_context="Evaluation for neurodegenerative disease",
        output_dir="output/example1",
        save_segmentation=True,
        save_features=True,
        save_report=True
    )

    print("\n✓ Processing completed!")
    print(f"Segmentation shape: {results['segmentation_mask'].shape}")
    print(f"Number of features: {len(results['features'])}")
    print(f"Report saved to: {results.get('report_path', 'N/A')}")


def example_batch_processing():
    """Batch processing example"""
    print("\n" + "=" * 60)
    print("Example 2: Batch Processing")
    print("=" * 60)

    # Initialize pipeline
    pipeline = MRIDiagnosisPipeline(
        segmentation_model_type="unet3d",
        use_llm=False
    )

    # List of images to process
    image_paths = [
        "data/raw/patient1_mri.nii.gz",
        "data/raw/patient2_mri.nii.gz",
        "data/raw/patient3_mri.nii.gz",
    ]

    # Process all images
    results_list = pipeline.batch_process(
        image_paths=image_paths,
        structure_name="Brain Tumor",
        output_dir="output/batch_results"
    )

    # Summary
    successful = sum(r['success'] for r in results_list)
    print(f"\n✓ Batch processing completed!")
    print(f"Successful: {successful}/{len(image_paths)}")


def example_custom_configuration():
    """Example with custom configuration"""
    print("\n" + "=" * 60)
    print("Example 3: Custom Configuration")
    print("=" * 60)

    from utils.config_loader import load_config

    # Load configuration
    config = load_config("configs/config.yaml")

    # Initialize pipeline with config
    pipeline = MRIDiagnosisPipeline(
        segmentation_model_type=config['segmentation']['model_type'],
        llm_model_name=config['interpretation']['model_name'],
        use_gpu=config['device']['use_gpu'],
        use_llm=config['interpretation']['use_llm']
    )

    print("✓ Pipeline initialized with custom configuration")


def example_with_visualization():
    """Example with visualization"""
    print("\n" + "=" * 60)
    print("Example 4: With Visualization")
    print("=" * 60)

    from utils.visualization import visualize_segmentation, visualize_3d_slices

    # Initialize pipeline
    pipeline = MRIDiagnosisPipeline(use_llm=False)

    # Run pipeline
    image_path = "data/raw/example_mri.nii.gz"
    results = pipeline.run(
        image_path=image_path,
        structure_name="Pituitary Gland",
        output_dir="output/example4"
    )

    # Visualize results
    if 'segmentation_path' in results:
        # Single slice visualization
        visualize_segmentation(
            image=image_path,
            mask=results['segmentation_path'],
            save_path="output/example4/visualization_single.png"
        )

        # Multiple slices visualization
        visualize_3d_slices(
            image=image_path,
            mask=results['segmentation_path'],
            num_slices=9,
            save_path="output/example4/visualization_multi.png"
        )

        print("✓ Visualizations saved!")


def example_feature_analysis():
    """Example focusing on feature extraction"""
    print("\n" + "=" * 60)
    print("Example 5: Detailed Feature Analysis")
    print("=" * 60)

    from feature_extraction.extractor import FeatureExtractor
    import nibabel as nib
    import json

    # Initialize feature extractor
    extractor = FeatureExtractor(use_radiomics=True)

    # Load image and mask
    image_path = "data/raw/example_mri.nii.gz"
    mask_path = "data/processed/example_mask.nii.gz"

    # Extract all features
    features = extractor.extract_all_features(
        image_path=image_path,
        mask_path=mask_path
    )

    # Print feature summary
    summary = extractor.get_feature_summary(features)
    print(summary)

    # Save features
    with open("output/features_detailed.json", 'w') as f:
        json.dump(features, f, indent=2)

    print(f"\n✓ Extracted {len(features)} features")
    print("✓ Features saved to output/features_detailed.json")


def example_llm_interpretation():
    """Example focusing on LLM interpretation"""
    print("\n" + "=" * 60)
    print("Example 6: LLM Interpretation Only")
    print("=" * 60)

    from interpretation.llm_interpreter import LLMInterpreter, SimpleLLMInterpreter

    # Example features (you would get these from feature extraction)
    features = {
        'volume_cm3': 2.3,
        'surface_area_mm2': 1450.5,
        'sphericity': 0.65,
        'elongation': 0.72,
        'flatness': 0.58,
        'intensity_mean': 145.3,
        'intensity_std': 23.8
    }

    # Patient info
    patient_info = {
        'age': 72,
        'sex': 'Female',
        'symptoms': 'Memory impairment, difficulty with word finding'
    }

    # Use simple interpreter (no LLM required)
    interpreter = SimpleLLMInterpreter()

    # Generate interpretation
    interpretation = interpreter.generate_interpretation(
        features=features,
        structure_name="Hippocampus",
        patient_info=patient_info,
        clinical_context="Cognitive decline assessment"
    )

    print("\nClinical Interpretation:")
    print("-" * 60)
    print(interpretation)
    print("-" * 60)

    # Generate full report
    report = interpreter.generate_report(
        features=features,
        structure_name="Hippocampus",
        patient_info=patient_info
    )

    formatted = interpreter.format_report(report)
    print("\n" + formatted)


if __name__ == "__main__":
    print("\n🏥 MRI Diagnosis Pipeline - Examples")
    print("=" * 60)

    # Note: Most examples require actual MRI data
    # You can run individual examples by uncommenting them

    # example_basic_usage()
    # example_batch_processing()
    # example_custom_configuration()
    # example_with_visualization()
    # example_feature_analysis()
    example_llm_interpretation()  # This one works without actual MRI data

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
