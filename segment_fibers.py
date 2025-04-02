import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
from main import process_fiber_image
import os
from skimage.measure import regionprops

def segment_fibers(image_path, output_path='./results'):
    # Create masks directory if it doesn't exist
    masks_dir = "./masks"
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir)
    
    # First, get the denoised binary image from main.py
    results = process_fiber_image(image_path)
    denoised_binary = results['segmented_denoised']
    
    # Read the original image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Get image dimensions for area calculation
    height, width = image.shape[:2]
    total_area = height * width
    max_mask_area = total_area / 3
    
    # Create a light gray background (RGB: 240, 240, 240)
    background = np.ones_like(image) * 240
    
    # Use the denoised binary as a mask to combine original image with background
    mask_binary = denoised_binary > 0
    processed_image = image.copy()
    processed_image[~mask_binary] = background[~mask_binary]
    
    # Save the background-removed image
    cv2.imwrite("./results/background_removed.png", processed_image)
    
    # Set up SAM model
    model_type = "vit_h"
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sam = sam_model_registry[model_type]()
    sam.load_state_dict(torch.load(sam_checkpoint, weights_only=True))
    sam.to(device)
    
    # Adjusted parameters for better fiber detection and separation
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,             # Increased points per side based on user edit
        pred_iou_thresh=0.88,           # Lowered to accept more masks
        stability_score_thresh=0.92,    # Lowered to accept less stable masks
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=800,       # Lowered significantly to detect smaller fibers/fragments
        output_mode="binary_mask",
        stability_score_offset=0.1,
        box_nms_thresh=0.7,             # Lowered significantly to reduce suppression of overlapping masks (crucial for crossed fibers)
        crop_overlap_ratio=0.01
    )
    
    # Convert BGR to RGB for SAM
    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    
    # Generate masks
    print("Generating masks with SAM...")
    generated_masks = mask_generator.generate(processed_image_rgb)
    print(f"Generated {len(generated_masks)} initial masks.")
    
    # Filter masks based on area and shape (eccentricity)
    filtered_masks = []
    eccentricity_threshold = 0.95 # Keep only highly elongated shapes
    
    for i, mask_data in enumerate(generated_masks):
        mask_area = mask_data['area']
        # --- Area Filter --- 
        if mask_area < max_mask_area:
            # --- Shape Filter --- 
            # Create binary mask for shape analysis (needs integer type)
            binary_mask_for_props = np.zeros((height, width), dtype=np.uint8)
            binary_mask_for_props[mask_data['segmentation']] = 1 # Use 1 for regionprops
            
            # Calculate shape properties
            props = regionprops(binary_mask_for_props)
            if props: # Ensure regionprops found something
                eccentricity = props[0].eccentricity
                if eccentricity > eccentricity_threshold: # Pass shape filter
                    filtered_masks.append(mask_data)
                    # Optional: print(f"Kept mask {i+1} (Area: {mask_area}, Ecc: {eccentricity:.3f})")
                else:
                    print(f"Filtering out mask {i+1} due to low eccentricity: {eccentricity:.3f} (Threshold > {eccentricity_threshold})")
            else:
                 print(f"Could not calculate properties for mask {i+1}, filtering out.")

        else: # Failed area filter
            print(f"Filtering out large mask {i+1} with area {mask_area} (Threshold < {max_mask_area:.0f})")

    print(f"Kept {len(filtered_masks)} masks after area and shape filtering.")
    masks = filtered_masks # Use the filtered list from now on
    
    # Create a copy of the processed image for visualization
    image_with_masks = processed_image.copy()
    
    # Draw each mask with red color and save individual masks
    for i, mask_data in enumerate(masks):
        # Create a binary mask image
        binary_mask = np.zeros((height, width), dtype=np.uint8)
        binary_mask[mask_data['segmentation']] = 255
        
        # Save individual mask
        mask_path = os.path.join(masks_dir, f'mask_{i+1}.png')
        cv2.imwrite(mask_path, binary_mask)
        
        # Apply the mask with red color (BGR: 0, 0, 255)
        mask_overlay = np.zeros_like(processed_image)
        mask_overlay[mask_data['segmentation']] = [0, 0, 255]  # Red in BGR
        # Blend the mask with the processed image
        image_with_masks = cv2.addWeighted(image_with_masks, 1, mask_overlay, 0.5, 0)
    
    # Save the result
    cv2.imwrite(output_path, image_with_masks)
    
    # Display the results
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # Background removed image
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    plt.title('Background Removed')
    plt.axis('off')
    
    # Segmented image
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(image_with_masks, cv2.COLOR_BGR2RGB))
    plt.title('Segmented Fibers (Filtered by Area & Shape)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return masks

if __name__ == "__main__":
    image_path = "./images/sample4.jpg"
    output_path = "./results/segmented_fibers_filtered_shape.png" # Changed output filename
    masks = segment_fibers(image_path, output_path)
    print(f"Segmentation completed. Results saved to {output_path}")
    print(f"Individual masks saved to ./masks directory") 