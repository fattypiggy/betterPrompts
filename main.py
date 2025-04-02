import cv2
import numpy as np
import os
from skimage.morphology import skeletonize
from skimage.util import invert
from scipy import ndimage
from skimage.measure import label

def process_fiber_image(image_path, output_dir='./results'):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to grayscale if it's not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Step 1: Segment foreground (black parts)
    # Assuming black is foreground (fibers) and gray is background
    # Threshold to get binary image
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Save initial segmented image
    cv2.imwrite(os.path.join(output_dir, 'segmented_initial.png'), binary)
    
    # Step 1.5: Fill in hollow fibers and remove small gaps/dots
    # First, apply morphological closing to close small gaps
    kernel_size = 5  # Adjust based on the size of gaps
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours and fill them to handle larger hollow areas
    contours, _ = cv2.findContours(binary_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_binary = np.zeros_like(binary)
    cv2.drawContours(filled_binary, contours, -1, 255, -1)  # -1 means fill the contours
    
    # Apply additional morphological operations to clean up the filled image
    filled_binary = cv2.morphologyEx(filled_binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    
    # Save filled binary image
    cv2.imwrite(os.path.join(output_dir, 'segmented_filled.png'), filled_binary)
    
    # Step 1.6: Denoise by removing small objects (area < 300 pixels)
    # Perform connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filled_binary, connectivity=8)
    
    # Create an output image to store the denoised result
    denoised_binary = np.zeros_like(filled_binary)
    
    # Process each component (skipping label 0, which is the background)
    for label_id in range(1, num_labels):
        # Get the area of the component
        area = stats[label_id, cv2.CC_STAT_AREA]
        
        # If the area is greater than or equal to threshold, keep it
        if area >= 900:
            denoised_binary[labels == label_id] = 255
    
    # Save denoised binary image
    cv2.imwrite(os.path.join(output_dir, 'segmented_denoised.png'), denoised_binary)
    
    # Step 2: Skeletonize the denoised binary image
    # Convert to format expected by skimage
    binary_for_skeleton = denoised_binary > 0
    
    # Perform skeletonization
    skeleton = skeletonize(binary_for_skeleton)
    
    # Convert back to uint8 for OpenCV
    skeleton_img = np.uint8(skeleton) * 255
    
    # Save skeleton image
    cv2.imwrite(os.path.join(output_dir, 'skeleton.png'), skeleton_img)
    
    # Step 3: Find junction points
    # Convolve with a filter to detect junctions
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0  # Remove the center
    
    # Count neighbors for each pixel
    neighbors = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    
    # Junction points have more than 2 neighbors
    junctions = np.logical_and(skeleton, neighbors > 2)
    junction_coords = np.where(junctions)
    
    # Mark junctions on original image
    result_img = img.copy()
    if len(result_img.shape) == 2:
        # Convert to color for marking junctions
        result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
    
    # Draw red circles at junction points
    for y, x in zip(junction_coords[0], junction_coords[1]):
        cv2.circle(result_img, (x, y), 5, (0, 0, 255), -1)  # Red circle with radius 5
    
    # Save result with marked junctions
    cv2.imwrite(os.path.join(output_dir, 'marked_junctions.png'), result_img)
    
    # Step 4: Treat skeleton as a graph and identify individual fiber segments
    # Create a copy of the skeleton
    fiber_segments = skeleton.copy().astype(np.uint8)
    
    # Remove junction points to disconnect fiber segments
    for y, x in zip(junction_coords[0], junction_coords[1]):
        fiber_segments[y, x] = 0
    
    # Label the disconnected segments
    labeled_segments = label(fiber_segments)
    num_segments = np.max(labeled_segments)
    
    # Create image with labeled fibers
    labeled_img = img.copy()
    if len(labeled_img.shape) == 2:
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_GRAY2BGR)
    
    # Find the center point of each segment for label placement
    for segment_id in range(1, num_segments + 1):
        # Get coordinates of points in this segment
        segment_points = np.where(labeled_segments == segment_id)
        
        if len(segment_points[0]) > 0:
            # Calculate center point of the segment
            center_y = int(np.mean(segment_points[0]))
            center_x = int(np.mean(segment_points[1]))
            
            # Draw ID on the image with larger font size (increased scale from 0.7 to 1.2)
            cv2.putText(labeled_img, str(segment_id), (center_x, center_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Save result with labeled fibers
    cv2.imwrite(os.path.join(output_dir, 'labeled_fibers.png'), labeled_img)
    
    return {
        'original': img,
        'segmented_initial': binary,
        'segmented_filled': filled_binary,
        'segmented_denoised': denoised_binary,
        'skeleton': skeleton_img,
        'labeled_fibers': labeled_img,
        'result': result_img
    }

# Example usage
if __name__ == "__main__":
    # Replace with the path to your image
    image_path = "./images/sample.jpg"
    results = process_fiber_image(image_path)
    print(f"Results saved to ./results folder")
