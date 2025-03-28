# Cotton Fiber Image Analysis

This tool processes microscope images of cotton fibers to identify and analyze fiber structures.

## Features

- Segmentation of cotton fibers from background
- Filling of hollow fibers to create solid structures
- Skeletonization to extract the centerline of fibers
- Detection and marking of junction points (where fibers cross)
- Saving of intermediate processing results

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- scikit-image
- SciPy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/cotton-fiber-analysis.git
   cd cotton-fiber-analysis
   ```

2. Install required packages:
   ```
   pip install numpy opencv-python scikit-image scipy
   ```

## Usage

1. Place your cotton fiber microscope images in the `images` folder (create it if it doesn't exist)

2. Run the script:
   ```
   python main.py
   ```

3. By default, the script will process `./images/sample.jpg` and save results to `./results/`

4. To process a different image, modify the `image_path` variable in the script or import the function:
   ```python
   from main import process_fiber_image
   
   results = process_fiber_image("path/to/your/image.jpg")
   ```

## Output

The script generates the following output files in the `results` directory:

- `segmented_initial.png`: Initial binary segmentation of fibers
- `segmented_filled.png`: Filled binary image with hollow areas closed
- `skeleton.png`: 1-pixel wide skeleton/centerline of the fibers
- `marked_junctions.png`: Original image with detected junctions marked in red

## Background

This tool is designed for analyzing cotton fibers which:
1. Appear as thin, elongated strips in microscope images
2. Show as black objects against a gray background
3. May have hollow centers or small gaps that need filling
4. Often cross each other, creating junction points

The analysis helps in quantifying fiber structure and arrangement. 