# Panoramic Image Stitching

This project is a Python application that stitches together three images to create a panoramic view. It uses OpenCV and feature detection methods such as SIFT and ORB.

## Requirements

This project requires Python 3.9.6 and the following Python libraries:

- matplotlib=3.4.2
- pandas=1.5.3
- numpy=1.20.3
- tqdm=4.65.0
- opencv-contrib-python=4.5.1.48


## Installation

1. Clone the repository:

```bash
git clone https://github.com/CristianComellas21/panoramic_view_creator.git
```

2. Navigate to the project directory:

```bash
cd panoramic_view_creator
```

3. Install miniconda (if not already installed) following the instructions [here](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html).

4. Create a new conda environment with the required dependencies:

```bash
conda env create -f requirements.yaml -n [DESIRED_ENV_NAME]
```

| :exclamation:  This step can take a few minutes. |
| --------------------------------------------------- |


## Usage

### Terminal

You can run the script `create_panoramic.py` with the following command:

```bash
python create_panoramic.py [-h] [-o OUTPUT] [-r RATIO] [-m {'SIFT','ORB'}] [-v] [-c] -i IMAGE_PATHS [IMAGE_PATHS ...]
```

**Arguments**

- `-i` or `--image_paths`: Paths to the 3 images to be stitched in order left-to-right.
- `-o` or `--output`: Path to the output file. It will create the intermediate directories if they don't exist (default: "output/panoramic_result.png").
- `-r` or `--ratio`: Ratio for the correct matches selection (default: 0.65).
- `-m` or `--method`: Method for feature detection (choices: ["SIFT", "ORB"], default: "SIFT").
- `-v` or `--verbose`: Print verbose output (optional).
- `-c` or `--crop`: Crop the result image (optional).

You can also run the script with the `-h` or `--help` flag to display the help message.

```bash
python create_panoramic.py -h
```

### Function

You can also use the `create_panoramic_view` function in your Python code. Here's an example:

```python
from algorithm import create_panoramic_view

# Provide paths to the 3 images in order left-to-right
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]

# Call the function
panoramic_image = create_panoramic_view(
    images=image_paths,
    match_ratio=0.65,
    crop=True,
    detector="SIFT",
    verbose=True
)

# Save the final image
cv.imwrite("output/panoramic_result.png", panoramic_image)
```

Adjust the parameters as needed.


## License
This project is licensed under the terms of the MIT license.