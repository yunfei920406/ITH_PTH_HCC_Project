import ants
import numpy as np


def process_and_save_image(input_path, output_path, resample_params, interp_type=0):
    """
    Read the image from the specified path, perform resampling, normalize the image to the range [0, 255],
    convert it to uint8, and save it to the specified path.

    Parameters:
    - input_path (str): Path to the input image.
    - output_path (str): Path to save the processed image.
    - resample_params (tuple): Target spacing for resampling, e.g., (1.0, 1.0, 4.0).
    - interp_type (int): Interpolation type for resampling. Default is 0 (linear). Other options include
      1 (nearest neighbor), 2 (gaussian), 3 (windowed sinc), 4 (bspline).

    Returns:
    - None
    """
    # Read the image
    img = ants.image_read(input_path)

    # Resample the image
    img_resampled = ants.resample_image(img, resample_params, interp_type=interp_type)

    # Convert to numpy array and normalize to [0, 255]
    img_array = img_resampled.numpy()
    img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array)) * 255
    img_array = img_array.astype(np.uint8)

    # Create a new ANTs image object
    img_normalized = ants.from_numpy(img_array, origin=img_resampled.origin,
                                     spacing=img_resampled.spacing,
                                     direction=img_resampled.direction)

    # Save the processed image
    ants.image_write(img_normalized, output_path)
