import cv2
import numpy as np
import os
import pywt

def wavelet_denoise(
    img: np.ndarray, 
    threshold: int | None = 0.04, 
    wavelet: str | None = 'sym4', 
    level: int | None = 1
):
    """
    Apply wavelet denoising on an image.

    Args:
        img (ndarray): Input image array.
        threshold (float): Threshold for filtering noise in wavelet coefficients.
        wavelet (str): Type of wavelet to use.
        level (int): Number of decomposition levels.

    Returns:
        denoised_img (ndarray): Denoised image.
    """
    # Decompose the image using discrete wavelet transform
    coeffs = pywt.wavedec2(data=img, wavelet=wavelet, level=level)

    # List to hold modified coefficients
    coeffs_modified = coeffs[:]
    for i in range(1, len(coeffs)):
        coeffs_modified[i] = list(coeffs[i])
        # Apply thresholding to each set of coefficients except the approximation coefficients
        for j in range(len(coeffs_modified[i])):
            coeffs_modified[i][j] = pywt.threshold(coeffs_modified[i][j], 
                                                   threshold * np.max(coeffs_modified[i][j]), 
                                                   mode='soft')

    # Reconstruct the image from the modified coefficients
    denoised_img = pywt.waverec2(coeffs_modified, wavelet)
    denoised_img = np.clip(denoised_img, 0, 255)  # Ensure pixel values are valid
    denoised_img = np.uint8(denoised_img)  # Convert to unsigned byte format

    return denoised_img

def process_images(input_dir: str, output_dir: str):
    """
    Process all images in the input directory and save the denoised images in the output directory.

    Args:
        input_dir (str): Directory containing the original images.
        output_dir (str): Directory where denoised images will be saved.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all image files in the input directory
    img_names = os.listdir(input_dir)
    for index, img_name in enumerate(img_names, 1):
        input_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, f'cq{index}.tif')

        # Read image
        img = cv2.imread(input_path)
        if img is None:
            print(f"Failed to load image {input_path}")
            continue

        # Convert to grayscale (if needed)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Denoise the image
        denoised_img = wavelet_denoise(img)

        # Save the denoised image
        cv2.imwrite(output_path, denoised_img)
        print(f"Denoised image saved to {output_path}")


if __name__ == '__main__':
    old_folder = 'data_old'
    new_folder = 'data'
    for child_folder in os.listdir(old_folder):
        input_folder = os.path.join(old_folder,child_folder)
        output_folder = os.path.join(new_folder,child_folder)
        os.makedirs(output_folder, exist_ok=True)
        process_images(input_folder, output_folder)