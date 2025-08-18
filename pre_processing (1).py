import cv2
import numpy as np
import os
import argparse


def histogram_equalization_for_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        if filename.lower().endswith('.png'):
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            hist, bins = np.histogram(image.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_min = cdf[cdf > 0].min()

            mn = image.size
            h = ((cdf - cdf_min) / (mn - cdf_min)) * 255
            h = h.clip(0, 255).astype('uint8')

            equalized_image = h[image]

            cv2.imwrite(file_path, equalized_image)

            print(f"Processed and saved: {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform histogram equalization on all PNG images in a directory.')
    parser.add_argument('--datapath', type=str, required=True, help='Path to the directory containing PNG images')

    args = parser.parse_args()

    histogram_equalization_for_directory(args.datapath)
