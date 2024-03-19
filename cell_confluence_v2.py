import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import argparse
import csv

def kmeans_segmentation(image, num_clusters=3):
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 3)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(pixels)
    
    # Get the cluster centers and labels
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    
    # Reshape the labels to the original image shape
    segmented_image = cluster_centers[cluster_labels].reshape(image.shape)
    
    # Convert the segmented image to 8-bit unsigned integer format
    segmented_image_uint8 = segmented_image.astype(np.uint8)
    
    return segmented_image_uint8

def load_image(image_path):
    # Read the image
    return cv2.imread(image_path)

def blur_image(image, kernel_size=(17, 17)):
    # Apply Gaussian blurring to the image
    return cv2.GaussianBlur(image, kernel_size, 0)

def edge_detection(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform Canny edge detection
    return cv2.Canny(gray_image, 30, 100)

def display_images(images, titles, output_folder):
    num_images = len(images)
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    for i in range(num_images):
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.savefig(os.path.join(output_folder, f'{titles[i]}.png'), bbox_inches='tight', pad_inches=0)  # Save with the title as the image name
        plt.close()



def create_mask_from_edges(edges):
    # Create a binary mask where edges are white (255) and everything else is black (0)
    mask = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)[1]
    
    return mask

# Function to save the mask as an image
def save_mask(mask, filename):
    # Write the mask to an image file
    cv2.imwrite(filename, mask)

def overlay_mask(image, mask, alpha_image=0.6, alpha_mask=1.0, color=(0, 255, 0)):
    # Resize mask to match image size if needed
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # Convert mask to float and normalize
    mask = mask.astype(np.float32) / 255.0
    
    # Convert image to float and apply alpha
    image = image.astype(np.float32) * alpha_image
    
    # Modify the color channels of the mask
    mask_color = np.zeros_like(image)
    mask_color[:, :, 1] = mask[:, :, 0] * color[1]  # Green channel
    
    # Combine image and modified mask
    overlay = cv2.addWeighted(image, 1.0, mask_color, alpha_mask, 0)
    return overlay.astype(np.uint8)

def calculate_green_to_red_ratio(image):
    # Split the image into color channels
    b, g, r = cv2.split(image)
    
    # Calculate the total number of green pixels
    total_green_pixels = np.sum(g)
    
    # Calculate the total number of red pixels
    total_red_pixels = np.sum(r)
    
    # Calculate the ratio of green pixels to red pixels
    ratio = total_green_pixels / (total_red_pixels + total_green_pixels)
    
    # Convert the ratio to a percentage with three significant figures
    ratio_percentage = round(ratio * 100, 3)
    
    return ratio_percentage



def process_image_pipeline(original_image_path, processed_image_path, save_path):
    # Load the original and model images
    model_image = load_image(original_image_path)
    original_image = load_image(processed_image_path)
    
    # Blur the model image
    blurred_image = blur_image(model_image)
    
    # Perform k-means segmentation on the blurred image
    kmeans_image = kmeans_segmentation(blurred_image)
    
    # Calculate the green to red ratio in the k-means image
    green_to_red_ratio = calculate_green_to_red_ratio(kmeans_image)
    
    
    # Perform edge detection on the k-means image
    edges = edge_detection(kmeans_image)
    
    # Create a mask from the edges
    mask = create_mask_from_edges(edges)
    
    # Save the mask as an image
    save_mask(mask, save_path)
    
    # Load the saved mask image
    mask_image = cv2.imread(save_path)
    
    # Overlay the modified mask with green outlines onto the original image
    overlay = overlay_mask(original_image, mask_image, color=(0, 255, 0))
    
    return original_image, model_image, blurred_image, kmeans_image, green_to_red_ratio, edges, overlay



def main():
    parser = argparse.ArgumentParser(description='Process images.')
    parser.add_argument('input_folder', type=str, help='Path to the input folder containing processed images')
    parser.add_argument('output_folder', type=str, help='Path to the output folder for saving results')
    args = parser.parse_args()
    
    # Initialize a list to store image names and their associated green-to-red ratios
    image_ratios = []
    
    # Iterate over each file in the input folder
    for filename in os.listdir(args.input_folder):
        # Check if the file is an image (e.g., JPEG or PNG)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Create a directory for the current image
            image_output_folder = os.path.join(args.output_folder, filename.replace('.', '_'))
            os.makedirs(image_output_folder, exist_ok=True)
            
            # Construct the full paths for the original and processed images
            original_image_path = os.path.join(args.output_folder, filename.replace(".jpg", "_probabilities.png"))
            processed_image_path = os.path.join(args.input_folder, filename)
            save_path = os.path.join(image_output_folder, "mask_" + filename.replace(".jpg", ".png"))
            
            # Process the image pipeline
            original_image, model_image, blurred_image, kmeans_image, green_to_red_ratio, edges, overlay = process_image_pipeline(original_image_path, processed_image_path, save_path)
            print(f"Processing image: {filename}")
            print("Green to Red Ratio in K-means Image:", green_to_red_ratio)
            
            # Add image name and its associated green-to-red ratio to the list
            image_ratios.append((filename, green_to_red_ratio))

            # Display the processed images and save them in the newly created directory
            titles = ["Original Image", "Model Image", "Blurred Image", "K-means Image", "Edge Image", "Overlay with Mask"]
            display_images([original_image, model_image, blurred_image, kmeans_image, edges, overlay], titles, image_output_folder)
    
    # Save the image names and their associated green-to-red ratios to a CSV file
    csv_file_path = os.path.join(args.output_folder, "image_ratios.csv")
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image Name', 'Green-to-Red Ratio in K-means Image'])
        csv_writer.writerows(image_ratios)


if __name__ == "__main__":
    main()


