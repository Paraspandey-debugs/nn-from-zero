import os
import numpy as np
import cv2

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    
    if np.mean(image) >= 127.5:
        image = 255 - image
    
    image_data = (image.reshape(1, -1).astype(np.float32) - 127.5) / 127.5
    return image_data

def preprocess_dataset(dataset_path):
    images = []
    labels = []
    
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                image_data = load_and_preprocess_image(image_path)
                images.append(image_data)
                labels.append(int(label))
    
    return np.array(images), np.array(labels)