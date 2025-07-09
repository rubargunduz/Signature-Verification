# signature.py
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist


def resize_with_padding(image, size=(300, 300)):
    old_size = image.shape[:2]  # (height, width)
    ratio = min(size[0] / old_size[0], size[1] / old_size[1])
    new_size = (int(old_size[1] * ratio), int(old_size[0] * ratio))
    image_resized = cv2.resize(image, new_size)

    delta_w = size[1] - new_size[0]
    delta_h = size[0] - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    new_image = cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_image


def remove_small_components(image, dist_ratio=0.5):
    h, w = image.shape[:2]
    thresh_dist = dist_ratio * ((w + h) / 2)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = (gray < 128).astype(np.uint8) * 255

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    output_mask = np.zeros_like(binary)

    for label in range(1, num_labels):
        ys, xs = np.where(labels == label)
        points = np.column_stack((xs, ys))

        if len(points) < 2:
            continue

        distances = pdist(points)
        max_dist = distances.max()

        if max_dist >= thresh_dist:
            output_mask[labels == label] = 255

    white_bg = np.full_like(image, 255)
    cleaned = np.where(output_mask[:, :, None] == 255, image, white_bg)
    return cleaned


def load_and_preprocess(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image at {path}")

    img = resize_with_padding(img, (300, 300))
    
    # Remove unwanted components
    cleaned = remove_small_components(img)

    # Convert to grayscale, invert, binarize
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Convert to 3-channel RGB
    img_rgb = cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB)

    # Resize and normalize
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    return transform(img_rgb).unsqueeze(0)


def get_embedding(model, img_tensor):
    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding.squeeze().numpy()


# Load pretrained ResNet18 model without the final classifier
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Identity()
resnet.eval()



import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
# Convert tensor back to image (unnormalize first)
def show_tensor(tensor, title=""):
    # Undo normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.squeeze(0) * std + mean  # [3, H, W]

    # Convert to numpy image and display
    img = F.to_pil_image(img.clamp(0, 1))  # clamp to valid range
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()



def match(path1, path2):
    img1_tensor = load_and_preprocess(path1)
    img2_tensor = load_and_preprocess(path2)

    #show_tensor(img1_tensor, "Image 1 (after preprocessing)") # you can visualize the tensors here
    #show_tensor(img2_tensor, "Image 2 (after preprocessing)") # uncomment if not running with flask server

    emb1 = get_embedding(resnet, img1_tensor)
    emb2 = get_embedding(resnet, img2_tensor)

    similarity = cosine_similarity([emb1], [emb2])[0][0]
    return round(similarity * 100, 2)
