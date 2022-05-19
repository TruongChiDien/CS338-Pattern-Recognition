import torch
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
from facenet_pytorch import fixed_image_standardization

DATA_PATH = './data'

def trans(img):
    transform = transforms.Compose([
            transforms.ToTensor(),
            # fixed_image_standardization
        ])
    return transform(img)

def load_faceslist():
    embeds = torch.load(DATA_PATH+'/faceslist.pth')
    names = np.load(DATA_PATH+'/usernames.npy')
    return embeds, names


def extract_face(box, img):
    img = img[box[1]:box[3], box[0]:box[2]]
    face = cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA)
    face = Image.fromarray(face)
    return face


def inference(model, face, local_embeds, names):
    embed = model(torch.unsqueeze(trans(face), 0).to(model.device))
    norm_score = torch.nn.functional.cosine_similarity(embed, local_embeds)
    min_dist, embed_idx = torch.max(norm_score, dim=0)
    print(min_dist, names[embed_idx])
    return embed_idx, min_dist.double()


def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)
