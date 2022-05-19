import glob
import torch 
from facenet_pytorch import InceptionResnetV1
import os
from PIL import Image
import numpy as np
from utils import trans

IMG_PATH = './data/test_images'
DATA_PATH = './data'
embeddings = []
names = []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


model = InceptionResnetV1(
    classify=False,
    pretrained="vggface2"
).to(device)

model.eval()

for usr in os.listdir(IMG_PATH):
    embeds = []
    for file in glob.glob(os.path.join(IMG_PATH, usr)+'/*.jpg'):
        try:
            img = Image.open(file)
        except:
            continue
        with torch.no_grad():
            embed = model(torch.unsqueeze(trans(img), 0).to(device))
            embeds.append(embed) #1 anh, kich thuoc [1,512]
    if len(embeds) == 0:
        continue
    embedding = torch.cat(embeds).mean(0, keepdim=True) #dua ra trung binh cua 50 anh, kich thuoc [1,512]
    embeddings.append(embedding) # 1 cai list n cai [1,512]
    names.append(usr)

embeddings = torch.cat(embeddings) #[n,512]
names = np.array(names)
torch.save(embeddings, DATA_PATH+"/faceslist.pth")
np.save(DATA_PATH+"/usernames", names)
