import cv2
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from utils import load_faceslist, extract_face, inference, automatic_brightness_and_contrast

embeddings = []
names = []

device = 'cpu'

model = InceptionResnetV1(
    classify=False,
    pretrained="vggface2"
).to(device)
model.eval()

cap = cv2.VideoCapture(0)
mtcnn = MTCNN(thresholds=[0.7, 0.7, 0.8], post_process=False, keep_all=True, device=device)
embeddings, names = load_faceslist()

while cap.isOpened():
    isSuccess, frame = cap.read()

    if not isSuccess:
        continue
    
    frame, _, _ = automatic_brightness_and_contrast(frame)

    boxes, _ = mtcnn.detect(frame)
    if boxes is not None:
        boxes = boxes.astype('int').tolist()
        for bbox in boxes:
            if bbox[2] - bbox[0] <= 0 or bbox[3] - bbox[1] <= 0:
                continue
            try:
                face = extract_face(bbox, frame)
            except:
                continue
            idx, score = inference(model, face, embeddings, names)
            frame = cv2.rectangle(
                frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)
            score = torch.Tensor.cpu(score).detach().numpy()
            if score >=  0.7:
                frame = cv2.putText(frame, names[idx] + '_{:.2f}'.format(score), 
                (bbox[0], bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 2, cv2.LINE_8)
            else:
                frame = cv2.putText(frame, 'Unknown', 
                (bbox[0], bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 2, cv2.LINE_8)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
