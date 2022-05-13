import cv2
import matplotlib.pyplot as plt

def plot_box(image, boxes, labels, color=(0, 255, 0), thickness=2, fontScale=.5):
    assert len(boxes) == len(labels)
    for i in range(len(boxes)):
        box = boxes[i]
        label = labels[i]
        w = box[2]-box[0]
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness)
        cv2.putText(image, label, (box[0]+10, box[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness)

    plt.imshow(image)