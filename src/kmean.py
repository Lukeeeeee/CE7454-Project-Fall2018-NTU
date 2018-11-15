import cv2
import numpy as np


class Segment:
    def __init__(self, segments=5):
        # define number of segments, with default 5
        self.segments = segments

    def kmeans(self, image):
        image = cv2.GaussianBlur(image, (7, 7), 0)
        vectorized = image.reshape(-1, 3)
        vectorized = np.float32(vectorized)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(vectorized, self.segments, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        res = center[label.flatten()]
        segmented_image = res.reshape((image.shape))
        return label.reshape((image.shape[0], image.shape[1])), segmented_image.astype(np.uint8)

    def extractComponent(self, image, label_image, label):
        component = np.zeros(image.shape, np.uint8)
        component[label_image == label] = image[label_image == label]
        return component


def get_stats(res, mask):
    overlap = res * mask  # Logical AND
    union = res + mask  # Logical OR

    IOU = overlap.sum() / float(union.sum())
    return IOU


def extract_mask(image, mask):
    segment_num = 5
    seg = Segment(segment_num)
    label, result = seg.kmeans(image)
    top_IOU = -1
    best_layer = 0
    clabel = -1
    for x in range(segment_num):
        result = seg.extractComponent(image, label, x)
        res_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        res_gray_signed = np.sign(res_gray)

        IOU = get_stats(res_gray_signed, mask)

        if IOU > top_IOU:
            top_IOU = IOU
            clabel = x
            best_layer = res_gray
    return best_layer
