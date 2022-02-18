import cv2
import numpy as np
import torchbox.transforms as T


def test_brightness(img):
    augs = T.AugmentationList([
        T.RandomBrightness(0.5, 1.5),
    ])
    print(augs)
    data = T.AugInput(img)
    transform = augs(data)
    img_t = data.image
    cv2.imwrite('z.png', img_t)


def test_flip(img, boxes):
    augs = T.AugmentationList([
        T.RandomFlip(0.5),
    ])
    print(augs)
    data = T.AugInput(img, boxes=boxes)
    transform = augs(data)
    img_t = data.image
    boxes_t = data.boxes
    cv2.imwrite('z.png', img_t)
    print(boxes_t)


def test_crop(img, boxes):
    augs = T.AugmentationList([
        T.RandomCrop('absolute', (640, 640)),
    ])
    print(augs)
    data = T.AugInput(img, boxes=boxes)
    transform = augs(data)
    img_t = data.image
    boxes_t = data.boxes
    cv2.imwrite('z.png', img_t)
    print(boxes_t)


def test_cutout(img):
    augs = T.AugmentationList([
        T.Cutout(0.5),
    ])
    print(augs)
    data = T.AugInput(img)
    transform = augs(data)
    img_t = data.image
    cv2.imwrite('z.png', img_t)


if __name__ == '__main__':
    img = cv2.imread('./img/test.jpg')
    boxes = np.array([[0, 0, 100, 100]]).astype(np.float32)
    # test_brightness(img)
    # test_flip(img, boxes)
    # test_crop(img, boxes)
    test_cutout(img)
