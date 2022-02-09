import math
import numbers

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters.thresholding import (
    threshold_isodata,
    threshold_li,
    threshold_local,
    threshold_mean,
    threshold_minimum,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)
from skimage.morphology import erosion, dilation

from get_orientation import get_orientation
from unsharp_mask import unsharp_mask


def write_sharpened(img):
    for i in range(3, 8):
        sharpened = unsharp_mask(img, amount=i)
        cv2.imwrite(f"out/sharpened/sharpened{i}.png", sharpened)
        plt.figure()
        plt.hist(sharpened.ravel(), bins=256)
        plt.savefig(f"out/sharpened/hist/sharpened{i}-hist.png")


def binary_threshold(img, t):
    im = img <= t
    return np.uint8([np.uint8([(255 if x else 0) for x in row]) for row in im])


def write_threshold(img):
    thresholds = [
        threshold_isodata(img),
        threshold_li(img),
        threshold_local(img, block_size=25, offset=32),
        threshold_mean(img),
        threshold_minimum(img),
        threshold_otsu(img),
        threshold_triangle(img),
        threshold_yen(img),
    ]

    for i in range(len(thresholds)):
        t = thresholds[i]
        im = binary_threshold(img, t)
        cv2.imwrite(f"out/threshold/threshold{i}.png", im)

        if isinstance(t, numbers.Number):
            plt.figure()
            plt.hist(img.ravel(), bins=256)
            plt.axvline(t, color="r")
            plt.savefig(f"out/threshold/hist/threshold{i}-hist.png")
        else:
            data = t.astype(np.float64)
            data = 255 * data
            cv2.imwrite(f"out/threshold/hist/threshold{i}-hist.png", data.astype(np.uint8))


def apply_erosion(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return erosion(img, kernel)


def apply_dilation(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    return dilation(img, kernel)


def remove_small_components(img, min_size):
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(img)
    # connectedComponentsWithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time
    # we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # your answer image
    out_img = img.copy()
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] < min_size:
            out_img[output == i + 1] = 0

    return out_img


def label_components(img):
    _, components = cv2.connectedComponents(img)

    # Map component labels to hue val
    label_hue = np.uint8(179 * components / np.max(components))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    return labeled_img


def put_text(src, text, org):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)
    thickness = 1
    line_type = 2

    cv2.putText(img=src,
                text=text,
                org=org,
                fontFace=font,
                fontScale=font_scale,
                color=font_color,
                thickness=thickness,
                lineType=line_type,
                )


def get_text_org(contour):
    x, y, w, h = cv2.boundingRect(contour)
    ext = np.array([w, h]) // 2
    pos = [x, y]
    org = (pos + ext)
    org[0] -= 23
    org[1] += h // 2 + 20

    return org


def put_area_text(src, contour):
    text = str(cv2.contourArea(contour))

    put_text(src, text, get_text_org(contour))


def get_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * math.pi * (area / perimeter ** 2)
    return circularity


def put_circularity_text(src, contour):
    text = str(round(get_circularity(contour), 2))

    put_text(src, text, get_text_org(contour))


def put_compactness_text(src, contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    compactness = area / perimeter
    text = str(round(compactness, 2))

    put_text(src, text, get_text_org(contour))


def main():
    img = cv2.imread("input.png", cv2.IMREAD_GRAYSCALE)
    img = unsharp_mask(img, amount=6)
    img = binary_threshold(img, threshold_local(img, block_size=25, offset=32))
    img = apply_erosion(img)
    img = apply_dilation(img)
    img = remove_small_components(img, min_size=115)

    # img = cv2.imread("out/small_components_removed.png", cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # cv2.imshow("contours", img)

    # img = label_components(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # for c in contours:
    #     put_contour_text(img, c)

    for i, c in enumerate(contours):
        # cv2.drawContours(img, contours, i, (0, 0, 255), 1)
        if get_circularity(c) > 0.7:
            cv2.drawContours(img, contours, i, (0, 0, 255), 2)

    for c in contours:
        # put_compactness_text(img, c)
        put_circularity_text(img, c)
        # put_area_text(img, c)

    for c in contours:
        get_orientation(c, img)

    # cv2.imwrite("out/circularity.png", img)
    cv2.imshow("contour", img)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
