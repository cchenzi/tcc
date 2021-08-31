import cv2


def prepare_composition(
    blurry, blurry_value, thresholding, contrast, filtering, bilateral_filtering_sigma
):
    funcs = []
    if blurry:
        funcs.append((remove_noise, {"kernel": blurry_value}))
    if thresholding:
        funcs.append((apply_otsu_thresholding, {}))
    if contrast:
        funcs.append((histogram_equalization, {}))
    if filtering:
        funcs.append((bilateral_filtering, {"sigma": bilateral_filtering_sigma}))
    return funcs


def compose_functions(image, funcs):
    for (func, args) in funcs:
        image = func(image, **args)
    return image


def remove_noise(image, kernel):
    print("Doing Gaussian Blur")
    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def apply_otsu_thresholding(image):
    print("Doing Thresholding with Otsu")
    return cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)[1]


def histogram_equalization(image):
    print("Doing Histogram Equalization")
    return cv2.equalizeHist(image)


def bilateral_filtering(image, sigma):
    print("Doing Bilateral Filtering")
    return cv2.bilateralFilter(image, 9, sigma, sigma)


def resize_img(img, scale_percent):
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
