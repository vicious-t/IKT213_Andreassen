import cv2
import numpy as np


def save_image(filename, image):
    success = cv2.imwrite(filename, image)
    if not success:
        print(f"Uh oh, could not save {filename}")


def grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image


def smoothing(image, ksize, sigma_x):
    smoothed_image = cv2.GaussianBlur(
        image,
        ksize,
        sigma_x
    )
    return smoothed_image


def sobel_edge_detection(image):
    smoothed_image = smoothing(grayscale(image), (3, 3), 0)

    sobel_image = cv2.Sobel(
        smoothed_image,
        cv2.CV_64F,
        1,
        1,
        ksize=1
    )

    sobel_image = cv2.convertScaleAbs(sobel_image)
    return sobel_image


def canny_edge_detection(image, threshold_1, threshold_2):
    smoothed_image = smoothing(grayscale(image),
                               (3, 3),
                               0
                               )

    canny_image = cv2.Canny(
        smoothed_image,
        threshold_1,
        threshold_2
    )
    return canny_image


def template_match(image, template):
    grayscale_image = grayscale(image)
    grayscale_template_image = grayscale(template)

    w, h = grayscale_template_image.shape[::-1]

    result = cv2.matchTemplate(
        grayscale_image,
        grayscale_template_image,
        cv2.TM_CCOEFF_NORMED
    )

    threshold = 0.9
    locations = np.where(result >= threshold)

    for x, y in zip(*locations[::-1]):
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )
    return image


def resize(image, scale_factor: int, up_or_down: str):
    height, width = image.shape[:2]

    if up_or_down == "up":
        resized_image = cv2.pyrUp(
            image,
            dstsize=(
                width * scale_factor,
                height * scale_factor
            )
        )

    elif up_or_down == "down":
        resized_image = cv2.pyrDown(
            image,
            dstsize=(
                width // scale_factor,
                height // scale_factor
            )
        )
    else:
        raise ValueError("Uh oh, up_or_down must be 'up' or 'down'")
    return resized_image


def main():
    image = cv2.imread("images/lambo.png")
    if image is None:
        print("Uh oh, try again, image didn't load.")
        return

    shapes_image = cv2.imread("images/shapes.png")
    if shapes_image is None:
        print("Uh oh, try again, shapes image didn't load.")
        return

    template_image = cv2.imread("templates/shapes_template.jpg")
    if template_image is None:
        print("Uh oh, try again, template image didn't load.")
        return

    # Sobel edge detection
    sobel_image = sobel_edge_detection(image)
    save_image("solutions/sobel_image.png", sobel_image)

    # Canny edge detection
    canny_image = canny_edge_detection(image, 50, 50)
    save_image("solutions/canny_image.png", canny_image)

    # Template match
    matched_image = template_match(shapes_image, template_image)
    save_image("solutions/matched_image.png", matched_image)

    # Resizing
    resized_up_image = resize(image, 2, "up")
    save_image("solutions/resized_up_image.png", resized_up_image)

    resized_down_image = resize(image, 2, "down")
    save_image("solutions/resized_down_image.png", resized_down_image)


if __name__ == '__main__':
    main()
