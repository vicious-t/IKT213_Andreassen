import cv2
import numpy as np

def get_image_info(image):
    height = image.shape[0]
    width = image.shape[1]

    if len(image.shape) == 3:
        channels = image.shape[2]
    else:
        channels = 1

    return {
        "height": height,
        "width": width,
        "channels": channels,
        "size": image.size,
        "dtype": image.dtype
    }

def save_image(filename, image):
    success = cv2.imwrite(filename, image)

    if not success:
        print(f"Uh oh, could not save {filename}")


def padding (image, border_width):
    padded_image = cv2.copyMakeBorder(
        image,
        border_width,
        border_width,
        border_width,
        border_width,
        cv2.BORDER_REFLECT
    )
    return padded_image

def crop(image, x_0, x_1, y_0, y_1):
    cropped_image = image[y_0:y_1, x_0:x_1]
    return cropped_image

def resize(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    return resized_image

def copy(image, emptyPictureArray):
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            emptyPictureArray[y, x] = image[y, x]

    return emptyPictureArray

def grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image

def hue_shifted(image, emptyPictureArray, hue):
    height = image.shape[0]
    width = image.shape[1]

    for y in range(height):
        for x in range(width):
            for channel in range(3):
                shifted_value = int(image[y,x,channel]) + hue

                if shifted_value > 255:
                    shifted_value = 255
                elif shifted_value < 0:
                    shifted_value = 0

                emptyPictureArray[y, x, channel] = shifted_value
    return emptyPictureArray

def smoothing(image):
    smoothed_image = cv2.GaussianBlur(
        image,
        (15, 15),
        0
    )

    return smoothed_image

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated_image = cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE
        )
    elif rotation_angle == 180:
        rotated_image = cv2.rotate(
            image,
            cv2.ROTATE_180
        )
    else:
        rotated_image = image

    return rotated_image

def main():
    image = cv2.imread("./images/iris-1.png")

    if image is None:
        print("Uh oh, try again, image didn't load.")
        return

    image_info = get_image_info(image)

# Padded image
    padded_image = padding(image, 100)
    save_image("solutions/padded_image.jpg", padded_image)

# Cropped image
    cropped_image = crop(
        image,
        200, image_info["width"] - 130,
        200, image_info["height"] - 130
    )
    save_image("solutions/cropped_image.jpg", cropped_image)

# Resized image
    resized_image = resize(image, 200, 200)
    save_image("solutions/resized_image.jpg", resized_image)

# Manual image copy
    emptyPictureArray = np.zeros((
                                image_info["height"],
                                image_info["width"], 3), np.uint8)
    copied_image = copy(image, emptyPictureArray)
    save_image("solutions/copied_image.jpg", copied_image)

# Grayscale
    grayscale_image = grayscale(image)
    save_image("solutions/grayscale_image.jpg", grayscale_image)

# HSV
    hsv_image = hsv(image)
    save_image("solutions/hsv_image.jpg", hsv_image)

# Color shift
    emptyPictureArray = np.zeros((
                                image_info["height"],
                                image_info["width"], 3),np.uint8)
    hue_shifted_image = hue_shifted(image, emptyPictureArray, 50)
    save_image("solutions/hue_shifted_image.jpg", hue_shifted_image)

# Smoothing
    smoothed_image = smoothing(image)
    save_image("solutions/smoothed_image.jpg", smoothed_image)

# Rotation
    rotated_image = rotation(image, 180)
    save_image("solutions/rotated_image.jpg", rotated_image)

if __name__ == '__main__':
    main()
