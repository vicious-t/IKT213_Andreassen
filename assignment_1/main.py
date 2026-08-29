import cv2


def print_image_information(image):
    height, width, channels = image.shape
    size = image.size
    dtype = image.dtype
    print(f"\nHeight: {height}, "
          f"\nWidth: {width}, "
          f"\nChannels: {channels}, "
          f"\nSize: {size}, "
          f"\nData type: {dtype}")

def save_webcamera_information():
    webcamera = cv2.VideoCapture(0)

    if not webcamera.isOpened():
        print("Try again. Webcamera didn't open.")
        return

    fps = webcamera.get(cv2.CAP_PROP_FPS)
    height = webcamera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = webcamera.get(cv2.CAP_PROP_FRAME_WIDTH)

    with open("solutions/camera_outputs.txt", "w") as file:
        file.write(f"Height: {height}, "
                   f"\nWidth: {width}, "
                   f"\nFPS: {fps}, ")
    webcamera.release()

def main():
    image = cv2.imread("./images/iris-1.jpg")

    if image is None:
        print("Try again, image didn't load.")
        return

    print_image_information(image)
    save_webcamera_information()

if __name__ == "__main__":
        main()
