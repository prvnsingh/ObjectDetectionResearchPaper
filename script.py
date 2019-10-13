# import os
# import cv2
# cam = cv2.VideoCapture("C:\\Users\\prvns\\Downloads\\research paper\\vid.mp4")
# try:

#     # creating a folder named data
#     if not os.path.exists('data'):
#         os.makedirs('data')

# # if not created then raise error
# except OSError:
#     print('Error: Creating directory of data')

# # frame
# currentframe = 0

# while(True):

#     # reading from frame
#     ret, frame = cam.read()

#     if ret:
#         # if video is still left continue creating images
#         name = './data/frame' + str(currentframe) + '.jpg'
#         print('Creating...' + name)

#         # writing the extracted images
#         cv2.imwrite(name, frame)

#         # increasing counter so that it will
#         # show how many frames are created
#         currentframe += 1
#     else:
#         break

# # Release all space and windows once done
# cam.release()
# cv2.destroyAllWindows()

import math
import numpy as np
import cv2 as cv
import os
import shutil
import sys

L = 256


def get_dark_channel(img, *, size):
    """Get dark channel for an image.
    @param img: The source image.
    @param size: Patch size.
    @return The dark channel of the image.
    """
    minch = np.amin(img, axis=2)
    box = cv.getStructuringElement(cv.MORPH_RECT, (size // 2, size // 2))
    return cv.erode(minch, box)


def get_atmospheric_light(img, *, size, percent):
    """Estimate atmospheric light for an image.
    @param img: the source image.
    @param size: Patch size for calculating the dark channel.
    @param percent: Percentage of brightest pixels in the dark channel
    considered for the estimation.
    @return The estimated atmospheric light.
    """
    m, n, _ = img.shape

    flat_img = img.reshape(m * n, 3)
    flat_dark = get_dark_channel(img, size=size).ravel()
    count = math.ceil(m * n * percent / 100)
    indices = np.argpartition(flat_dark, -count)[:-count]

    return np.amax(np.take(flat_img, indices, axis=0), axis=0)


def get_transmission(img, atmosphere, *, size, omega, radius, epsilon):
    """Estimate transmission map of an image.
    @param img: The source image.
    @param atmosphere: The atmospheric light for the image.
    @param omega: Factor to preserve minor amounts of haze [1].
    @param radius: (default: 40) Radius for the guided filter [2].
    @param epsilon: (default: 0.001) Epsilon for the guided filter [2].
    @return The transmission map for the source image.
    """
    division = np.float64(img) / np.float64(atmosphere)
    raw = (1 - omega * get_dark_channel(division, size=size)).astype(np.float32)
    return cv.ximgproc.guidedFilter(img, raw, radius, epsilon)


def get_scene_radiance(img,
                       *,
                       size=15,
                       omega=0.95,
                       trans_lb=0.1,
                       percent=0.1,
                       radius=40,
                       epsilon=0.001):
    """Get recovered scene radiance for a hazy image.
    @param img: The source image to be dehazed.
    @param omega: (default: 0.95) Factor to preserve minor amounts of haze [1].
    @param trans_lb: (default: 0.1) Lower bound for transmission [1].
    @param size: (default: 15) Patch size for filtering etc [1].
    @param percent: (default: 0.1) Percentage of pixels chosen to compute atmospheric light [1].
    @param radius: (default: 40) Radius for the guided filter [2].
    @param epsilon: (default: 0.001) Epsilon for the guided filter [2].
    @return The final dehazed image.
    """
    atmosphere = get_atmospheric_light(img, size=size, percent=percent)
    trans = get_transmission(img, atmosphere, size=size, omega=omega, radius=radius, epsilon=epsilon)
    clamped = np.clip(trans, trans_lb, omega)[:, :, None]
    img = np.float64(img)
    return np.uint8(((img - atmosphere) / clamped + atmosphere).clip(0, L - 1))


def process_imgdir(imgdir):
    resultdir = os.path.join(imgdir, 'results')
    inputdir = os.path.join(imgdir, 'inputs')
    shutil.rmtree(resultdir)
    os.mkdir(resultdir)
    directory = list(os.listdir(inputdir))
    for index, fullname in enumerate(directory):
        filepath1 = os.path.join(inputdir, fullname)
        filepath2 = os.path.join(inputdir, directory[index + 1])
        filepath3 = os.path.join(inputdir, directory[index + 2])
        if os.path.isfile(filepath1 and filepath2 and filepath3):
            basename = os.path.basename(filepath1)
            image1 = cv.imread(filepath1, cv.IMREAD_COLOR)
            # image2 = cv.imread(filepath2, cv.IMREAD_COLOR)
            image3 = cv.imread(filepath3, cv.IMREAD_COLOR)
            image = cv.subtract(image3, image1)
            # image_2 = cv.subtract(image3, image2)
            # image = cv.subtract(image_2, image_1)
            # print(image1, image2)
            # print("Image 3")
            # print(image3)

            # print("Image 2 - Image 1", image_1)
            # print(image_1)

            # print("Image 3 - Image 2")
            print(image)
            # ret, image = cv.threshold(image, 0, 0, cv.THRESH_BINARY)
            ret, image = cv.threshold(image, 150, 1, cv.THRESH_BINARY)
            print(image)
            if len(image.shape) == 3 and image.shape[2] == 3:
                print('Processing %s ...' % basename)
            else:
                sys.stderr.write('Skipping %s, not RGB' % basename)
                continue
            dehazed = get_scene_radiance(image)
            side_by_side = np.concatenate((image1, dehazed), axis=1)
            cv.imwrite(os.path.join(resultdir, basename), side_by_side)


def main():
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    imgdir = os.path.join(scriptdir, 'data')
    print(imgdir)
    process_imgdir(imgdir)


if __name__ == '__main__':
    main()
