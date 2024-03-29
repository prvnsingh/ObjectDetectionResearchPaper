import math
import numpy as np
import cv2 as cv
import os
import shutil
import sys
from scipy import misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from skimage.io import imread
from skimage.transform import radon, rescale
import glob2



L = 256


def generate_image_frames(scriptdir):
    video_name = 'new vid.mp4'
    video_path=os.path.join(scriptdir,video_name)
    cam = cv.VideoCapture(video_path)
    try:

        # creating a folder named data
        if not os.path.exists(os.path.join(scriptdir,'data','input')):
            os.makedirs(os.path.join(scriptdir,'data','input'))

    # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')
    os.chdir(os.path.join(scriptdir,'data','input'))
    # frame
    currentframe = 0

    while (True):

        # reading from frame
        ret, frame = cam.read()

        if ret:
            # if video is still left continue creating images
            name = str(currentframe) + '.jpg'
            print('Creating...' + name)

            # writing the extracted images
            cv.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv.destroyAllWindows()

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

# Video Generating function


def generate_video(imgdir):
    image_folder = imgdir
    video_name = 'Radon1.avi'
    # os.chdir("C:\\Users\\prvns\\Downloads\\research paper\\video\\")

    images = [img for img in os.listdir(image_folder)
              if img.endswith(".jpg") or
              img.endswith(".jpeg") or
              img.endswith("png")]

    frame = cv.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv.VideoWriter(video_name, 0, 30, (width, height))

    # Appending the images to the video one by one
    for image in images:
        print (image)
        print(os.path.join(image_folder, image))
        video.write(cv.imread(os.path.join(image_folder, image)))

    # Deallocating memories taken for window creation
    cv.destroyAllWindows()
    video.release()
    print("Video Generated!")


def radon_transform(image, fullname, x, y):
    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 4.5))
    # ax1.imshow(image, cmap=plt.cm.Greys_r)
    # ax1.set_title("Original")

    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image[:,:,0], theta=theta, circle=True)
    ax2.set_title("Radon transform\n(Sinogram)")
    ax2.set_xlabel("Projection angle (deg)")
    ax2.set_ylabel("Projection position (pixels)")
    ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
               extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')
    return image, sinogram, theta
    # fig.tight_layout()
    # plt.show()

def  inverse_radon_transfrom(image,sinogram,theta):
    reconstruction_fbp = iradon(sinogram, theta=theta )
    # error = reconstruction_fbp - image
    # print(f"FBP rms reconstruction error: {np.sqrt(np.mean(error ** 2)):.3g}")

    # imkwargs = dict(vmin=-0.2, vmax=0.2)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),sharex=True, sharey=True)
    # ax1.set_title("Reconstruction\nFiltered back projection")
    # ax1.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
    # ax2.set_title("Reconstruction error\nFiltered back projection")
    # ax2.imshow(reconstruction_fbp - image, cmap=plt.cm.Greys_r, **imkwargs)
    return reconstruction_fbp
    # plt.show()

def process_imgdir(imgdir):
    resultdir = os.path.join(imgdir, 'result')
    inputdir = os.path.join(imgdir, 'input')
    if os.path.exists(resultdir):
        shutil.rmtree(resultdir)
    os.mkdir(resultdir)
    directory = list(os.listdir(inputdir))

    # video_name = 'ForthQuadrant.avi'
    # video = cv.VideoWriter(video_name, 0, 60, (160, 124))

    for fullname in range(0, len(directory)):
        fullname1 = str(fullname) + '.jpg'
        print(fullname1)
        # fullname2 = str(fullname + 20) + '.jpg'
        filepath1 = os.path.join(inputdir, fullname1)
        # filepath2 = os.path.join(inputdir, fullname2)
        if os.path.isfile(filepath1):
            basename = os.path.basename(filepath1)
            image1 = cv.imread(filepath1)
            # image2 = cv.imread(filepath2, cv.IMREAD_COLOR)
            # image = cv.subtract(image2, image1)
            image = image1

            # To Divide the Image into 3 vertical Equal Parts and processing the middle track
            imgheight = image.shape[0]

            # imgwidth = image.shape[1]
            #
            # y1 = 0
            # M = imgheight // 2
            # N = imgwidth // 2
            #
            # for y in range(0, imgheight, M):
            #     for x in range(0, imgwidth, N):
            #         y1 = y + M
            #         x1 = x + N
            #         tiles = image[y:y + M, x:x + N]
            #
            #         cv.rectangle(image, (x, y), (x1, y1), (0, 255, 0))
            #         if (x == 80) and (y == 64):
            #             cv.imwrite("results/" + str(fullname) + '-' + str(x) + '_' + str(y) + ".jpg", tiles)
            #             # testImg = cv.imread("results/" + str(fullname) + '-' + str(x) + '_' + str(y) + ".jpg", 0)
            #             # radon_transform(testImg, fullname, x, y)
            #             # print("Added" + "radon_transform_video/" + "RD" + str(fullname) + '-' + str(x) + '_' + str(y) + ".jpg - To video")
            #             # video.write(cv.imread("results/" + str(fullname) + '-' + str(x) + '_' + str(y) + ".jpg"))
            #
            # cv.imwrite("asas.png", image)
    # cv.destroyAllWindows()
    # video.release()
    # print("Video Generated!")
    #
    #
    #
    #         # ret, image = cv.threshold(image, 75, 255, cv.THRESH_BINARY)
    #         if len(image.shape) == 3 and image.shape[2] == 3:
    #             print('Processing %s ...' % basename)
    #         else:
    #             sys.stderr.write('Skipping %s, not RGB' % basename)
    #             continue
    #         # dehazed = get_scene_radiance(image)
    #         # side_by_side = np.concatenate((image1, dehazed), axis=1)
    #         # cv.imwrite(os.path.join(resultdir, basename), image)
    # generate_video(radonResult)
            imgwidth = image.shape[1]

            y1 = 0
            M = imgheight
            N = imgwidth // 3
            y=0
            x = N
            y1 = M
            x1 = x + N
            tiles = image[y:M, x:x + N]
            cv.rectangle(image, (x, y), (x1, y1), (0, 255, 0))
            os.chdir(resultdir)
            cv.imwrite(basename, tiles)
            testImg = cv.imread(basename)
            image,sinogram, theta = radon_transform(testImg)
            ret, image = cv.threshold(image, 75, 255, cv.THRESH_BINARY)
            reconstructed_image = inverse_radon_transfrom(image,sinogram,theta)
            cv.imwrite(basename,reconstructed_image)

            if len(image.shape) == 3 and image.shape[2] == 3:
                print('Processing %s ...' % basename)
            else:
                sys.stderr.write('Skipping %s, not RGB' % basename)
                continue
            # dehazed = get_scene_radiance(image)
            # side_by_side = np.concatenate((image1, dehazed), axis=1)
            # cv.imwrite(os.path.join(resultdir, basename), image)
    generate_video(resultdir)

def main():
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    # generate_image_frames(scriptdir)
    imgdir = os.path.join(scriptdir, 'data')
    process_imgdir(imgdir)


if __name__ == '__main__':
    main()
