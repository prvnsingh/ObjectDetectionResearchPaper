# import cv2
# image1 = cv2.imread("frame912.jpg", cv2.IMREAD_COLOR)
# image2 = cv2.imread("frame972.jpg", cv2.IMREAD_COLOR)
# image = cv2.subtract(image2, image1)
# print(image.shape)
# # cv2.imwrite("result.jpg", image)
# # result = cv2.imread("result.jpg", cv2.IMREAD_COLOR)
# # ret, image = cv2.threshold(result, 150, 255, cv2.THRESH_BINARY)
# # print(image.shape)
# # cv2.imwrite("threshold.jpg", image)

# dirs = list(range(10))
# print(dirs)
# for index, i in enumerate(dirs[0:len(dirs):2]):
#     print("i", i)
#     print("i+index", i + index)
#     print("index", index)


l = ['frame0.jpg', 'frame2.jpg', 'frame10.jpg', 'frame100.jpg', 'frame1000.jpg', 'frame1001.jpg', 'frame1002.jpg', 'frame1003.jpg', 'frame1004.jpg', 'frame1005.jpg', 'frame1006.jpg', 'frame1007.jpg', 'frame1008.jpg', 'frame1009.jpg', 'frame101.jpg', 'frame1010.jpg', 'frame1011.jpg', 'frame1012.jpg', 'frame1013.jpg']
l.sort()
print(l)
