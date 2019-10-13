import cv2
image1 = cv2.imread("frame912.jpg", cv2.IMREAD_COLOR)
image2 = cv2.imread("frame972.jpg", cv2.IMREAD_COLOR)
image = cv2.subtract(image2, image1)
print(image)
cv2.imwrite("result.jpg", image)
result = cv2.imread("result.jpg", cv2.IMREAD_COLOR)
ret, image = cv2.threshold(result, 150, 1, cv2.THRESH_BINARY)
cv2.imwrite("threshold.jpg", image)
