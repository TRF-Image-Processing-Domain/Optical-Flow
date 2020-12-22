import cv2 as cv
import numpy as np

cap = cv.VideoCapture(r"Videos\video3.mp4")
ret, first_frame = cap.read()

prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

mask = np.zeros_like(first_frame)

mask[..., 1] = 255


while (cap.isOpened()):
    ret, frame = cap.read()
    cv.imshow("input", frame)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    """Calculating the dense optical flow by farne back method"""
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray,None,0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    #print("mag=",magnitude,"ang=",angle)

    """The angle (direction) of flow by hue is visualized"""
    mask[..., 0] = angle * 180 / np.pi / 2

    """The distance (magnitude) of flow by the value of HSV color representation"""
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)


    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

    cv.imshow("Dense optical flow", rgb)

    prev_gray = gray

    k = cv.waitKey(1)
    if k == 32:
        break

cap.release()
cv.destroyAllWindows()
