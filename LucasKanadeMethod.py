import numpy as np
import cv2

cap = cv2.VideoCapture("Videos/1.mp4")

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 300,
                       qualityLevel = 0.01,
                       minDistance = 5,
                       )


# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take first frame and find corners in it
ret, old_frame_1 = cap.read()
old_frame=cv2.resize(old_frame_1,(600,400))
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)


while(1):
    left = 0
    right = 0
    down = 0
    up = 0
    ret,frame_o = cap.read()
    frame=cv2.resize(frame_o,(600,400))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculating optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params )

    if p1 is None:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Selecting good points
    good_new = p1[st==1]
    good_old = p0[st==1]


    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()

        if (a - c > 2):
            #cv2.putText(frame, "Right", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            frame = cv2.line(frame, (a, b), (c, d), (0, 0, 255), 2)
            right+=1
        elif (a - c < -2):
            #cv2.putText(frame, "Left", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            frame = cv2.line(frame, (a, b), (c, d), (0, 255, 255), 2)
            left+=1
        if (b-d>2):
            #cv2.putText(frame, "DOWN", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frame = cv2.line(frame, (a, b), (c, d), (255, 255, 255), 2)
            down+=1
        elif (b-d<-2):
            #cv2.putText(frame, "UP", (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            frame = cv2.line(frame, (a, b), (c, d), (0, 0, 0), 2)
            up+=1

    if up>10:
        cv2.putText(frame, "Up", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    if down > 10:
        cv2.putText(frame, "Down", (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if right > 10:
        cv2.putText(frame, "Right", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if left > 10:
        cv2.putText(frame, "Left", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('frame', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()