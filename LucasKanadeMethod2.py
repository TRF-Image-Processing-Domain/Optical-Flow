import numpy as np
import cv2

cap = cv2.VideoCapture("Videos/2.mp4")

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

    flag=0


    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()

        '''if (a - c > 2):
            if flag == 0:
                cv2.putText(frame, "Right", (int(a+50), b), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                flag += 1
            frame = cv2.line(frame, (a, b), (c, d), (0, 255, 0), 2)
        if (a - c < -2):
            if flag == 0:
                cv2.putText(frame, "Left", (int(a+50), b), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                flag += 1
            frame = cv2.line(frame, (a, b), (c, d), (0, 255, 0), 2)'''
        if (b-d>2):
            if flag==0:
                cv2.putText(frame, "DOWN", (a, b), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                flag+=1
            frame = cv2.line(frame, (a, b), (c, d), (0, 255, 0), 2)
        if (b-d<-2):
            if flag==0:
                cv2.putText(frame, "UP", (a, b), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                flag+=1
            frame = cv2.line(frame, (a, b), (c, d), (0,255,0), 2)
    cv2.imshow('frame', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()