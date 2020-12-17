import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect

myFrameNumber = 10
cap = cv2.VideoCapture("/Users/gunhoro/Desktop/repo/cv-final-kalman-motion/data/hold_square.mov")

# get total number of frames
totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# check for valid frame number
if myFrameNumber >= 0 & myFrameNumber <= totalFrames:
    # set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES,myFrameNumber)
    cap.set(1, myFrameNumber)

ret, frame = cap.read()
# cv2.imshow("Video", frame)
# cv2.waitKey(0) 
# cv2.destroyAllWindows()
#template= frame[403:962,87:615]
template= cv2.imread("/Users/gunhoro/Desktop/repo/cv-final-kalman-motion/data/real/square_template.png")
#template= template[740:2060,480:1800]
#cv2.imwrite('square_template.png',template)d
# # cv2.imshow("template", template)
# # cv2.waitKey(0) 
# # cv2.destroyAllWindows()
#template_gray= cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
kp,des = sift.detectAndCompute(template,None)
#pts = np.float([kp[idx].pt for idx in len(kp)]).reshape(-1, 1, 2)

print(kp[0].pt)
img=cv2.drawKeypoints(template,kp,template)
cv2.imwrite('hold_square.png',img)
file1=open("kp_hold_square","w")
for i in range(len(kp)):
    file1.write(str(kp[i].pt)+"\n")

file1.close()


with open('des_hold_square.npy','wb') as f:
    np.save(f,des)


