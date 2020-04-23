import cv2
import numpy as np
import imutils
from sklearn.metrics import pairwise

video = cv2.VideoCapture(0)
num_of_iteration = 0
background = None
accumilate_weight_alpha = 0.5
top, right, bottom, left = 0, 0, 325, 280

while True:
    ret, frame = video.read()
    #frame = imutils.resize(frame, width=700)
    frame = cv2.flip(frame, 1)
    hand = frame[top:bottom, right:left]
    frame_gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (9,9), 0)
    if num_of_iteration < 30:
        if background is None:
            background = frame_gray.copy().astype("float")
        else:
            background = cv2.accumulateWeighted(frame_gray, background, accumilate_weight_alpha)
    else:
        normalize_background = background.copy().astype("uint8")
        foreground = cv2.absdiff(normalize_background, frame_gray)
        thresholded = cv2.threshold(foreground, 25, 250, cv2.THRESH_BINARY)[1]
        contours, hierarchy = cv2.findContours(thresholded,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            hand_contour = max(contours, key=cv2.contourArea)
            cv2.imshow('hand', thresholded)
            cv2.drawContours(frame, [hand_contour+(right, top)], -1, (0, 255, 255))
            hull_list = []
            for i in range(len(contours)):
                hull = cv2.convexHull(contours[i])
                hull_list.append(hull)
            # Draw contours + hull results

            for i in range(len(contours)):
                color = (255, 0, 0)
                cv2.drawContours(frame, hull_list, i, color)

            hand_convex_hull = cv2.convexHull(hand_contour)

            # find the most extreme points in the convex hull
            max_top = tuple(hand_convex_hull[hand_convex_hull[:, :, 1].argmin()][0])
            max_bottom = tuple(hand_convex_hull[hand_convex_hull[:, :, 1].argmax()][0])
            max_left = tuple(hand_convex_hull[hand_convex_hull[:, :, 0].argmin()][0])
            max_right = tuple(hand_convex_hull[hand_convex_hull[:, :, 0].argmax()][0])

            palm_X = int((max_left[0] + max_right[0]) / 2)
            palm_Y = int((max_top[1] + max_bottom[1]) / 2)

            distance = pairwise.euclidean_distances([(palm_X, palm_Y)], Y=[max_left, max_right, max_top, max_bottom])[0]
            maximum_distance = distance[distance.argmax()]

            radius = int(0.8 * maximum_distance)
            circumference = (2 * np.pi * radius)
            circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
            cv2.circle(circular_roi, (palm_X, palm_Y), radius, 255, 1)
            circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
            (cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            count = 0
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                if ((palm_Y + (palm_Y * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
                    count += 1

            cv2.putText(frame, '# of fingers: ' +str(count), (10, 450), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    num_of_iteration+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()