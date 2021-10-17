import cv2
from queue import Queue, SimpleQueue
import time

if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    _, oldframe = cap.read()
    _, lastframe = cap.read()
    frame_queue = SimpleQueue()
    recent_frames = []
    shorter_recent_frames = []


    while True:
        tick = time.time()
        ret, frame = cap.read()

        ofgray = cv2.GaussianBlur(cv2.cvtColor(oldframe, cv2.COLOR_BGR2GRAY), (21,21), 0)
        nfgray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21,21), 0)

        nieve = cv2.absdiff(ofgray, nfgray)

        threshed = cv2.threshold(nieve, 30,255,cv2.THRESH_BINARY)[1]
        blurred = cv2.GaussianBlur(threshed, (21,21), 0)

        colored = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

        recent_frames.append(colored)
        shorter_recent_frames.append(colored)
        boxy = colored.copy()

        for f in recent_frames:
            colored = cv2.add(colored, f)
        for f in shorter_recent_frames:
            boxy = cv2.add(boxy, f)


        boxy = cv2.cvtColor(boxy, cv2.COLOR_BGR2GRAY)
        boxy = cv2.threshold(boxy, 30,255, cv2.THRESH_BINARY)[1]

        overlayed = cv2.add(frame, colored)
        underlayed = cv2.min(frame, colored)

        while len(recent_frames) > 40:
            recent_frames.pop(0)

        while len(shorter_recent_frames) > 10:
            shorter_recent_frames.pop(0)

        dispframe = frame.copy()

        cnts, _ = cv2.findContours(boxy.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts.sort(key=cv2.contourArea, reverse=True)

        for i, contour in enumerate(cnts):
            if cv2.contourArea(contour) < 1000:
                continue
            motion = 1
            (x, y, w, h) = cv2.boundingRect(contour)
            if i == 0:
                cv2.rectangle(dispframe, (x, y), (x + w, y + h), (0, 0, 255), 3)
            else:
                cv2.rectangle(dispframe, (x, y), (x + w, y + h), (0, 255, 0), 3)
            print(x,y,w,h)

        cv2.imshow("newframe", dispframe)
        cv2.imshow("diff", threshed)
        cv2.imshow("overlay", underlayed)
        cv2.imshow("boxy?", boxy)


        oldframe = frame
        #frame_queue.put(frame)
        #if frame_queue.qsize() > 3:
        #    oldframe = frame_queue.get()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        while time.time() - tick < (1/20):
            pass