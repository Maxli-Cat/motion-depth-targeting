import cv2
import numpy as np

HEIGHT, WIDTH = 720, 1280

if __name__ == "__main__":
    right_capture = cv2.VideoCapture(0)
    right_capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    right_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    left_capture = cv2.VideoCapture(1)
    left_capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    left_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    video_write = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M',"J","P","G"), 10, (WIDTH*2, HEIGHT))
    _, left_delay_frame = left_capture.read()

    while True:
        _, right = right_capture.read()
        left = left_delay_frame
        _, left_delay_frame = left_capture.read()
        testframe = np.concatenate((left, right),axis=1)
        video_write.write(testframe)
        print(len(testframe[0]), len(testframe))
        cv2.imshow("window", testframe)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break