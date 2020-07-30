import cv2
import argparse

from os import path
from Pipeline import *
from LaneMemory import LaneMemory


def main():
    parser = argparse.ArgumentParser("Do stuff")
    parser.add_argument("filename")

    args = parser.parse_args()
    filename, ext = path.splitext(args.filename)

    cap = cv2.VideoCapture(filename + ext)
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter("output/" + filename + ".mp4", fourcc,
                          fps, (frame_width, frame_height), True)

    pipeline = Pipeline(height=frame_height)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret == True:
            output = pipeline(frame)
            cv2.imshow("Output", output)
        else:
            cap.release()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
