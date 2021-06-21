import os
import cv2
import numpy as np

input_path = "data_224/"

_VIDEO_FRAMES = 79

_Sample_bgr = []
_Sample_flow = []


def main():
    for path in os.listdir(input_path):
        video = cv2.VideoCapture(input_path+path)
        frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        all_frames = []
        all_flow = []

        if frames < _VIDEO_FRAMES:
            for i in range(_VIDEO_FRAMES):
                video.set(cv2.CAP_PROP_POS_FRAMES, i % frames)
                all_frames.append(video.read()[1])
        else:
            step = int(frames/_VIDEO_FRAMES)
            for i in range(_VIDEO_FRAMES):
                video.set(cv2.CAP_PROP_POS_FRAMES, i*step)
                all_frames.append(video.read()[1])
        print('complete bgr: ' + input_path + path)

        prev = all_frames[0]
        prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        for frame_curr in range(_VIDEO_FRAMES):
            curr = all_frames[frame_curr]
            curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            flow = compute_TVL1(prev, curr)
            all_flow.append(flow)
            prev = curr
        print('complete flow: ' + input_path + path)

        _Sample_bgr.append(all_frames)
        _Sample_flow.append(all_flow)


def compute_TVL1(prev, curr, bound=15):
    """comput the TV-L1 optical flow."""
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow

if __name__ == "__main__":
    main()
    np.save("sample_bgr.npy", _Sample_bgr)
    np.save("sample_flow.npy", _Sample_flow)