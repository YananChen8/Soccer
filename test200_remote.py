import time, os, sys
import cv2, torch
DATA = "/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/datasets/SoccerNetGS/test"
t0 = time.time()
n = 0
for i in range(1, 51):
    im = cv2.imread(f"{DATA}/SNGS-116/img1/{i:06d}.jpg")
    if im is not None:
        n += 1
print(f"frame_read: {n}/50 frames in {round(time.time()-t0,2)}s")
t1 = time.time()
x = torch.zeros(2000, 2000, device="cuda")
y = x @ x
torch.cuda.synchronize()
print(f"cuda_ok: {round(time.time()-t1,2)}s on {torch.cuda.get_device_name(0)}")
print("ALL_OK")
