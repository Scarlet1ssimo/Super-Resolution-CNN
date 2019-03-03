import SR_export as exp
import glob
import sys
import os

def qwqwq(path):
    images_paths = glob.glob(path + "*.bmp")
    images_paths += glob.glob(path + "*.jpg")
    images_paths += glob.glob(path + "*.png")
    cnt = 0
    for path in images_paths:
        cnt += 1
        print(cnt)
        if os.path.getsize(path) < 1048576//2 and cnt > 124:
            exp.qwq(path, "sese/"+str(cnt)+".jpg")

if __name__ == "__main__":
    qwqwq(sys.argv[1])
