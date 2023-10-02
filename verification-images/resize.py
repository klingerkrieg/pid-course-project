import cv2
import os

imgs = os.listdir(".")
for img_name in imgs:
    if img_name.endswith("png") or img_name.endswith("jpg"):
        name = img_name
        print(name)
        try:
            image = cv2.imread(name)
            res = cv2.resize(image, (1024,1024) ,interpolation = cv2.INTER_AREA)
            cv2.imwrite(name, res)
        except:
            print("Erro")