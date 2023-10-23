import os

dirs = os.listdir(".")

for dir in dirs:

    if (os.path.isdir(dir)):

        files = os.listdir(dir)

        for f in files:
            print(f)
            path = dir+"/"+f
            os.rename(path,path.replace(".png.png",".png"))