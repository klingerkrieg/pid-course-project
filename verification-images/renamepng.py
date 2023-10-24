import os

files = os.listdir("malaria-app")

for f in files:
    print(f)
    path = "./malaria-app/"+f
    os.rename(path,path.replace("_segmented.png",".png"))