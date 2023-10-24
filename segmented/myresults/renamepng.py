import os

files = os.listdir("standalone")

for f in files:
    print(f)
    path = "./standalone/"+f
    os.rename(path,path.replace(".png",""))