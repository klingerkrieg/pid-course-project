import os

f = open("verificadas.txt",'r')
txt = f.read()
f.close()

images = txt.split("\n")
for i in range(len(images)):
    images[i] = images[i][:30]

def clear_if_not_in(path, images):
    imgs = os.listdir(path)

    for im in imgs:
        if im.endswith("png") or im.endswith("jpg"):
            file_name = im[:30]
            if file_name not in images:
                os.unlink(path+"/"+im)

    

clear_if_not_in(".",images)

def count(path):
    imgs = os.listdir(path)
    return len(imgs)

print("Sobraram:", count(".") )