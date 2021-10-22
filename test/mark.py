

f = open("./blood_smear_1_infoText.txt","r")
txt = f.read()
f.close()

lines = txt.split("\n")
del lines[0]

for line in lines:
    print(line.split(" "))



