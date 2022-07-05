import cv2

for i in range(1,31):
    name = 'blood_smear_%d.jpg' %i
    print(name)
    try:
        image = cv2.imread(name)
        res = cv2.resize(image, (640,480) ,interpolation = cv2.INTER_AREA)
        cv2.imwrite('blood_smear_%d.jpg' % i, res)
    except:
        print("Erro")