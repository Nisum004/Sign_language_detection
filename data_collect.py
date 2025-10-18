import cv2
import os

data_dir = './data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

category = 3
dataset_size = 100

cap = cv2.VideoCapture(0)
for j in range(category):
    if not os.path.exists(os.path.join(data_dir, str(j))):
        os.makedirs(os.path.join(data_dir, str(j)))

    print(f"collecting data for class {str(j)}")

    done = False
    while not done:
        ret, frame = cap.read()
        cv2.putText(frame,'Press Q to start capturing data ! ', (100,100),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),2)
        cv2.imshow('frame',frame)

        if cv2.waitKey(25) & ord('q'):
            break
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame',frame)
        cv2.imwrite(os.path.join(data_dir, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1
    cap.release()
    cv2.destroyAllWindows()




