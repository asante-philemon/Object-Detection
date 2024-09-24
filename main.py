import cv2

#initialize the camera
cap = cv2.VideoCapture(0)

#opencv DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320,320), scale=1/255)

#load class lists
classes = []
with open("dnn_model/classes.txt","r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print("object list")
print(classes[0])
            

while True:
    #get frame
    ret, frame = cap.read()

    #object detection
    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores,bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id[0]]

        cv2.putText(frame, class_name,(x , y - 5), cv2.FONT_HERSHEY_PLAIN, 2)
        cv2.rectangle(frame, (x, y), (x + w, + h),(200,0, ), 3 )

    print("class ids", class_ids)
    print("scores", scores)
    print("bboxes", bboxes )










    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
