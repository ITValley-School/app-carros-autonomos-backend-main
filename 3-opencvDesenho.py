import cv2

webcam = cv2.VideoCapture(0)


while webcam.isOpened():
    ref, frame = webcam.read()

    if not ref:
        print("Failed to capture image")
        break

    #desenhar um retangulo
    cv2.rectangle(frame, (50,50), (800,600), (0,255,0), 2)

    cv2.imshow("Image from Webcam is showing", frame)
    cv2.putText(frame, "Adams Zago", (50, 650), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Aborting...")
        break

webcam.release()
cv2.destroyAllWindows()