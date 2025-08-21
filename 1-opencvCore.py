import cv2

webcam = cv2.VideoCapture(0)

while webcam.isOpened():
    ref, frame = webcam.read()

    if not ref:
        print("Failed to capture image")
        break

    cv2.imshow("Image from Webcam is showing", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Aborting...")
        break

webcam.release()
cv2.destroyAllWindows()