#pip install opencv-python
import cv2
from camera import getcamera
# Abrir la camara web
camera = getcamera()
cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

# Cargar el detector de rostros
faceClassif = cv2.CascadeClassifier("rostros.xml")
while True:
    # Tomar fotografia
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceClassif.detectMultiScale(gray,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(120,120),
                                        maxSize=(1000, 1000))
    #Dibujar un rectangulo en el rostro
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostrar en pantalla
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# cv2.waitKey(0)

