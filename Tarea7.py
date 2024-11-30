# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:48:34 2019

@author: GabrielAsus
"""
import cv2 as cv

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv.CascadeClassifier(cascPath)
gatitoO=cv.imread("catFilter.png",cv.IMREAD_UNCHANGED)

#obtener acceso a la webcam
video_capture = cv.VideoCapture(0,cv.CAP_DSHOW)
codigo = cv.VideoWriter_fourcc(*'mp4v')
#Guardar el video
salida = cv.VideoWriter('FiltroTiktok.mp4',codigo,40,(640,480))

anterior = 0
if not video_capture.isOpened():
        print('No se pudo acceder a la camara')
else:
    while True:
        #revisar si ya puedo leer imagenes de la camara
        ret, frame = video_capture.read()
        frame=cv.flip(frame,1)
        imagenGrises = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        video = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)
        faces = faceCascade.detectMultiScale(
            imagenGrises,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        #por cada cara detectada pintar un cuadro
        for (x, y, w, h) in faces:
            alto = int(h*.9)
            gatitoF = cv.resize(gatitoO, (w,alto))
            colDeseada = x
            filaDeseada = y - int(h*0.2)
            for i in range(gatitoF.shape[0]):
                for j in range(gatitoF.shape[1]):
                    if(gatitoF[i,j][3]!=0):
                       video[i+filaDeseada,j+colDeseada]=gatitoF[i,j]
                    else:
                        pass
        # Mostrar la deteccion
        cv.imshow('Video', video)
        
        salida.write(cv.cvtColor(video, cv.COLOR_BGRA2BGR))
        #se motraran las caras mientra no presionemos la tecla q
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    #liberar la camara
    video_capture.release()
    salida.release()
    #cerrar todas las ventanas
    cv.destroyAllWindows()

