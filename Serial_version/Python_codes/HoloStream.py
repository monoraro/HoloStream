import cv2
import tkinter as tk
from tkinter import ttk, messagebox
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
import numpy as np
from numpy import asarray
import math as mt
from Funciones import *
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from tkinter import filedialog
import subprocess
import sys
import os
def tiro(holo,fx_0,fy_0,fx_tmp, fy_tmp,lamb,M,N,dx,dy,k,m,n):
    
    #Calculo de los angulos de inclinación

    theta_x=mt.asin((fx_0 - fx_tmp) * lamb /(M*dx))
    theta_y=mt.asin((fy_0 - fy_tmp) * lamb /(N*dy))

    #Creación de la fase asociada

    fase= np.exp(1j*k* ((mt.sin(theta_x) * m * dx)+ ((mt.sin(theta_y) * n * dy))))
    fase1=fase
    holo=holo*fase
    
    fase = np.angle(holo, deg=False)
    min_val = np.min(fase)
    max_val = np.max(fase)
    fase = (fase - min_val) / (max_val - min_val)
    threshold_value = 0.2
    fase = np.where(fase > threshold_value, 1, 0)
    value=np.sum(fase)
    return value, fase1

class CameraApp:
    def __init__(self, ventana):
        #creamos la ventana de la app e inicializamos
        self.ventana = ventana
        self.ini=0
        screen_width = ventana.winfo_screenwidth()
        screen_height = ventana.winfo_screenheight()
        
        # maximizamos el tamaño de la ventana
        ventana.geometry(f"{screen_width}x{screen_height}")
        
        # Optional: make the window resizable
        ventana.resizable(True, True)
        ventana.title("HoloStream")

        #Llamamos opencv para leer la cámara y leemos el tamaño de la imagen

        self.cap = cv2.VideoCapture(0)
        ret, frame = self.cap.read()

        # Verificar si la cámara fue detectada correctamente
        if not ret:
            messagebox.showerror("Error", "No camera were deteced. The app will close, to compensate a pre-recorded hologram open holoStream_compensation_interface.exe or to track a object open holoStream_tracking_interface.exe")
            ventana.destroy()  # Cierra la aplicación
            return
        ancho = cv2.CAP_PROP_FRAME_WIDTH
        largo = cv2.CAP_PROP_FRAME_HEIGHT

        #Aquí noté que el tamaño que lee está partido por la mitad la resolución de la camara en cada eje
        #Por lo cual hago un realease y después la vuelvo a llamar obligando la resolución adecuada
        self.cap.release()

        self.cap = cv2.VideoCapture(0)
        print("a")
        #Obligo la resolución
        self.cap.set(ancho, frame.shape[1]*2)  # Ancho deseado
        self.cap.set(largo,  frame.shape[0]*2)  # Alto deseado
        print(frame.shape[1])
        print(frame.shape[0])
        # Establecer el tema "cosmo" al inicio (el estilo blanquito)
        ventana.style = ttkb.Style(theme="cosmo")

        # Encabezado con línea superior completa
        self.encabezado_frame = ttk.Frame(ventana, padding="10")
        self.encabezado_frame.grid(row=0, column=0, columnspan=8, sticky="ew")
        self.titulo_label = ttk.Label(self.encabezado_frame, text="HoloStream", font=("Helvetica", 16, "bold"))
        self.titulo_label.grid(row=0, column=0, sticky="w")
        self.hline = ttk.Separator(ventana, orient="horizontal")
        self.hline.grid(row=1, column=0, columnspan=10, sticky="ew", pady=(5, 10))

        # Selector de tema en la esquina derecha superior (seleccionar temas)
        ttk.Label(ventana, text="Select theme:",font=("Helvetica", 14)).grid(row=0, column=8, padx=(100, 5), sticky="e")
        self.theme_selector = ttk.Combobox(ventana, values=["cosmo", "darkly", "vapor", "cyborg"], state="readonly",width= 0)
        self.theme_selector.grid(row=0, column=9, padx=(0, 20), sticky="ew")
        self.theme_selector.set(ventana.style.theme.name)  # Establecer el tema actual como seleccionado
        self.theme_selector.bind("<<ComboboxSelected>>", self.cambiar_tema)

        # Configuración de los labels para mostrar las imágenes
        self.label_original = ttk.Label(ventana)
        self.label_original.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=(0, 10))

        # Cuadro de texto uno (dx)
        ttk.Label(ventana, text="Definition of parameters", font=("Helvetica", 16, "bold")).grid(row=7, column=1, padx=(0, 5), pady=10,columnspan=3)
        ttk.Label(ventana, text="Delta x:",font=("Helvetica", 14)).grid(row=8, column=1, padx=(10, 5), pady=10,sticky="e")
        self.entry_param1 = tk.Entry(ventana,width= 5)
        self.entry_param1.grid(row=8, column=2, padx=(5, 0), pady=10, sticky="w")

        # Insertar el valor inicial de dx
        self.entry_param1.insert(0, "3.75")

        #Cuadro de texto dos (dy)
        ttk.Label(ventana, text="Delta y:",font=("Helvetica", 14)).grid(row=9, column=1, padx=(10, 5), pady=10, sticky="e")
        self.entry_param2 = tk.Entry(ventana,width= 5)
        self.entry_param2.grid(row=9, column=2, padx=(5, 0), pady=10, sticky="w")

        # Insertar el valor inicial de dy
        self.entry_param2.insert(0, "3.75")

        # Combobox con valores 1, 2, 3 y 4
        ttk.Label(ventana, text="Quadrant:",font=("Helvetica", 14)).grid(row=8, column=2, padx=(10, 5), pady=10, sticky="e")
        self.entry_param4 = ttk.Combobox(root, values=[1, 2, 3, 4], state="readonly", width=1)
        self.entry_param4.grid(row=8, column=3, padx=(0, 0), pady=10, sticky="w")

        #Cuadro de texto tres (lambda)
        ttk.Label(ventana, text="Wave length:",font=("Helvetica", 14)).grid(row=9, column=2, padx=(10, 5), pady=10, sticky="e")
        self.entry_param3 = tk.Entry(ventana,width= 5)
        self.entry_param3.grid(row=9, column=3, padx=(0, 0), pady=10, sticky="w")

        # Insertar el valor inicial de longitud de onda
        self.entry_param3.insert(0, "0.633")

        #self.mask_len = tk.Entry(ventana,width = 11)
        #self.mask_len.grid(row=8, column=3, padx=(0, 0), pady=10, sticky="e")

        # Colocar el placeholder 
        #self.mask_len.insert(0, "Mask radius")

        # Vincular los eventos para quitar y agregar el placeholder
        #self.mask_len.bind("<FocusIn>", self.clear_placeholder)
        #self.mask_len.bind("<FocusOut>", self.add_placeholder)
        # Boton de aplicar la configuración
        self.boton_aplicar = ttk.Button(ventana, text="Apply", command=self.aplicar_transformaciones)
        self.boton_aplicar.grid(row=8, column= 3, padx=10, pady=10, sticky='e')

        self.boton_stop = ttk.Button(ventana, text="Stop", command=self.stop_recording)
        self.boton_stop.grid(row=9, column= 3, padx=10, pady=10, sticky='e')
        self.boton_stop.config(state="disabled")  # Deshabilita el botón si no hay texto
        
        #Sección de grabar
        ttk.Label(ventana, text="Record", font=("Helvetica", 16, "bold")).grid(row=7, column=5, padx=(0, 0), pady=10,columnspan=2)
        
        ttk.Label(ventana, text="What do you want to record",font=("Helvetica", 14)).grid(row=8, column=5, padx=(10, 5), pady=10, sticky="e")
        self.tipo_grabar = ttk.Combobox(root, values=["Only the reconstruction", "Hologram and reconstruction"], state="readonly", width=20,font=("Helvetica", 14))
        self.tipo_grabar.grid(row=8, column=6, padx=(0, 0), pady=10, sticky="w")
        
        # Boton de grabar
        self.boton_grabar = ttk.Button(ventana, text="Record", command=self.toggle_grabacion)
        self.boton_grabar.grid(row=9, column=5, padx=10, pady=10,columnspan=2)
        self.boton_grabar.config(state="disabled")  # Deshabilita el botón si no hay texto


        #Ajusto las imagenes al tamaño de la pantalla meramente para visualización
        self.ancho = round(screen_width/3.05)
        self.largo = round(screen_height/3)
        # Imágenes iniciales de fondo negro 
        self.black_image1 = Image.new("RGB", (self.ancho, self.largo), "black")
        self.black_image2 = Image.new("RGB", (self.ancho, self.largo), "black")
        self.black_image3 = Image.new("RGB", (self.ancho, self.largo), "black")

        self.black_image1_tk = ImageTk.PhotoImage(self.black_image1)
        self.black_image2_tk = ImageTk.PhotoImage(self.black_image2)
        self.black_image3_tk = ImageTk.PhotoImage(self.black_image3)
        
        ttk.Label(ventana, text="Hologram", font=("Helvetica", 16, "bold")).grid(row=2, column=1, padx=(0, 0), pady=10,columnspan=3)
        self.label_original = ttk.Label(ventana, image=self.black_image1_tk)
        self.label_original.grid(row=3, column=1, columnspan=3,rowspan=3)

        ttk.Label(ventana, text="Fourier transform", font=("Helvetica", 16, "bold")).grid(row=2, column=4, padx=(0, 0), pady=10,columnspan=3)
        self.label_transformacion1 = ttk.Label(ventana, image=self.black_image2_tk)
        self.label_transformacion1.grid(row=3, column=4, columnspan=3,rowspan=3)

        ttk.Label(ventana, text="Phase map", font=("Helvetica", 16, "bold")).grid(row=2, column=7, padx=(0, 0), pady=10,columnspan=3)
        self.label_transformacion2 = ttk.Label(ventana, image=self.black_image3_tk)
        self.label_transformacion2.grid(row=3, column=7, columnspan=3,rowspan=3)

        # Crear un botón que abrirá el zoom de la fft para alinear
        ttk.Label(ventana, text="Zoom the fft for aligment", font=("Helvetica", 16, "bold")).grid(row=7, column=8, padx=(0, 0), pady=7)
        self.btn_abrir_ventana = ttk.Button(ventana, text="Zoom to the fourier transform", command=self.abrir_nueva_ventana)
        self.btn_abrir_ventana.grid(row=8, column=8)
        self.btn_abrir_ventana.config(state="disabled")  # Deshabilita el botón hasta comience la captura 
        self.hline = ttk.Separator(ventana, orient="horizontal")
        self.hline.grid(row=6, column=0, columnspan=10, sticky="ew", pady=(10, 0))

        #Sección para acceder a interfaz e tracking
        
        self.ini2=0
        self.grabacion = 0
        self.tipo = "a"
        self.mostrar_fotogramas()
        self.hline = ttk.Separator(ventana, orient="horizontal")
        self.hline.grid(row=10, column=0, columnspan=10, sticky="ew", pady=(10, 0))
        ttk.Label(ventana, text="Want to Track a hologram", font=("Helvetica", 16, "bold")).grid(row=11, column=8, padx=(0, 0), pady=7)
        self.btn_tracking = ttk.Button(ventana, text="Open Tracking interface", command=self.run_other_program)
        self.btn_tracking.grid(row=12, column=8)
        
        #Seccion para acceder a la compensación de hologramas
        ttk.Label(ventana, text="Want to compensate a hologram", font=("Helvetica", 16, "bold")).grid(row=11, column=5, padx=(0, 0), pady=10,columnspan=2)
        self.btn_tracking = ttk.Button(ventana, text="Open compensation interface", command=self.run_other_program_DSHPC)
        self.btn_tracking.grid(row=12, column=5,columnspan=2)
        
        self.ini2=0
        self.grabacion = 0
        self.tipo = "a"
        self.mostrar_fotogramas()

    # Función que limpia el texto cuando el usuario hace clic en el Entry
    def clear_placeholder(self,event):
        if self.mask_len.get() == "Mask radius":
            self.mask_len.delete(0, tk.END)

    # Función que vuelve a colocar el placeholder si el campo está vacío
    def add_placeholder(self,event):
        if self.mask_len.get() == "":
            self.mask_len.insert(0, "Mask radius")
    def cambiar_tema(self, event):
        nuevo_tema = self.theme_selector.get()
        self.ventana.style.theme_use(nuevo_tema)

    def stop_recording(self):
        self.ini = 0
        self.boton_stop.config(state="disabled")  # Deshabilita el botón si no hay texto
    
    def toggle_grabacion(self):
        if(self.grabacion==0):
            self.tipo_grabar.config(state="disabled")
            self.boton_grabar.config(text="Stop")
            self.grabacion=1
            self.tipo = self.tipo_grabar.get()
            self.recons = []
            self.origen = []
        else:
            
            self.boton_grabar.config(text="Saving files")
            #Si entra y ya grabó, necesita guardar, por lo tanto
            self.grabacion=0
            self.boton_grabar.config(state="disabled") 
            if(self.tipo=="Hologram and reconstruction"):
                hora = hora_y_fecha()
                video_origen = f'Captured_hologram_{hora}.avi'
                video_recons = f'Reconstructed_hologram_{hora}.avi'
                fps = 24
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                frame_width = self.origen[0].shape[1]
                frame_height = self.origen[0].shape[0]
                video = cv2.VideoWriter(video_origen, fourcc, fps, (frame_width, frame_height), isColor=False)
                frame_width = self.recons[0].shape[1]
                frame_height = self.recons[0].shape[0]
                video2 = cv2.VideoWriter(video_recons, fourcc, fps, (frame_width, frame_height), isColor=False)
                for i in range(len(self.origen)):
                    frame = self.origen[i]
                    video.write(frame)
                    frame = self.recons[i]
                    video2.write(frame)
                video.release()
                video2.release()
                cv2.destroyAllWindows()
            elif(self.tipo=="Only the reconstruction"):
                hora = hora_y_fecha()
                video_recons = f'Captured_hologram_{hora}.avi'
                fps = 24
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                frame_width = self.recons[0].shape[1]
                frame_height = self.recons[0].shape[0]
                video2 = cv2.VideoWriter(video_recons, fourcc, fps, (frame_width, frame_height), isColor=False)
                for i in range(len(self.recons)):
                    frame = self.recons[i]
                    video2.write(frame)
                video2.release()
                cv2.destroyAllWindows()
            self.boton_grabar.config(text="Record")
            self.boton_grabar.config(state="normal") 
            self.tipo_grabar.config(state="readonly")
    def capturar_fotograma(self):
        # Capturar un fotograma de la cámara
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame
        else: 
            return None
    
    #Esta transforamción es para el 1er frame
    def transformacion_1(self, frame):
        frame=ajuste_tamano1(frame)
        N, M = frame.shape
        self.N = N
        self.M = M
        k= 2*np.pi/self.lamb
        Fox= M/2
        Foy= N/2
        cuadrante=self.cuadrante
        self.ini = 2
        x = np.arange(0, M, 1)
        y = np.arange(0, N, 1)

        #Un meshgrid para la paralelizacion
        m, n = np.meshgrid(x - (M/2), y - (N/2))
        self.m = m
        self.n = n
        #Definiendo cuadrantes, solo calidad
        primer_cuadrante= np.zeros((N,M))
        primer_cuadrante[0:round(N/2 - (N*0.1)),round(M/2 + (M*0.1)):M]=1
        segundo_cuadrante= np.zeros((N,M))
        segundo_cuadrante[0:round(N/2 -(N*0.1)),0:round(M/2 - (M*0.1))]=1
        tercer_cuadrante= np.zeros((N,M))
        tercer_cuadrante[round(N/2 +(N*0.1)):N,0:round(M/2 - (M*0.1))]=1
        cuarto_cuadrante= np.zeros((N,M))
        cuarto_cuadrante[round(N/2 +(N*0.1)):N,round(M/2 + (M*0.1)):M]=1

        #Ahora a tirar fourier

        fourier1=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(frame)))
        if(cuadrante==1):
            self.mascara=primer_cuadrante
            fourier=primer_cuadrante*fourier1
        if(cuadrante==2):
            self.mascara=segundo_cuadrante
            fourier=segundo_cuadrante*fourier1
        if(cuadrante==3):
            self.mascara=tercer_cuadrante
            fourier=tercer_cuadrante*fourier1
        if(cuadrante==4):
            self.mascara=cuarto_cuadrante
            fourier=cuarto_cuadrante*fourier1
        a=amplitud(fourier)
        #Calculamos la amplitud del espectro de fourier

        #Encontramos la posición en x y y del máximo en el espacio de Fourier
        pos_max = np.unravel_index(np.argmax(a, axis=None), a.shape)
        #self.mascara = crear_mascara_circular(frame.shape,(pos_max[1],pos_max[0]),50)
        #Transformada insversa de fourier
        #fourier= fourier1*self.mascara
        fourier=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fourier)))




        #Ahora viene definición de parametros 

        paso=0.2
        fin=0
        fx=pos_max[1]
        fy=pos_max[0]

        G_temp=3
        suma_maxima=0


        while fin==0:
            temp=0
            frec_esp_x=np.arange(fx-paso*G_temp,fx+paso*G_temp,paso)
            frec_esp_y=np.arange(fy-paso*G_temp,fy+paso*G_temp,paso)
            for i in range(len(frec_esp_y)):
                for j in range(len(frec_esp_x)):
                    fx_temp=frec_esp_x[j]
                    fy_temp=frec_esp_y[i]
                    temp, faserina=tiro(fourier,Fox,Foy,fx_temp,fy_temp,self.lamb,M,N,self.dx,self.dy,k,m,n)
                    if(temp>suma_maxima):
                        x_max_out = fx_temp
                        y_max_out = fy_temp
                        suma_maxima = temp
            G_temp = G_temp - 1
            
            if(x_max_out == fx):
                if(y_max_out ==fy):
                    fin=1
            fx=x_max_out
            fy=y_max_out

        theta_x=mt.asin((Fox - fx) * self.lamb /(M*self.dx))
        theta_y=mt.asin((Foy - fy) * self.lamb /(N*self.dy))
        fase= np.exp(1j*k* ((mt.sin(theta_x) * m * self.dx)+ ((mt.sin(theta_y) * n * self.dy))))
        holo=fourier*fase
        fase = np.angle(holo, deg=False)
        min_val = np.min(fase)
        max_val = np.max(fase)
        fase = 255*(fase - min_val) / (max_val - min_val)
        fourier1 = np.log(np.abs(fourier1)+1)
        min_val = np.min(fourier1)
        max_val = np.max(fourier1)
        fourier1 = 255*(fourier1 - min_val) / (max_val - min_val)
        self.fx = fx
        self.fy = fy
        self.start_time = time.time()
        self.frame_count = 0
        self.boton_grabar.config(state="normal")  # Habilita los botones
        self.btn_abrir_ventana.config(state="normal")
        return fourier1,fase
    
    #Algoritmo para los demás frames
    def transformacion_2(self, frame):
        frame=ajuste_tamano1(frame)
        G_temp = 1
        paso=0.2
        k = 2*mt.pi/self.lamb
        U = asarray(frame)
        N = self.N
        M = self.M
        Fox = M/2
        Foy = N/2

        #Sera que esta se puede paralelizar
        fin=0
        fx = self.fx
        fy = self.fy
        fourier1=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(frame)))
        fourier= fourier1*self.mascara
        fourier1 = np.log(np.abs((fourier1)+1))
        fourier=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fourier)))
        suma_maxima=0
        while fin==0:
            temp=0
            frec_esp_x=np.arange(self.fx-paso*G_temp,self.fx+paso*G_temp,paso)
            frec_esp_y=np.arange(self.fy-paso*G_temp,self.fy+paso*G_temp,paso)
            for i in range(len(frec_esp_y)):
                for j in range(len(frec_esp_x)):
                    fx_temp=frec_esp_x[j]
                    fy_temp=frec_esp_y[i]
                    temp, faserina=tiro(fourier,Fox,Foy,fx_temp,fy_temp,self.lamb,M,N,self.dx,self.dy,k,self.m,self.n)
                    if(temp>suma_maxima):
                        x_max_out = fx_temp
                        y_max_out = fy_temp
                        suma_maxima = temp
            G_temp = G_temp - 1
            
            if(x_max_out == fx):
                if(y_max_out ==fy):
                    fin=1
            fx=x_max_out
            fy=y_max_out
        theta_x=mt.asin((Fox - fx) * self.lamb /(M*self.dx))
        theta_y=mt.asin((Foy - fy) * self.lamb /(N*self.dy))
        fase= np.exp(1j*k* ((mt.sin(theta_x) * self.m * self.dx)+ ((mt.sin(theta_y) * self.n * self.dy))))
        holo=fourier*fase
        fase = np.angle(holo, deg=False)
        min_val = np.min(fase)
        max_val = np.max(fase)
        fase = 255*(fase - min_val) / (max_val - min_val)
        min_val = np.min(fourier1)
        max_val = np.max(fourier1)
        fourier1 = 255*(fourier1 - min_val) / (max_val - min_val)
        self.fx = fx
        self.fy = fy
        if(self.tipo=="Hologram and reconstruction"):
            self.origen.append(frame)
            self.recons.append(fase.astype(np.uint8))
            
        elif(self.tipo=="Only the reconstruction"):
            self.recons.append(fase.astype(np.uint8))
        self.frame1 = fourier1.astype(np.uint8)
        return fourier1,fase
    
    def update(self):
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 1.0:  # Actualiza cada segundo
            fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()

    #Esta es la función principal de la GUI, se encarga de actualizar fotograma a fotograma
    def mostrar_fotogramas(self):
        
        # Capturar un fotograma
        frame = self.capturar_fotograma()

        if frame is not None:
            if self.ini != 0:
                # Aplicar las transformaciones
                if self.ini == 1:
                    frame_trans1,frame_trans2= self.transformacion_1(frame)
                else:
                    frame_trans1,frame_trans2= self.transformacion_2(frame)
                # Convertir los fotogramas a formato adecuado para Tkinter
                
                M,N = frame.shape
                new_width = round(self.ancho)  # Ensure this is a valid integer

                # Calculate new height maintaining the aspect ratio
                aspect_ratio = M / N
                new_height = round(new_width * aspect_ratio)

                img_original = Image.fromarray((cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)))
                
                img_original = img_original.resize((self.ancho,new_height))
                #transformada de fourier
                frame_trans1 = frame_trans1.astype(np.uint8)
                
                img_trans1 = Image.fromarray(cv2.cvtColor(frame_trans1, cv2.COLOR_GRAY2RGB))
                img_trans1 = img_trans1.resize((self.ancho,new_height))
                
                frame_trans2 = frame_trans2.astype(np.uint8)
                img_trans2 = Image.fromarray(cv2.cvtColor(frame_trans2, cv2.COLOR_GRAY2RGB))
                img_trans2 = img_trans2.resize((self.ancho,new_height))
                
                #Resultado
                img_original_tk = ImageTk.PhotoImage(image=img_original)
                img_trans1_tk = ImageTk.PhotoImage(image=img_trans1)
                img_trans2_tk = ImageTk.PhotoImage(image= img_trans2)

                # Mostrar las imágenes en los widgets Label
                self.label_original.configure(image=img_original_tk)
                self.label_original.image = img_original_tk

                self.label_transformacion1.configure(image=img_trans1_tk)
                self.label_transformacion1.image = img_trans1_tk

                self.label_transformacion2.configure(image=img_trans2_tk)
                self.label_transformacion2.image = img_trans2_tk
                self.ventana.update_idletasks()  # Forzar actualización
                self.update()

        # Llamar al método cada 20 milisegundos para mostrar el siguiente fotograma
        self.ventana.after(20, self.mostrar_fotogramas)
    
    #Esta función permite reconocer los parámetros para la reconstrucción
    def aplicar_transformaciones(self):
        self.boton_stop.config(state="normal")
        # Obtener los parámetros ingresados por el usuario
        self.dx = float(self.entry_param1.get())
        self.dy = float(self.entry_param2.get())
        self.lamb = float(self.entry_param3.get())
        self.cuadrante = float(self.entry_param4.get())
        #self.mask_len_data = int(self.mask_len.get())
        self.ini=1
        # Aquí puedes realizar alguna acción con los parámetros, como aplicar alguna transformación adicional
    def abrir_nueva_ventana(self):
        # Crear una nueva ventana
        self.nueva_ventana = tk.Toplevel(self.ventana)
        self.nueva_ventana.title("FFT")
       
        # Configurar la figura de Matplotlib
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("FFT in real time")
        self.ax.axis('off')
        self.ax.autoscale(True)
        self.img1 = self.ax.imshow(cv2.cvtColor(self.frame1, cv2.COLOR_GRAY2RGB))
        # Actualizar la imagen en las figuras utilizando FuncAnimation
        self.ani = FuncAnimation(self.fig, self.actualizar)
        # Agregar el lienzo de Matplotlib a la ventana de Tkinter
        canvas = FigureCanvasTkAgg(self.fig, master=self.nueva_ventana)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # Agregar la barra de herramientas de Matplotlib
        toolbar = NavigationToolbar2Tk(canvas, self.nueva_ventana, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        

        # Configurar el cierre de la ventana
        self.nueva_ventana.protocol("WM_DELETE_WINDOW", self.cerrar_ventana)
                
        self.nueva_ventana.mainloop()

    def actualizar(self,frame):
        self.img1.set_array(cv2.cvtColor(self.frame1, cv2.COLOR_GRAY2RGB))
        agua = self.img1
        return agua,

    def cerrar_ventana(self):
        if self.ani is not None:
            self.ani.event_source.stop()
        if self.fig is not None:
            plt.close(self.fig)
        if self.nueva_ventana is not None:
            self.nueva_ventana.destroy()
        self.nueva_ventana = None
        self.fig = None
        self.ax = None
        self.ani = None

    #Seccion para el tracking
    #Seccion para el tracking
    def run_other_program(self):
        python_path = sys.executable  
        # Ejecutar el segundo script
        try:
            # Ruta al segundo ejecutable
            tracking_serial_exe = "HoloStream_tracking_interface.exe"
            
            # Ejecutar el segundo ejecutable
            subprocess.run([tracking_serial_exe], check=True)
        except:
            try:
                # Ruta al segundo ejecutable
                tracking_serial_exe = "HoloStream_tracking_interface.py"
                
                # Ejecutar el segundo ejecutable
                subprocess.run([python_path, tracking_serial_exe], check=True)
            except:
                print(f"Error executing the tracking interface")
    def run_other_program_DSHPC(self):
        # Ejecutar el segundo script
        python_path = sys.executable  
        try:
            # Ruta al segundo ejecutable
            tracking_serial_exe = "HoloStream_compensation_interface.exe"
            
            # Ejecutar el segundo ejecutable
            subprocess.run([tracking_serial_exe], check=True)
        except:
            try:
                print("tracking")
                # Ruta al segundo ejecutable
                tracking_serial_exe = "HoloStream_compensation_interface.py"
                env = os.environ.copy()
                # Ejecutar el segundo ejecutable
                subprocess.run([python_path, tracking_serial_exe], check=True)
                
            except:
                print(f"Error executing the compensation interface")


    def select_file(self):
        # Abrir el diálogo de selección de archivo
        file_path = filedialog.askopenfilename()
        if file_path:
            print(f"Selected file: {file_path}")
            # Mostrar la ruta del archivo seleccionado en el Entry
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)

# Creación de la ventana y de la aplicación
root = tk.Tk()  # Utilizamos una ventana normal de Tkinter
app = CameraApp(root)
def on_closing():
    if messagebox.askokcancel("Exit", "Do you want to exit?"):
        root.destroy()
        sys.exit()  # Exit the program
root.protocol("WM_DELETE_WINDOW", on_closing)
root.iconbitmap("icon.ico")
root.mainloop()
