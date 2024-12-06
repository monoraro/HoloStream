import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk, filedialog
import ttkbootstrap as ttkb
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from Funciones import *
import time
import imageio
from numpy import fft
import sys

#Funcion para extraer los frames de un video
def video_to_frames(video_path, form = 1):
    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    
    
    if form == 1:
        frames = []
        # Leer el video frame por frame
        while True:
            ret, frame = cap.read()
            
            # Si no hay más frames, salir del bucle
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Convertir el frame a uint8 si no lo está ya
            frame = frame.astype(np.uint8)
            
            # Añadir el frame a la lista
            frames.append(frame)
        cap.release()
    else: 
        ret, frame = cap.read()
            
        # Si no hay más frames, salir del bucle
        if not ret:
            cap.release()
            return
        else:            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Convertir el frame a uint8 si no lo está ya
            frame = frame.astype(np.uint8)
                
            # Añadir el frame a la lista
            frames = frame
            cap.release()
    # Liberar el objeto VideoCapture
    return frames

class SecondWindowApp:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("Compensate hologram")
        
        # Input file
        ttk.Label(ventana, text="Input File", font=("Helvetica", 16, "bold")).grid(row=0, column=0, columnspan=6, pady=(10, 5))
        self.file_entry = tk.Entry(ventana, width=50, state="readonly")
        self.file_entry.grid(row=1, column=0, columnspan=5, padx=10, pady=5, sticky="ew")
        file_button = tk.Button(ventana, text="Select File", command=self.select_file)
        file_button.grid(row=1, column=5, padx=10, pady=5)
        
        ttk.Separator(ventana, orient=tk.HORIZONTAL).grid(row=2, column=0, columnspan=8, sticky='ew', pady=20)

        # Tracking parameters
        ttk.Label(ventana, text="Definition of parameters", font=("Helvetica", 16, "bold")).grid(row=3, column=0, columnspan=8, pady=20)

        ttk.Label(ventana, text="Delta x (um)", font=("Helvetica", 14)).grid(row=6, column=0, padx=10, pady=5)
        self.delta_x_entry = tk.Entry(ventana, width=15)
        self.delta_x_entry.grid(row=7, column=0, padx=10, pady=5)

        ttk.Label(ventana, text="Delta y (um)", font=("Helvetica", 14)).grid(row=6, column=1, columnspan=2, padx=10, pady=5)
        self.delta_y_entry = tk.Entry(ventana, width=15)
        self.delta_y_entry.grid(row=7, column=1, columnspan=2, padx=10, pady=5)

        ttk.Label(ventana, text="Wavelenght (um)", font=("Helvetica", 14)).grid(row=6, column=3, columnspan=2, padx=10, pady=5)
        self.wavelength_entry = tk.Entry(ventana, width=15)
        self.wavelength_entry.grid(row=7, column=3, columnspan=2, padx=10, pady=5)

        ttk.Label(ventana, text="Quadrant", font=("Helvetica", 14)).grid(row=6, column=5, padx=10, pady=5)
        self.entry_param4 = ttk.Combobox(ventana, values=[1, 2, 3, 4], state="readonly", width=1)
        self.entry_param4.grid(row=7, column=5, padx=(0, 0), pady=10)

        #ttk.Label(ventana, text="Mask radius", font=("Helvetica", 14)).grid(row=6, column=6, columnspan=2, padx=10, pady=5)
        #self.mask_len = tk.Entry(ventana, width=15)
        #self.mask_len.grid(row=7, column=6, columnspan=2, padx=10, pady=5)

        ttk.Separator(ventana, orient=tk.HORIZONTAL).grid(row=11, column=0, columnspan=8, sticky='ew', pady=20)

        ttk.Label(ventana, text="Output name file", font=("Helvetica", 16, "bold")).grid(row=12, column=0, columnspan=6, pady=(10, 5))
        self.file_name_out = tk.Entry(ventana, width=50, state="normal")
        self.file_name_out.grid(row=13, column=0, columnspan=6, padx=10, pady=5, sticky="ew")

        self.btn_abrir_ventana = ttk.Button(ventana, text="Start compensating", command=self.DSHPC)
        self.btn_abrir_ventana.grid(row=14, column=1, columnspan=4, sticky='ew', pady=10)
        
        self.progress_bar = ttk.Progressbar(self.ventana, orient="horizontal", length=200, mode="determinate")
        self.progress_bar.grid(row=7, column=1,columnspan=4, padx=10, pady=10)
        # Etiqueta para el estado de la barra de progreso
        self.progress_label = ttk.Label(ventana, text="Uploading...")
        self.progress_label.grid(row=8, column=1,columnspan=4, padx=10, pady=10)

        self.load_values_from_file("parametros.txt")  # archivo de texto con parámetros

    def select_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_entry.config(state="normal")
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)
            self.file_entry.config(state="disabled")
    def load_values_from_file(self, filepath):
        """Lee un archivo de texto y llena los campos con los valores"""
        try:
            with open(filepath, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        
                        key, value = line.split('=', 1)  # Dividir en clave y valor, solo en el primer '='
                        if key == "delta_x":
                            self.delta_x_entry.insert(0, value)
                        elif key == "delta_y":
                            self.delta_y_entry.insert(0, value)
                        elif key == "wavelength":
                            self.wavelength_entry.insert(0, value)
                        elif key == "quadrant":
                            self.entry_param4.set(value)
                        #elif key == "mask_radius":
                            #self.mask_len.insert(0, value)
                        elif key == "output_file":
                            self.file_name_out.insert(0, value)
        except Exception as e:
            print(f"Error al cargar el archivo: {e}")
    def DSHPC(self):
        
        video_path = self.file_entry.get()
        if video_path == "":
            messagebox.showinfo("Información", "No input video has been specified.")
            return
        frames = video_to_frames(video_path)
        try:
            U = asarray(frames[0])
        except:
            messagebox.showinfo("Información", "The input file is not valid, please use a video file.")
            return
        
        N, M = U.shape
        #Parametros del montaje

        #Leemos la data indicada y comprobamos que exista
        
        try:
            dx = float(self.delta_x_entry.get())
        except:
            messagebox.showinfo("Información", "Delta x has not been entered.")
            return
        try:
            dy = float(self.delta_y_entry.get())
        except:
            messagebox.showinfo("Información", "Delta y has not been entered.")
            return
        try:
            lamb = float(self.wavelength_entry.get())
        except:
            messagebox.showinfo("Información", "The wavelength has not been entered.")
            return
        cuadrante=self.entry_param4.get()
        if cuadrante == "":
            messagebox.showinfo("Información", "A quadrant has not been specified.")
            return

        nombre = self.file_name_out.get()
        if nombre == "":
            messagebox.showinfo("Información", "A name for the output file has not been specified.")
            return

        # Muestra la barra de progreso y la etiqueta
        self.progress.pack(pady=10)
        self.progress_label.pack()
        self.progress["value"] = 0
        self.ventana.update_idletasks()
        k= 2*np.pi/lamb
        Fox= M/2
        Foy= N/2
        # pixeles en el eje x y y de la imagen de origen
        x = np.arange(0, M, 1)
        y = np.arange(0, N, 1)

        #Un meshgrid para la paralelizacion
        m, n = np.meshgrid(x - (M/2), y - (N/2))

        #Definiendo cuadrantes, solo calidad
        primer_cuadrante= np.zeros((N,M))
        primer_cuadrante[0:round(N/2 - (N*0.15)),round(M/2 + (M*0.15)):M]=1
        segundo_cuadrante= np.zeros((N,M))
        segundo_cuadrante[0:round(N/2 -(N*0.15)),0:round(M/2 - (M*0.15))]=1
        tercer_cuadrante= np.zeros((N,M))
        tercer_cuadrante[round(N/2 +(N*0.15)):N,0:round(M/2 - (M*0.15))]=1
        cuarto_cuadrante= np.zeros((N,M))
        cuarto_cuadrante[round(N/2 +(N*0.15)):N,round(M/2 + (M*0.15)):M]=1
        
        #Ahora a tirar fourier
        tiempo_inicial = time.time()
        fourier=fft.fftshift(fft.fft2(fft.fftshift(U)))
        if(cuadrante=="1"):
            mascara=primer_cuadrante
            fourier=primer_cuadrante*fourier
        if(cuadrante=="2"):
            mascara=segundo_cuadrante
            fourier=segundo_cuadrante*fourier
        if(cuadrante=="3"):
            mascara=tercer_cuadrante
            fourier=tercer_cuadrante*fourier
        if(cuadrante=="4"):
            mascara=cuarto_cuadrante
            fourier=cuarto_cuadrante*fourier
        a=amplitud(fourier)
        #Calculamos la amplitud del espectro de fourier

        #Encontramos la posición en x y y del máximo en el espacio de Fourier
        pos_max = np.unravel_index(np.argmax(a, axis=None), a.shape)
        #mayor = max(M,N)
        #tamano = int(round(mayor*0.14))
        #mascara = crear_mascara_circular(U.shape,(pos_max[1],pos_max[0]),tamano)
        #Transformada insversa de fourier
        #fourier= fourier*mascara
        fourier=fft.fftshift(fft.ifft2(fft.fftshift(fourier)))

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
                    temp, faserina=tiro(fourier,Fox,Foy,fx_temp,fy_temp,lamb,M,N,dx,dy,k,m,n)
                    if(temp>suma_maxima):
                        x_max_out = fx_temp
                        y_max_out = fy_temp
                        suma_maxima = temp
            G_temp = G_temp - 1
            
            if(x_max_out == fx):
                if(y_max_out ==fy):
                    fin=1
            if(G_temp==0):
                    fin=1
            fx=x_max_out
            fy=y_max_out
        
        theta_x=mt.asin((Fox - fx) * lamb /(M*dx))
        theta_y=mt.asin((Foy - fy) * lamb /(N*dy))
        fase= np.exp(1j*k* ((mt.sin(theta_x) * m * dx)+ ((mt.sin(theta_y) * n * dy))))
        holo=fourier*fase
        fase = np.angle(holo, deg=False)
        min_val = np.min(fase)
        max_val = np.max(fase)
        fase = 255*(fase - min_val) / (max_val - min_val)
        self.progress["value"] += (1/len(frames) *100)
        self.ventana.update_idletasks()  # Actualiza la GUI
        if  len(frames)==1:
            video_path = str(nombre) + ".bmp"
            # Oculta la barra de progreso y el mensaje de "Cargando..."
            self.progress.pack_forget()
            self.progress_label.pack_forget()
            cv2.imwrite(video_path, fase)
            messagebox.showinfo("Information", "The compensation task is done")
            return
        video = []
        tiempo_final = time.time()
        tiempo= tiempo_final-tiempo_inicial
        tiempo_frame=[tiempo]
        frames.pop(0)
        
        #Ahora la versión dinámica
        paso = 0.2
        
        for frame in frames:
            
            U = asarray(frame)
            
            fin=0
            G_temp=1
            
            tiempo_inicial= time.time()
            fourier=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(U)))  
            fourier= fourier*mascara
            fourier=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fourier)))
            
            while fin==0:
                temp=0
                frec_esp_x=np.arange(fx-paso*G_temp,fx+paso*G_temp,paso)
                frec_esp_y=np.arange(fy-paso*G_temp,fy+paso*G_temp,paso)
                for i in range(len(frec_esp_y)):
                    for j in range(len(frec_esp_x)):
                        fx_temp=frec_esp_x[j]
                        fy_temp=frec_esp_y[i]
                        temp, faserina=tiro(fourier,Fox,Foy,fx_temp,fy_temp,lamb,M,N,dx,dy,k,m,n)
                        if(temp>suma_maxima):
                            x_max_out = fx_temp
                            y_max_out = fy_temp
                            suma_maxima = temp
                G_temp = G_temp - 1
                
                if(x_max_out == fx):
                    if(y_max_out ==fy):
                        fin=1
                if(G_temp==0):
                    fin=1
                fx=x_max_out
                fy=y_max_out
            theta_x=mt.asin((Fox - fx) * lamb /(M*dx))
            theta_y=mt.asin((Foy - fy) * lamb /(N*dy))
            fase= np.exp(1j*k* ((mt.sin(theta_x) * m * dx)+ ((mt.sin(theta_y) * n * dy))))
            holo=fourier*fase 
            fase = np.angle(holo, deg=False)
            min_val = np.min(fase)
            max_val = np.max(fase)
            fase = 255*(fase - min_val) / (max_val - min_val)
            video.append(fase)
            tiempo_final = time.time()
            tiempo= tiempo_final-tiempo_inicial
            tiempo_frame.append(tiempo)
            self.progress["value"] += (1/len(frames) *100)
            self.ventana.update_idletasks()  # Actualiza la GUI
        # Guardar el array en un archivo de texto

        #Guardar los tiempos de computo
        #np.savetxt('tiempos_compensacion.txt', tiempo_frame, fmt='%f', delimiter='\t')

        video = [matriz.astype(np.uint8) for matriz in video]

        # Define el formato del video y crea un objeto VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Puedes ajustar el códec según tus necesidades
        # Oculta la barra de progreso y el mensaje de "Cargando..."
        self.progress.pack_forget()
        self.progress_label.pack_forget()
        video_path = str(nombre) + ".avi"
        imageio.mimsave(video_path, video, fps=15)
        
        messagebox.showinfo("Information", "The compensation task is done")


# Crear la ventana principal de la aplicación
root = tk.Tk()
def on_closing():
    if messagebox.askokcancel("Exit", "Do you want to exit the compensation interface?"):
        root.destroy()
        sys.exit()  # Exit the program
app = SecondWindowApp(root)
root.protocol("WM_DELETE_WINDOW", on_closing)
root.iconbitmap("icon.ico")
root.mainloop()
