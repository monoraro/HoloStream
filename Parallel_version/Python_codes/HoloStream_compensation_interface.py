
#Este codigo es para agregar el path, que es que sino no lo reconoce
import os
import sys

# Obtener el PATH global del sistema
global_path = os.environ['PATH']

# Dividir las rutas en una lista
paths = global_path.split(os.pathsep)

# Agregar cada ruta a sys.path
for path in paths:
    if path not in sys.path:  # Para evitar duplicados
        sys.path.append(path)

#Leemos otras librerias necesarias
from tkinter import filedialog, messagebox
import cv2
import tkinter as tk
from tkinter import ttk
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
import numpy as np
from numpy import asarray
import math as mt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import skcuda.fft as cu_fft
from pycuda.reduction import ReductionKernel
from Funciones import *
import time
import imageio

#Cargado de funciones CUDA
with open("funciones.cu", "r") as kernel_file:
    kernel_code = kernel_file.read()
mod = SourceModule(kernel_code)

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
        ttk.Label(ventana, text="Input File", font=("Helvetica", 16, "bold")).grid(row=0, column=0, columnspan=8, pady=(10, 5))
        self.file_entry = tk.Entry(ventana, width=50, state="readonly")
        self.file_entry.grid(row=1, column=0, columnspan=7, padx=10, pady=5, sticky="ew")
        file_button = tk.Button(ventana, text="Select File", command=self.select_file)
        file_button.grid(row=1, column=7, padx=10, pady=5)
        
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

        ttk.Label(ventana, text="Mask radius", font=("Helvetica", 14)).grid(row=6, column=6, columnspan=2, padx=10, pady=5)
        self.mask_len = tk.Entry(ventana, width=15)
        self.mask_len.grid(row=7, column=6, columnspan=2, padx=10, pady=5)

        ttk.Separator(ventana, orient=tk.HORIZONTAL).grid(row=11, column=0, columnspan=8, sticky='ew', pady=20)

        ttk.Label(ventana, text="Output name file", font=("Helvetica", 16, "bold")).grid(row=12, column=0, columnspan=8, pady=(10, 5))
        self.file_name_out = tk.Entry(ventana, width=50, state="normal")
        self.file_name_out.grid(row=13, column=1, columnspan=6, padx=10, pady=5, sticky="ew")

        self.btn_abrir_ventana = ttk.Button(ventana, text="Start compensating", command=self.DSHPC)
        self.btn_abrir_ventana.grid(row=14, column=1, columnspan=6, sticky='ew', pady=10)
        self.llamado_funciones_cuda()
        # Llamar la función que carga los valores
        self.load_values_from_file("parametros.txt")  # archivo de texto con parámetros

    def load_values_from_file(self, filepath):

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
                        elif key == "mask_radius":
                            self.mask_len.insert(0, value)
                        elif key == "output_file":
                            self.file_name_out.insert(0, value)
        except Exception as e:
            print(f"Error al cargar el archivo: {e}")
    def llamado_funciones_cuda(self):
        
        self.fase_refe = mod.get_function("fase_refe")
        self.fft_shift = mod.get_function("fft_shift")
        self.fft_shift_var_no_compleja = mod.get_function("fft_shift_var_no_compleja")
        self.coordenadas_maximo= mod.get_function("coordenadas_maximo")
        self.thresholding_kernel = mod.get_function("thresholding")
        self.mascara_1er_cuadrante = mod.get_function("mascara_1er_cuadrante")
        self.Normalizar = mod.get_function("Normalizar")
        self.Reseteo = mod.get_function("reseteo")
        self.Amplitud = mod.get_function("Amplitud")
        self.Intensidad = mod.get_function("Intensidad")
        self.Fase = mod.get_function("Fase")
        self.logaritmo = mod.get_function("Amplitud_log")
    def select_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_entry.config(state="normal")
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)
            self.file_entry.config(state="disabled")

    def DSHPC(self):
        
        video_path = self.file_entry.get()
        if video_path == "":
            messagebox.showinfo("Information", "Input video has not been specified")
            return
        try:
            frames = video_to_frames(video_path)
            U = asarray(frames[0])
            U = ajuste_tamano1(U)
        except:
            messagebox.showinfo("Information", "The input file is not valid, please use a video or image file file")
            return
        N, M = U.shape
        #Parametros del montaje

        #Leemos la data indicada y comprobamos que exista
        
        try:
            dx = float(self.delta_x_entry.get())
        except:
            messagebox.showinfo("Information", "Delta x has not been entered")
            return
        try:
            dy = float(self.delta_y_entry.get())
        except:
            messagebox.showinfo("Information", "Delta y has not been entered")
            return
        try:
            lamb = float(self.wavelength_entry.get())
        except:
            messagebox.showinfo("Information", "Wavelength has not been entered")
            return
        cuadrante = self.entry_param4.get()
        if cuadrante == "":
            messagebox.showinfo("Information", "A quadrant has not been specified")
            return

        try:
            mask_len = int(self.mask_len.get())
        except:
            messagebox.showinfo("Information", "mask size must be a integral number")
            return
        nombre = self.file_name_out.get()
        if nombre == "":
            messagebox.showinfo("Information", "A name for the output file has not been specified")
            return

        k= 2*np.pi/lamb
        Fox= M/2
        Foy= N/2
        # pixeles en el eje x y y de la imagen de origen
        x = np.arange(0, M, 1)
        y = np.arange(0, N, 1)

        #Un meshgrid para la paralelizacion
        m, n = np.meshgrid(x - (M/2), y - (N/2))

        self.N = N
        self.M = M
        x = np.arange(0, M, 1)
        y = np.arange(0, N, 1)

        #Un meshgrid para la paralelizacion
        m, n = np.meshgrid(x - (M/2), y - (N/2))
        G=3
        k = 2*mt.pi/lamb
        Fox = M/2
        Foy = N/2
        threso = 0.2
        #Esta variable sirve para inicializar el valor mínimo de la suma
        suma_max = np.array([[0]]) 
        #Definicion de tipos de variables compatibles con cuda
        U = U.astype(np.float32)
        
        m = m.astype(np.float32)
        n = n.astype(np.float32)
        #Variables definidas a la gpu
        self.U_gpu = gpuarray.to_gpu(U)
        self.m_gpu = gpuarray.to_gpu(m)
        self.n_gpu = gpuarray.to_gpu(n)

        if(int(cuadrante)==1):
            primer_cuadrante= np.zeros((N,M))
            primer_cuadrante[0:round(N/2 - (N*0.15)),round(M/2 + (M*0.15)):M] = 1
            primer_cuadrante = primer_cuadrante.astype(np.float32)
            self.cuadrante_gpu = gpuarray.to_gpu(primer_cuadrante)
        if(int(cuadrante)==2):
            segundo_cuadrante= np.zeros((N,M))
            segundo_cuadrante[0:round(N/2 -(N*0.15)),0:round(M/2 - (M*0.15))] = 1
            segundo_cuadrante = segundo_cuadrante.astype(np.float32)
            self.cuadrante_gpu = gpuarray.to_gpu(segundo_cuadrante)

        if(int(cuadrante)==3):
            tercer_cuadrante= np.zeros((N,M))
            tercer_cuadrante[round(N/2 +(N*0.15)):N,0:round(M/2 - (M*0.15))] = 1
            tercer_cuadrante = tercer_cuadrante.astype(np.float32)
            self.cuadrante_gpu = gpuarray.to_gpu(tercer_cuadrante)

        if(int(cuadrante)==4):
            cuarto_cuadrante= np.zeros((N,M))
            cuarto_cuadrante[round(N/2 +(N*0.15)):N,round(M/2 + (M*0.15)):M] = 1
            cuarto_cuadrante = cuarto_cuadrante.astype(np.float32)
            self.cuadrante_gpu = gpuarray.to_gpu(cuarto_cuadrante)
        
        #Creamos espacios de memoria en la GPU para trabajo
        self.holo = gpuarray.empty((N, M), np.complex128)
        self.holo2 = gpuarray.empty((N, M), np.complex128)
        self.temporal_gpu = gpuarray.empty((N, M), np.complex128)
        
        

        # definición de espacios para trabajar
        block_dim = (32, 32, 1)
        tiempo_inicial = time.time()
        # Mallado para la fft shift
        grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
        #fft_shift
        self.fft_shift_var_no_compleja(self.holo2,self.U_gpu,np.int32(N),np.int32(M),block=block_dim, grid=grid_dim)

        #Fourier
        self.plan = cu_fft.Plan((N,M), np.complex64, np.complex64)

        cu_fft.fft(self.holo2, self.holo, self.plan)

        #fft_shift
        grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
        self.fft_shift(self.holo2, self.holo, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
        #1280 x 960 las imagenes 
        grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)
        self.Amplitud(self.U_gpu,self.holo2,np.int32(N),np.int32(M),block=block_dim, grid=grid_dim)
        mai = self.U_gpu.get()
        frame = 255*np.log(mai.reshape((N, M))+1)
        grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)
        self.mascara_1er_cuadrante(self.holo,self.holo2,self.cuadrante_gpu, np.int32(N),np.int32(M), block=block_dim, grid=grid_dim)

        #1280 x 960 las imagenes 
        grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)
        self.Amplitud(self.U_gpu,self.holo,np.int32(N),np.int32(M),block=block_dim, grid=grid_dim)
        mai = self.U_gpu.get()
        finale = mai.reshape((N, M))
        

        #Creacion de la mascara circular
        pos_max = np.unravel_index(np.argmax(finale, axis=None), U.shape)
        mascara = crear_mascara_circular(U.shape,(pos_max[1],pos_max[0]),mask_len)
        mascara = asarray(mascara.astype(np.float32))
        self.cuadrante_gpu = gpuarray.to_gpu(mascara)
        
        grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)
        self.mascara_1er_cuadrante(self.holo,self.holo2,self.cuadrante_gpu, np.int32(N),np.int32(M), block=block_dim, grid=grid_dim)

        self.Amplitud(self.U_gpu,self.holo,np.int32(N),np.int32(M),block=block_dim, grid=grid_dim)
        mai = self.U_gpu.get()
        grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
        #Amplitud de la imagen hasta el momento
        self.fft_shift(self.holo2, self.holo, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)

        cu_fft.ifft(self.holo2, self.holo, self.plan)

        grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
        self.fft_shift(self.holo2, self.holo, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
        
        grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)
        block_dim = (32, 32, 1)
        # Configuración de la cuadrícula y los bloques
        block_size = 256
        grid_size = 1  # Solo un bloque para buscar el máximo global

        # Crear buffer para la posición del máximo en GPU
        self.max_position_gpu= gpuarray.zeros((U.shape[0],), dtype=np.int32)

        # Ejecutar el kernel de búsqueda binaria
        self.coordenadas_maximo(self.U_gpu, np.int32(U.shape[0]), np.int32(U.shape[1]), self.max_position_gpu, block=(block_size, 1, 1), grid=(grid_size, 1))

        max_position_cpu = self.max_position_gpu.get()[0]
        # Calcular las coordenadas (fila, columna) desde la posición
        col_index, row_index = divmod(max_position_cpu, U.shape[1])

        paso=0.2
        fin=0
        fy=col_index
        fx=row_index
        G_temp = G
        suma_maxima=0
        grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)

        
        while fin==0:
            i=0
            j=0
            frec_esp_x=np.arange(fx-paso*G_temp,fx+paso*G_temp,paso)
            frec_esp_y=np.arange(fy-paso*G_temp,fy+paso*G_temp,paso)
            
            for j in range(len(frec_esp_y)): 
                for i in range(len(frec_esp_x)):
                    fx_temp=frec_esp_x[i]
                    fy_temp=frec_esp_y[j]

                    #La propago
                    self.fase_refe(self.holo2, self.holo, self.temporal_gpu, self.m_gpu, self.n_gpu, np.int32(N), np.int32(M), np.float32(k), np.float32(Fox), np.float32(Foy), np.float32(fx_temp), np.float32(fy_temp), np.float32(lamb), np.float32(dx), np.float32(dy), block=block_dim, grid=grid_dim)

                    #La reconstruyo en fase

                    self.Fase(self.U_gpu,self.holo,np.int32(N),np.int32(M),block=block_dim, grid=grid_dim)
                    
                    #Ahora si toca encontrar maximos y minimos para normalizar
                    
                    max_value_gpu = gpuarray.max(self.U_gpu)
                    min_value_gpu = gpuarray.min(self.U_gpu)
                    
                    #Normalizar
                    
                    self.Normalizar(self.U_gpu,np.int32(N), np.int32(M),min_value_gpu,max_value_gpu, block=block_dim, grid=grid_dim)
                    
                    #Aplicamos el thresholding
                    
                    self.thresholding_kernel(self.U_gpu,np.int32(N), np.int32(M), np.float32(threso), block=block_dim, grid=grid_dim)
                    
                    #Suma de la matriz

                    self.sum_gpu = gpuarray.sum(self.U_gpu)
                    
                    temporal = self.sum_gpu.get()

                    if(temporal>suma_maxima):
                        x_max_out = fx_temp
                        y_max_out = fy_temp
                        suma_maxima = temporal
            G_temp = G_temp - 1
            
            if(x_max_out == fx):
                if(y_max_out ==fy):
                    fin=1
            fx=x_max_out
            fy=y_max_out


        self.fx = fx
        self.fy = fy

        self.fase_refe(self.holo2, self.holo, self.temporal_gpu, self.m_gpu, self.n_gpu, np.int32(N), np.int32(M), np.float32(k), np.float32(Fox), np.float32(Foy), np.float32(fx), np.float32(fy), np.float32(lamb), np.float32(dx), np.float32(dy), block=block_dim, grid=grid_dim)
        self.Fase(self.U_gpu,self.holo,np.int32(N),np.int32(M),block=block_dim, grid=grid_dim)

        max_value_gpu = gpuarray.max(self.U_gpu)
        min_value_gpu = gpuarray.min(self.U_gpu)
                        
        #Normalizar
                        
        self.Normalizar(self.U_gpu,np.int32(N), np.int32(M),min_value_gpu,max_value_gpu, block=block_dim, grid=grid_dim)
            
        #Obtención de la reconstrucción
        mai = self.U_gpu.get()
        tiempo_final= time.time()
        tiempo_frame= []
        tiempo_frame.append(tiempo_final - tiempo_inicial)
        frame = 255*mai.reshape((N, M))
        video = []
        if  len(frames)==1:
            video_path = str(nombre) + ".bmp"
            
            cv2.imwrite(video_path, frame)
            messagebox.showinfo("Information", "The compensation task is done")
            return
        video.append(frame)
        
        

        #Ahora comienza del frame 2 en adelante
        frames.pop(0)
        
        for frame in frames:
            G_temp=1
            
            frame = ajuste_tamano1(frame)
            U = asarray(frame)
            U = U.astype(np.float32)
            self.U_gpu = gpuarray.to_gpu(U)
            
            block_dim = (32, 32, 1)
            grid_dim = (self.N // (block_dim[0]*2), self.M // (block_dim[1]*2), 1)
            tiempo_inicial = time.time()
            #fft_shift
            self.fft_shift_var_no_compleja(self.holo2,self.U_gpu,np.int32(self.N),np.int32(self.M),block=block_dim, grid=grid_dim)

            #Fourier
            cu_fft.fft(self.holo2, self.holo, self.plan)
            
            #fft_shift
            grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
            self.fft_shift(self.holo2, self.holo, np.int32(self.N), np.int32(self.M), block=block_dim, grid=grid_dim)
            
            #Obtención del espacio valor
            grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)
            self.mascara_1er_cuadrante(self.holo,self.holo2,self.cuadrante_gpu, np.int32(N),np.int32(M), block=block_dim, grid=grid_dim)

            grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
            #Amplitud de la imagen hasta el momentojunpei girlfriend combatchidori co
            self.fft_shift(self.holo2, self.holo, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)

            cu_fft.ifft(self.holo2, self.holo, self.plan)

            grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
            self.fft_shift(self.holo2, self.holo, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)

            suma_maxima=0
            grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)

            fx = self.fx
            fy = self.fy
            fin=0
            
            while fin==0:

                frec_esp_x=np.arange(fx-paso*G_temp,fx+paso*G_temp,paso)
                frec_esp_y=np.arange(fy-paso*G_temp,fy+paso*G_temp,paso)
                for j in range(len(frec_esp_y)):
                    for i in range(len(frec_esp_x)):
                        fx_temp=frec_esp_x[i]
                        fy_temp=frec_esp_y[j]
                        #La propago
                        #Revisar función por función, para ver los tiempos
                        self.fase_refe(self.holo2, self.holo, self.temporal_gpu, self.m_gpu, self.n_gpu, np.int32(N), np.int32(M), np.float32(k), np.float32(Fox), np.float32(Foy), np.float32(fx_temp), np.float32(fy_temp), np.float32(lamb), np.float32(dx), np.float32(dy), block=block_dim, grid=grid_dim)

                        #La reconstruyo en fase
                        self.Fase(self.U_gpu,self.holo,np.int32(N),np.int32(M),block=block_dim, grid=grid_dim)
                        
                        #Ahora si toca encontrar maximos y minimos para normalizar
                        
                        max_value_gpu = gpuarray.max(self.U_gpu)
                        min_value_gpu = gpuarray.min(self.U_gpu)
                        
                        #Normalizar
                        #Revisar función por función, para ver los tiempos
                        self.Normalizar(self.U_gpu,np.int32(N), np.int32(M),min_value_gpu,max_value_gpu, block=block_dim, grid=grid_dim)
                        
                        #Aplicamos el thresholding
                        self.thresholding_kernel(self.U_gpu,np.int32(N), np.int32(M), np.float32(0.2), block=block_dim, grid=grid_dim)
                        
                        #Suma de la matriz
                        sum_gpu = gpuarray.sum(self.U_gpu)
                        
                        temporal = sum_gpu.get()

                        if(temporal>suma_maxima):
                            x_max_out = fx_temp
                            y_max_out = fy_temp
                            suma_maxima = temporal
                G_temp = G_temp - 1
                
                if(x_max_out == fx):
                    if(y_max_out ==fy):
                        fin=1
                fx=x_max_out
                fy=y_max_out
            
            self.fx = fx
            self.fy = fy
            self.fase_refe(self.holo2, self.holo, self.temporal_gpu, self.m_gpu, self.n_gpu, np.int32(N), np.int32(M), np.float32(k), np.float32(Fox), np.float32(Foy), np.float32(fx), np.float32(fy), np.float32(lamb), np.float32(dx), np.float32(dy), block=block_dim, grid=grid_dim)
            self.Fase(self.U_gpu,self.holo, np.int32(N),np.int32(M),block=block_dim, grid=grid_dim)
            max_value_gpu = gpuarray.max(self.U_gpu)
            min_value_gpu = gpuarray.min(self.U_gpu)
                        
            #Normalizar
                        
            self.Normalizar(self.U_gpu,np.int32(N), np.int32(M),min_value_gpu,max_value_gpu, block=block_dim, grid=grid_dim)
            
            #Obtención de la reconstrucción
            mai = self.U_gpu.get()
            finale = 255*mai.reshape((N, M))
            video.append(finale)
            tiempo_final= time.time()
            tiempo_frame.append(tiempo_final - tiempo_inicial)
        tiempo_final = time.time()
        # Guardar el array en un archivo de texto

        
        #np.savetxt(str(nombre)+'.txt', tiempo_frame, fmt='%f', delimiter='\t')
        video = [matriz.astype(np.uint8) for matriz in video]

        # Define el formato del video y crea un objeto VideoWriter
        video_path = str(nombre) + ".avi"
        imageio.mimsave(video_path, video, fps=15)
        
        messagebox.showinfo("Information", "The compensation task is done")

# Crear la ventana principal de la aplicación
root = tk.Tk()
app = SecondWindowApp(root)
root.mainloop()
