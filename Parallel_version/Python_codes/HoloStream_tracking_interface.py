import tkinter as tk
from tkinter import ttk, filedialog
import ttkbootstrap as ttkb
import cv2
import numpy as np
from numpy import asarray
from Funciones import *
import time
import imageio
from numpy import fft
from tkinter import messagebox
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import skcuda.fft as cu_fft
from pycuda.reduction import ReductionKernel
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
class RegionSelector:
    def __init__(self, master):
        self.master = master
        self.image_path = None
        self.min_area_coords = None
        self.selection_mode = "min"
        self.start_x = None
        self.start_y = None
        self.rect = None

        # Crear una figura y un eje en Matplotlib
        self.figure, self.ax = plt.subplots()
        self.cid_press = self.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.figure.canvas.mpl_connect('button_release_event', self.on_release)

    def open_image(self,image):
        self.image = image
        self.ax.imshow(self.image)
        self.figure.canvas.draw()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.start_x = int(event.xdata)
        self.start_y = int(event.ydata)

    def on_release(self, event):
        if self.start_x is None or self.start_y is None:
            return
        end_x, end_y = int(event.xdata), int(event.ydata)
        width = abs(end_x - self.start_x)
        height = abs(end_y - self.start_y)
        x0, y0 = min(self.start_x, end_x), min(self.start_y, end_y)
        x1, y1 = x0 + width, y0 + height
        
        if self.rect is not None:
            self.rect.remove()
        
        self.rect = plt.Rectangle((x0, y0), width, height, edgecolor='red', facecolor='none')
        self.ax.add_patch(self.rect)
        self.figure.canvas.draw()

        if self.selection_mode == "min":
            self.min_area_coords = (x0, y0, x1, y1)
            min_area_width = width
            min_area_height = height
            min_area = min_area_width * min_area_height
            plt.close(self.figure)


    def start_selection(self,image):
        self.open_image(image)
        plt.show()


class SecondWindowApp:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("Tracking")
        s = ttk.Style()
        s.configure('my.TButton', font=('Helvetica', 30))
        # Archivo de entrada
        tk.Label(ventana, text="Input file", font=("Helvetica", 16, "bold")).grid(row=0, column=0, columnspan=7, pady=(10, 5))
        
        self.file_entry = tk.Entry(ventana, width=50, state="readonly", font=("Helvetica", 14))
        self.file_entry.grid(row=1, column=0, columnspan=6, padx=10, pady=5, sticky="ew")

        file_button = tk.Button(ventana, text="Select file", command=self.select_file)
        file_button.grid(row=1, column=6, padx=10, pady=5)

        # Línea de separación
        ttk.Separator(ventana, orient=tk.HORIZONTAL).grid(row=2, column=0, columnspan=7, sticky='ew', pady=10)

        # RadioButton agregado
        self.opcion = tk.IntVar()
        tk.Label(ventana, text="Select type of tracking:", font=("Helvetica", 16, "bold")).grid(row=3, column=0, padx=10, pady=5, sticky="w")
        
        radio_button1 = tk.Radiobutton(ventana, text="2D tracking", variable=self.opcion, value=1, font=("Helvetica", 16), command=self.Change_principal_funtion)
        radio_button1.grid(row=3, column=3, padx=10, pady=5, sticky="w",columnspan=2)
        
        radio_button2 = tk.Radiobutton(ventana, text="3D Tracking", variable=self.opcion, value=2, font=("Helvetica", 16), command=self.Change_principal_funtion)
        radio_button2.grid(row=3, column=5, padx=10, pady=5, sticky="w",columnspan=2)

        # Línea de separación
        ttk.Separator(ventana, orient=tk.HORIZONTAL).grid(row=4, column=0, columnspan=7, sticky='ew', pady=10)

        # Parameters for tracking
        tk.Label(ventana, text="Parameters for tracking", font=("Helvetica", 16, "bold")).grid(row=5, column=0, columnspan=7)

        # Reconstruction range
        tk.Label(ventana, text="Reconstruction range", font=("Helvetica", 14)).grid(row=6, column=0, columnspan=5, padx=10, pady=5)

        self.min_range_entry = tk.Entry(ventana, width=10)
        self.min_range_entry.grid(row=7, column=0, padx=(10, 5), pady=5)
        self.min_range_entry.config(state="disabled")

        tk.Label(ventana, text="< z <", font=("Helvetica", 14)).grid(row=7, column=2, pady=5)

        self.max_range_entry = tk.Entry(ventana, width=10)
        self.max_range_entry.grid(row=7, column=3, padx=(5, 10), pady=5)
        self.max_range_entry.config(state="disabled")

        tk.Label(ventana, text="(um)", font=("Helvetica", 14)).grid(row=7, column=1, pady=5, padx=(2, 10))
        tk.Label(ventana, text="(um)", font=("Helvetica", 14)).grid(row=7, column=4, pady=5, padx=(2, 10))

        # Cantidad de pasos
        tk.Label(ventana, text="Number of steps", font=("Helvetica", 14)).grid(row=6, column=5, padx=10, pady=5, columnspan=2)

        self.steps_entry = tk.Entry(ventana, width=20)
        self.steps_entry.grid(row=7, column=5, padx=10, pady=5, columnspan=2)
        self.steps_entry.config(state="disabled")

        # Otros parámetros
        tk.Label(ventana, text="Delta x (um)", font=("Helvetica", 14)).grid(row=8, column=0, padx=10, pady=5)
        self.delta_x_entry = tk.Entry(ventana, width=15)
        self.delta_x_entry.grid(row=9, column=0, padx=10, pady=5)
        self.delta_x_entry.config(state="disabled")

        tk.Label(ventana, text="Delta y (um)", font=("Helvetica", 14)).grid(row=8, column=1, columnspan=2, padx=10, pady=5)
        self.delta_y_entry = tk.Entry(ventana, width=15)
        self.delta_y_entry.grid(row=9, column=1, columnspan=2, padx=10, pady=5)
        self.delta_y_entry.config(state="disabled")

        tk.Label(ventana, text="Wavelength (um)", font=("Helvetica", 14)).grid(row=8, column=3, padx=10, pady=5)
        self.wavelength_entry = tk.Entry(ventana, width=15)
        self.wavelength_entry.grid(row=9, column=3, padx=10, pady=5)
        self.wavelength_entry.config(state="disabled")

        tk.Label(ventana, text="Quadrant", font=("Helvetica", 14)).grid(row=8, column=4, padx=10, pady=5)
        self.entry_param4 = ttk.Combobox(ventana, values=[1, 2, 3, 4], state="readonly", width=1)
        self.entry_param4.grid(row=9, column=4, padx=(0, 0), pady=10)
        self.entry_param4.config(state="disabled")

        ttk.Label(ventana, text="Threshold (0 - 100)", font=("Helvetica", 14)).grid(row=8, column=5, padx=10, pady=5)
        self.thresh = tk.Entry(ventana, width=10)
        self.thresh.grid(row=9, column=5, padx=10, pady=5)
        self.thresh.config(state="disabled")
        
        ttk.Separator(ventana, orient=tk.HORIZONTAL).grid(row=16, column=0, columnspan=7, sticky='ew', pady=10)

        ttk.Label(ventana, text="Mask radius", font=("Helvetica", 14)).grid(row=8, column=6, padx=10, pady=5)
        self.mask_len = tk.Entry(ventana, width=10)
        self.mask_len.grid(row=9, column=6, padx=10, pady=5)
        self.mask_len.config(state="disabled")

        ttk.Separator(ventana, orient=tk.HORIZONTAL).grid(row=12, column=0, columnspan=7, sticky='ew', pady=10)

        # Area of the object
        ttk.Label(ventana, text="Area of the object", font=("Helvetica", 14)).grid(row=13, column=0, columnspan=7, padx=10, pady=5)
        self.min_area_entry = tk.Entry(ventana, width=30)
        self.min_area_entry.grid(row=14, column=0, columnspan=7, padx=10, pady=5)
        self.min_area_entry.config(state="disabled")

        self.btn_select_areas = ttk.Button(ventana, text="Select object", command=self.select_areas)
        self.btn_select_areas.grid(row=15, column=0, columnspan=7, padx=10, pady=5)
        self.btn_select_areas.config(state="disabled")

        # Nombre de archivo de salida
        ttk.Label(ventana, text="Output File name", font=("Helvetica", 16, "bold")).grid(row=17, column=0, columnspan=7, pady=(10, 5))
        self.file_name_out = tk.Entry(ventana, width=50, state="normal")
        self.file_name_out.grid(row=18, column=1, columnspan=5, padx=10, pady=5, sticky="ew")

        # Botón para iniciar tracking
        self.btn_abrir_ventana = ttk.Button(ventana, text="Start tracking")
        self.btn_abrir_ventana.grid(row=19, column=1, columnspan=5, sticky='ew', pady=10)
        self.btn_abrir_ventana.config(state="disabled")

        self.region_selector = RegionSelector(self.ventana)
        self.load_values_from_file("parameters.txt")
        self.llamado_funciones_cuda()

    def Change_principal_funtion(self):

        if self.opcion.get() == 2:
            self.btn_abrir_ventana.config(command=self.funcion_tracking)
            self.min_range_entry.config(state="normal")
            self.max_range_entry.config(state="normal")
            self.steps_entry.config(state="normal")
            self.delta_x_entry.config(state="normal")
            self.delta_y_entry.config(state="normal")
            self.wavelength_entry.config(state="normal")
            self.entry_param4.config(state="normal")
            self.thresh.config(state="normal")
            self.btn_abrir_ventana.config(state="normal")
            self.mask_len.config(state="normal")

        elif self.opcion.get() == 1:
            self.btn_abrir_ventana.config(command=self.funcion_tracking_2D)
            self.min_range_entry.config(state="disabled")
            self.max_range_entry.config(state="disabled")
            self.steps_entry.config(state="disabled")
            self.delta_x_entry.config(state="disabled")
            self.delta_y_entry.config(state="disabled")
            self.wavelength_entry.config(state="disabled")
            self.entry_param4.config(state="disabled")
            self.thresh.config(state="disabled")
            self.mask_len.config(state="disabled")
            self.btn_abrir_ventana.config(state="normal")

    def load_values_from_file(self, filepath):

        try:
            with open(filepath, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        key, value = line.strip().split('=')
                        if key == "delta_x":
                            self.delta_x_entry.config(state="normal")
                            self.delta_x_entry.insert(0, value)
                            self.delta_x_entry.config(state="disabled")
                        elif key == "delta_y":
                            self.delta_y_entry.config(state="normal")
                            self.delta_y_entry.insert(0, value)
                            self.delta_y_entry.config(state="disabled")
                        elif key == "wavelength":
                            self.wavelength_entry.config(state="normal")
                            self.wavelength_entry.insert(0, value)
                            self.wavelength_entry.config(state="disabled")
                        elif key == "quadrant":
                            self.entry_param4.config(state="normal")
                            self.entry_param4.set(value)
                            self.entry_param4.config(state="disabled")
                        elif key == "mask_radius":
                            self.mask_len.config(state="normal")
                            self.mask_len.insert(0, value)
                            self.mask_len.config(state="disabled")
                        elif key == "threshold":
                            self.thresh.config(state="normal")
                            self.thresh.insert(0, value)
                            self.thresh.config(state="disabled")
                        elif key == "min_range":
                            self.min_range_entry.config(state="normal")
                            self.min_range_entry.insert(0, value)
                            self.min_range_entry.config(state="disabled")
                        elif key == "max_range":
                            self.max_range_entry.config(state="normal")
                            self.max_range_entry.insert(0, value)
                            self.max_range_entry.config(state="disabled")
                        elif key == "steps":
                            self.steps_entry.config(state="normal")
                            self.steps_entry.insert(0, value)
                            self.steps_entry.config(state="disabled")
                        elif key == "output_file":
                            self.file_name_out.insert(0, value)   
        except Exception as e:
            print(f"Error al cargar el archivo: {e}")
    def llamado_funciones_cuda(self):
        with open("funciones.cu", "r") as kernel_file:
            kernel_code = kernel_file.read()
        mod = SourceModule(kernel_code)
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
        self.espectro_angular = mod.get_function("espectro_angular")
    def select_file(self):
        
        # Abrir el diálogo de selección de archivo
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_entry.config(state="normal")
            # Mostrar la ruta del archivo seleccionado en el Entry
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)
            self.file_entry.config(state="disabled")
            self.btn_select_areas.config(state="normal")
    def select_areas(self):
        imagen = video_to_frames(self.file_entry.get(), form = 2)
        self.region_selector.start_selection(imagen)
        # Ensure the areas are selected and stored
        self.min_area_coords = self.region_selector.min_area_coords
        

        # Calculate areas
        min_area_x0, min_area_y0, min_area_x1, min_area_y1 = self.min_area_coords
        min_area_width = min_area_x1 - min_area_x0
        min_area_height = min_area_y1 - min_area_y0

        min_area = min_area_width * min_area_height

        # Update the entries with selected areas
        self.min_area_entry.config(state="normal")  
        self.min_area_entry.delete(0, tk.END)
        self.min_area_entry.insert(0, str(min_area))
        self.min_area_entry.config(state="disabled")  

    def funcion_tracking_2D(self):
        video_path = self.file_entry.get()
        if video_path == "":
            messagebox.showinfo("Information", "No input video has been specified.")
            return
        min_area = self.min_area_entry.get()

        # Validate the area values
        if not min_area:
            messagebox.showerror("Error", "Please select the object.")
            return

        min_area = float(min_area)  # Convert string to float

        nombre = self.file_name_out.get()
        if nombre == "":
            messagebox.showinfo("Information", "A name for the output file has not been specified.")
            return
        video = cv2.VideoCapture(video_path)
        ret, frame = video.read()
        if not ret:
            exit()
        # Crear un VideoWriter para guardar el video de salida
        fps = video.get(cv2.CAP_PROP_FPS)  # Obtener la tasa de frames por segundo del video original
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = str(nombre) + ".avi"

        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Puedes ajustar el códec según tus necesidades
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Usar las coordenadas del área mínima seleccionada
        min_area_coords = self.region_selector.min_area_coords
        x0, y0, x1, y1 = min_area_coords

        # Calcular ancho y alto del área mínima
        w = x1 - x0
        h = y1 - y0

        # Inicializar el tracker para el área mínima
        bbox = (x0, y0, w, h)
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)  # Inicializar el tracker con el área mínima

        # Dibujar el cuadro inicial en el frame original
        cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), 2)
        out.write(frame)

        # Inicializar contador de frames
        frame_count = 1
        posiciones_cuerpos = [["frame","pos_x","pos_y"]]

        # Loop para procesar cada frame del video
        while True:
            ret, frame = video.read()
            if not ret:
                break

            frame_count += 1

            # Actualizar el tracker con el nuevo frame
            success, box = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]

                # Verificar si el cuadro está fuera de la pantalla
                if x < 0 or y < 0 or x + w > width or y + h > height:
                    break  # Salir si el objeto está fuera de la pantalla

                # Dibujar el rectángulo del tracker
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

                posiciones_cuerpos.append([frame_count, int(x+w/2), int(y+h/2)])
            else:
                break  # Salir si el tracker falla

            # Escribir el frame procesado en el video de salida
            out.write(frame)
        video.release()
        out.release()
        messagebox.showinfo("Information", "The task is done")
        np.savetxt(str(nombre) +"_object_position.txt", posiciones_cuerpos, fmt='%s', delimiter=',')
    
    
    def funcion_tracking(self):
        
        video_path = self.file_entry.get()
        if video_path == "":
            messagebox.showinfo("Información", "No se ha indicado el video de entrada")
            return
        frames = video_to_frames(video_path)
        try:
            U = asarray(frames[0])
        except:
            messagebox.showinfo("Información", "El archivo de entrada no es valido, por favor usar un archivo de video")
            return
        tiempo_inicial = time.time()
        N, M = U.shape
        #Parametros del montaje

        #Leemos la data indicada y comprobamos que exista

        
        try:
            minimo = float(self.min_range_entry.get())
        except:
            messagebox.showinfo("Información", "No se ha ingresado el valor minimo del intervalo de propagación")
            return
        try:
            maximo = float(self.max_range_entry.get())
        except:
            messagebox.showinfo("Información", "No se ha ingresado el valor maximo del intervalo de propagación")
            return
        if(minimo>=maximo):
            messagebox.showinfo("Información", "el valor mínimo es mayor al valor maximo o igual al del intervalo de propagación")
            return
        try:
            pasos = int(self.steps_entry.get())
        except:
            messagebox.showinfo("Información", "No se ha ingresado un valor valido para los pasos, recuerda que debe ser un valor entero")
            return
        try:
            dx = float(self.delta_x_entry.get())
        except:
            messagebox.showinfo("Información", "No se ha ingresado delta x")
            return
        try:
            dy = float(self.delta_y_entry.get())
        except:
            messagebox.showinfo("Información", "No se ha ingresado delta y")
            return
        try:
            lamb = float(self.wavelength_entry.get())
        except:
            messagebox.showinfo("Información", "No se ha ingresado la longitud de onda")
            return
        cuadrante=self.entry_param4.get()
        if cuadrante == "":
            messagebox.showinfo("Información", "No se ha indicado un cuadrante")
            return
        min_area = self.min_area_entry.get()
        
        # Validate the area values
        if not min_area :
            messagebox.showerror("Error", "Por favor seleccione el área")
            return

        min_area = float(min_area)  # Convert string to float

        nombre = self.file_name_out.get()
        if nombre == "":
            messagebox.showinfo("Información", "No se ha indicado un nombre para el archivo de salida")
            return
        thresh = self.thresh.get()
        if thresh == "":
            messagebox.showinfo("Information", "A threshold has not been specified.")
            return
        try:
            thresh = int(thresh)
        except:
            messagebox.showinfo("Information", "The threshold must be a number from 0 to 100.")
            return
        if (thresh < 0 or 100 < thresh):
            messagebox.showinfo("Information", "The threshold must be a number from 0 to 100.")
            return
        thresh = int(round(thresh/100 * 255))
        try:
            mask_len = int(self.mask_len.get())
        except:
            messagebox.showinfo("Information", "Mask size must be a integral number")
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
        #Parametros para tracking
        z = np.linspace(minimo,maximo,pasos)
        z_frames = []
        fx_esp = 1/(dx*N)
        fy_esp = 1/(dy*M)

        ksize = (15, 15)  # Tamaño del kernel del filtro gaussiano (esto es para homogenizar el ruido)
        #Variables definidas a la gpu
        self.U_gpu = gpuarray.to_gpu(U)
        self.m_gpu = gpuarray.to_gpu(m)
        self.n_gpu = gpuarray.to_gpu(n)

        if(int(cuadrante)==1):
            primer_cuadrante= np.zeros((N,M))
            primer_cuadrante[0:round(N/2 - (N*0.1)),round(M/2 + (M*0.1)):M] = 1
            primer_cuadrante = primer_cuadrante.astype(np.float32)
            self.cuadrante_gpu = gpuarray.to_gpu(primer_cuadrante)
        if(int(cuadrante)==2):
            segundo_cuadrante= np.zeros((N,M))
            segundo_cuadrante[0:round(N/2 -(N*0.1)),0:round(M/2 - (M*0.1))] = 1
            segundo_cuadrante = segundo_cuadrante.astype(np.float32)
            self.cuadrante_gpu = gpuarray.to_gpu(segundo_cuadrante)

        if(int(cuadrante)==3):
            tercer_cuadrante= np.zeros((N,M))
            tercer_cuadrante[round(N/2 +(N*0.1)):N,0:round(M/2 - (M*0.1))] = 1
            tercer_cuadrante = tercer_cuadrante.astype(np.float32)
            self.cuadrante_gpu = gpuarray.to_gpu(tercer_cuadrante)

        if(int(cuadrante)==4):
            cuarto_cuadrante= np.zeros((N,M))
            cuarto_cuadrante[round(N/2 +(N*0.1)):N,round(M/2 + (M*0.1)):M] = 1
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
        self.Amplitud(self.U_gpu,self.holo,np.int32(N),np.int32(M),block=block_dim, grid=grid_dim)

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
            
            grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
            #Amplitud de la imagen hasta el momentojunpei girlfriend combatchidori co
            self.fft_shift(self.holo2, self.holo, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)

            cu_fft.ifft(self.holo2, self.holo, self.plan)

            grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
            self.fft_shift(self.holo2, self.holo, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
            
            suma_maxima = float('inf')
            for j in z:
                # Entrada, temporal, salida, m_gpu, n_gpu, fx,fy ,z,np.pi,m,lamb
                grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)
                self.espectro_angular(self.holo2,self.temporal_gpu,self.holo,self.m_gpu,self.n_gpu, np.float32(fx_esp),np.float32(fy_esp),np.float32(j),np.float32(np.pi), np.int32(M), np.float32(lamb), block=block_dim, grid=grid_dim)
                #salida, entrada
                grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
                self.fft_shift(self.temporal_gpu, self.holo, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)

                cu_fft.ifft(self.temporal_gpu,self.holo,self.plan)
                grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
                self.fft_shift(self.temporal_gpu, self.holo, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
                grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)
                self.Amplitud(self.U_gpu,self.temporal_gpu,np.int32(N),np.int32(M),block=block_dim, grid=grid_dim)
                
                #Aqui podría cambiarse el criterio, esto toca hablarlo
                sum_gpu = gpuarray.sum(self.U_gpu)
                        
                temporal = sum_gpu.get()
                    
                if(suma_maxima>temporal):
                    z_fin = j
                    suma_maxima = temporal
            grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)
            self.espectro_angular(self.holo2,self.temporal_gpu,self.holo,self.m_gpu,self.n_gpu, np.float32(fx_esp),np.float32(fy_esp),np.float32(z_fin),np.float32(np.pi), np.int32(M), np.float32(lamb), block=block_dim, grid=grid_dim)
            grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
            self.fft_shift(self.temporal_gpu, self.holo, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
            cu_fft.ifft(self.temporal_gpu,self.holo,self.plan)

            grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
            self.fft_shift(self.temporal_gpu, self.holo, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
            grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)
            self.Fase(self.U_gpu,self.temporal_gpu,np.int32(N),np.int32(M),block=block_dim, grid=grid_dim)
            
            grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)
            max_value_gpu = gpuarray.max(self.U_gpu)
            min_value_gpu = gpuarray.min(self.U_gpu)
                        
            #Normalizar
                        
            self.Normalizar(self.U_gpu,np.int32(N), np.int32(M),min_value_gpu,max_value_gpu, block=block_dim, grid=grid_dim)
            
            #Obtención de la reconstrucción
            mai = self.U_gpu.get()
            finale = 255*np.abs(mai.reshape((N, M))-1)
            #Ahora sigue guardar los datos
            z_frames.append(z_fin)

            #Nuevo espacio donde proagar (alrededor del anterior)
            z = np.linspace(z_fin*10-1,z_fin*10+1,20)/10

            video.append(finale)
            
        # Convertir la lista de frames a un array numpy
        video = [matriz.astype(np.uint8) for matriz in video]
        
        # Define el formato del video y crea un objeto VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Puedes ajustar el códec según tus necesidades
        video_path = 'Imagenes/Temp/video_fase_serial.avi'  # Puedes cambiar la extensión según el formato deseado (mp4, gif, etc.)
        imageio.mimsave(video_path, video, fps=15)

        # Configura la ruta del video de entrada
        video = cv2.VideoCapture(video_path)
        # Crear un VideoWriter para guardar el video de salida
        fps = video.get(cv2.CAP_PROP_FPS)  # Obtener la tasa de frames por segundo del video original
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = str(nombre) + ".avi"

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Leer el primer frame para la detección inicial
        ret, frame = video.read()
        if not ret:
            exit()

        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplicar umbralización para detectar los cuadros
        _, thresholded = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

        # Usar las coordenadas del área mínima seleccionada
        min_area_coords = self.region_selector.min_area_coords
        x0, y0, x1, y1 = min_area_coords

        # Calcular ancho y alto del área mínima
        w = x1 - x0
        h = y1 - y0

        # Inicializar el tracker para el área mínima
        bbox = (x0, y0, w, h)
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)  # Inicializar el tracker con el área mínima

        # Dibujar el cuadro inicial en el frame original
        cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), 2)
        out.write(frame)

        # Inicializar contador de frames
        frame_count = 1
        posiciones_cuerpos = [["frame","pos_x","pos_y","pos_z"]]

        # Loop para procesar cada frame del video
        while True:
            ret, frame = video.read()
            if not ret:
                break

            frame_count += 1

            # Actualizar el tracker con el nuevo frame
            success, box = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]

                # Verificar si el cuadro está fuera de la pantalla
                if x < 0 or y < 0 or x + w > width or y + h > height:
                    break  # Salir si el objeto está fuera de la pantalla

                # Dibujar el rectángulo del tracker
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

                # Registrar posición del objeto
                z = z_frames[frame_count-1]
                posiciones_cuerpos.append([frame_count, int(x+w/2), int(y+h/2), z])
            else:
                break  # Salir si el tracker falla

            # Escribir el frame procesado en el video de salida
            out.write(frame)


        # Liberar recursos
        video.release()
        out.release()
        cv2.destroyAllWindows()
        np.savetxt(str(nombre) +"_object_position.txt", posiciones_cuerpos, fmt='%s', delimiter=',')
        messagebox.showinfo("Information", "The task is done")

        

# Crear la ventana secundaria de la aplicación
root = tk.Tk()


def on_closing():
    if messagebox.askokcancel("Exit", "Do you want to exit?"):
        root.destroy()
        sys.exit()  # Exit the program
app = SecondWindowApp(root)
root.protocol("WM_DELETE_WINDOW", on_closing)
root.iconbitmap("icon.ico")
root.mainloop()
