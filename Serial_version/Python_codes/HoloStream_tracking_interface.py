import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk, filedialog
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from Funciones import *
import time
import imageio
from numpy import fft
import ttkbootstrap as ttkb
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
        print("maguiver")
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
        radio_button1.grid(row=3, column=2, padx=10, pady=5, sticky="w",columnspan=2)
        
        radio_button2 = tk.Radiobutton(ventana, text="3D Tracking", variable=self.opcion, value=2, font=("Helvetica", 16), command=self.Change_principal_funtion)
        radio_button2.grid(row=3, column=4, padx=10, pady=5, sticky="w",columnspan=2)

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

        tk.Label(ventana, text="Wavelength (um)", font=("Helvetica", 14)).grid(row=8, column=3,columnspan=2, padx=10, pady=5)
        self.wavelength_entry = tk.Entry(ventana, width=15)
        self.wavelength_entry.grid(row=9, column=3,columnspan=2, padx=10, pady=5)
        self.wavelength_entry.config(state="disabled")

        tk.Label(ventana, text="Quadrant", font=("Helvetica", 14)).grid(row=8, column=5,columnspan=2, padx=10, pady=5)
        self.entry_param4 = ttk.Combobox(ventana, values=[1, 2, 3, 4], state="readonly", width=1)
        self.entry_param4.grid(row=9, column=5,columnspan=2, padx=(0, 0), pady=10)
        self.entry_param4.config(state="disabled")

        #ttk.Label(ventana, text="Threshold (0 - 100)", font=("Helvetica", 14)).grid(row=8, column=5, padx=10, pady=5)
        #self.thresh = tk.Entry(ventana, width=10)
        #self.thresh.grid(row=9, column=5, padx=10, pady=5)
        #self.thresh.config(state="disabled")
        
        ttk.Separator(ventana, orient=tk.HORIZONTAL).grid(row=16, column=0, columnspan=7, sticky='ew', pady=10)

        #ttk.Label(ventana, text="Mask radius", font=("Helvetica", 14)).grid(row=8, column=6, padx=10, pady=5)
        #self.mask_len = tk.Entry(ventana, width=10)
        #self.mask_len.grid(row=9, column=6, padx=10, pady=5)
        #self.mask_len.config(state="disabled")

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
        self.file_name_out.grid(row=18, column=0, columnspan=7, padx=10, pady=5, sticky="ew")

        # Botón para iniciar tracking
        self.btn_abrir_ventana = ttk.Button(ventana, text="Start tracking")
        self.btn_abrir_ventana.grid(row=19, column=1, columnspan=3, sticky='ew', pady=10)
        self.btn_abrir_ventana.config(state="disabled")

        self.region_selector = RegionSelector(self.ventana)
        self.load_values_from_file("parameters.txt")

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
            #self.thresh.config(state="normal")
            self.btn_abrir_ventana.config(state="normal")
            #self.mask_len.config(state="normal")

        elif self.opcion.get() == 1:
            self.btn_abrir_ventana.config(command=self.funcion_tracking_2D)
            self.min_range_entry.config(state="disabled")
            self.max_range_entry.config(state="disabled")
            self.steps_entry.config(state="disabled")
            self.delta_x_entry.config(state="disabled")
            self.delta_y_entry.config(state="disabled")
            self.wavelength_entry.config(state="disabled")
            self.entry_param4.config(state="disabled")
            #self.thresh.config(state="disabled")
            #self.mask_len.config(state="disabled")
            self.btn_abrir_ventana.config(state="normal")


    def load_values_from_file(self, filepath):
        """Lee un archivo de texto y llena los campos con los valores"""
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
                        #elif key == "mask_radius":
                        #    self.mask_len.config(state="normal")
                        #    self.mask_len.insert(0, value)
                        #    self.mask_len.config(state="disabled")
                        #elif key == "threshold":
                        #    self.thresh.config(state="normal")
                        #    self.thresh.insert(0, value)
                        #    self.thresh.config(state="disabled")
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
    def select_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_entry.config(state="normal")
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
            messagebox.showinfo("Information", "No input video has been specified.")
            return
        frames = video_to_frames(video_path)
        try:
            U = asarray(frames[0])
        except:
            messagebox.showinfo("Information", "The input file is not valid, please use a video file.")
            return
        tiempo_inicial = time.time()
        N, M = U.shape
        # Parameters of the setup

        # Read the indicated data and check that it exists

        try:
            minimo = float(self.min_range_entry.get())
        except:
            messagebox.showinfo("Information", "The minimum value of the propagation range has not been entered.")
            return
        try:
            maximo = float(self.max_range_entry.get())
        except:
            messagebox.showinfo("Information", "The maximum value of the propagation range has not been entered.")
            return
        if (minimo >= maximo):
            messagebox.showinfo("Information", "The minimum value is greater than or equal to the maximum value of the propagation range.")
            return
        try:
            pasos = int(self.steps_entry.get())
        except:
            messagebox.showinfo("Information", "A valid value for the steps has not been entered; remember it must be an integer.")
            return
        try:
            dx = float(self.delta_x_entry.get())
        except:
            messagebox.showinfo("Information", "Delta x has not been entered.")
            return
        try:
            dy = float(self.delta_y_entry.get())
        except:
            messagebox.showinfo("Information", "Delta y has not been entered.")
            return
        try:
            lamb = float(self.wavelength_entry.get())
        except:
            messagebox.showinfo("Information", "The wavelength has not been entered.")
            return
        cuadrante = self.entry_param4.get()
        if cuadrante == "":
            messagebox.showinfo("Information", "A quadrant has not been specified.")
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
        #thresh = self.thresh.get()
        #if thresh == "":
        #    messagebox.showinfo("Information", "A threshold has not been specified.")
        #    return
        #try:
        #    thresh = int(thresh)
        #except:
        #    messagebox.showinfo("Information", "The threshold must be a number from 0 to 100.")
        #    return
        #if (thresh < 0 or 100 < thresh):
        #    messagebox.showinfo("Information", "The threshold must be a number from 0 to 100.")
        #    return
        #thresh = int(round(thresh / 100 * 255))
        #try:
        #    mask_len = int(self.mask_len.get())
        #except:
        #    messagebox.showinfo("Information", "Mask size must be an integer.")
        #    return
        # pixeles en el eje x y y de la imagen de origen
        x = np.arange(0, M, 1)
        y = np.arange(0, N, 1)

        #Un meshgrid para la paralelizacion
        m, n = np.meshgrid(x - (M/2), y - (N/2))
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

        fourier2=fft.fftshift(fft.fft2(fft.fftshift(U)))

        z = np.linspace(minimo,maximo,pasos)
        z_frames = []
        
        if(cuadrante=="1"):
            mascara=primer_cuadrante
            fourier=primer_cuadrante*fourier2
        if(cuadrante=="2"):
            mascara=segundo_cuadrante
            fourier=segundo_cuadrante*fourier2
        if(cuadrante=="3"):
            mascara=tercer_cuadrante
            fourier=tercer_cuadrante*fourier2
        if(cuadrante=="4"):
            mascara=cuarto_cuadrante
            fourier=cuarto_cuadrante*fourier2

        a=amplitud(fourier)

        #Encontramos la posición en x y y del máximo en el espacio de Fourier
        pos_max = np.unravel_index(np.argmax(a, axis=None), a.shape)
        mascara = crear_mascara_circular(U.shape,(pos_max[1],pos_max[0]),10)
        #Transformada insversa de fourier
        fourier= fourier*mascara
        fourier=fft.fftshift(fft.ifft2(fft.fftshift(fourier)))
        
        paso=0.2
        fin=0
        fx=pos_max[1]
        fy=pos_max[0]
        G_temp=3
        suma_maxima=0
        Fox= M/2
        Foy= N/2
        k= 2*np.pi/lamb
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
            fx=x_max_out
            fy=y_max_out
        
        theta_x=mt.asin((Fox - fx) * lamb /(M*dx))
        theta_y=mt.asin((Foy - fy) * lamb /(N*dy))
        fase= np.exp(1j*k* ((mt.sin(theta_x) * m * dx)+ ((mt.sin(theta_y) * n * dy))))
        holo=fourier*fase

        fourier = fft.fftshift(fft.fft2(fft.fftshift(holo)))  
        #holo 2 es mi caballito de batalla
        suma_maxima = float('inf')
        fx_esp = 1/(dx*M)
        fy_esp = 1/(dy*N)
        for j in z:
            temporal = Espectro_angular(fourier,j,lamb,fx_esp,fy_esp,n,m)
            temporal = fft.fftshift(fft.ifft2(fft.fftshift(temporal)))
            
            temporal = np.abs(temporal)
            #Aqui podría cambiarse el criterio, esto toca hablarlo
            suma = np.sum(temporal)
            
            if(suma_maxima>suma):
                z_fin = j
                suma_maxima = suma
        

        fourier = Espectro_angular(fourier,z_fin,lamb,fx_esp,fy_esp,n,m)
        fourier = fft.fftshift(fft.ifft2(fft.fftshift(fourier)))
        z_frames.append(z_fin)
        fase = np.abs(fourier)
        min_val = np.min(fase)
        max_val = np.max(fase)
        fase = 255*(fase - min_val) / (max_val - min_val)
        video = []
        video.append(fase)
        tiempo_final = time.time()
        tiempo= tiempo_final-tiempo_inicial
        tiempo_frame=[tiempo]
        frames.pop(0)
        
        #Ahora la versión dinámica
        paso = 0.2
        for frame in frames:
            G_temp=1
            U = asarray(frame)
            
            fin=0

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
                fx=x_max_out
                fy=y_max_out
            theta_x=mt.asin((Fox - fx) * lamb /(M*dx))
            theta_y=mt.asin((Foy - fy) * lamb /(N*dy))
            fase= np.exp(1j*k* ((mt.sin(theta_x) * m * dx)+ ((mt.sin(theta_y) * n * dy))))
            holo=fourier*fase
            fourier = fft.fftshift(fft.fft2(fft.fftshift(holo)))  
            #holo 2 es mi caballito de batalla
            suma_maxima = float('inf')
            
            for j in z:
                temporal = Espectro_angular(fourier,j,lamb,fx_esp,fy_esp,n,m)
                temporal = fft.fftshift(fft.ifft2(fft.fftshift(temporal)))  
                temporal = np.angle(temporal)
                #Aqui podría cambiarse el criterio, esto toca hablarlo
                suma = np.sum(temporal)
                    
                if(suma_maxima>suma):
                    z_fin = j
                    suma_maxima = suma
            z = np.linspace(z_fin*-1,z_fin*+1,5)

            fourier = Espectro_angular(fourier,z_fin,lamb,fx_esp,fy_esp,n,m)
            fourier = fft.fftshift(fft.ifft2(fft.fftshift(fourier))) 
            z_frames.append(z_fin)
            fase = np.abs(fourier)
            min_val = np.min(fase)
            max_val = np.max(fase)
            fase = 255*(fase - min_val) / (max_val - min_val)
            video.append(fase)


        video = [matriz.astype(np.uint8) for matriz in video]

        # Define el formato del video y crea un objeto VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Puedes ajustar el códec según tus necesidades
        video_path = 'Imagenes/Temp/video_fase_serial.avi'  # Puedes cambiar la extensión según el formato deseado (mp4, gif, etc.)
        imageio.mimsave(video_path, video, fps=15)

        print("guardo")
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
            print("Error al abrir el video.")
            exit()

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

        
# Crear la ventana principal de la aplicación
root = tk.Tk()
def on_closing():
    if messagebox.askokcancel("Exit", "Do you want to exit the tracking interface?"):
        root.destroy()
        sys.exit()  # Exit the program
app = SecondWindowApp(root)
root.protocol("WM_DELETE_WINDOW", on_closing)
root.iconbitmap("icon.ico")
root.mainloop()
