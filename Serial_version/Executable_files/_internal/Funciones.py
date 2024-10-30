"""
Funciones auxiliares
"""

import numpy as np
import math as mt
from matplotlib import pyplot as plt
from PIL import Image
import glob
import cv2
from datetime import datetime
##Funciones hechas dentro de CUDA
#Funciones desde python
def hora_y_fecha():
    return datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

#Función de lectura de una imagen dada
def lectura(name_file):
    replica = Image.open("./" +
                         str(name_file)).convert('L')
    replica.save("./Imagenes/copia.png")
    return (replica)
def ajuste_tamano(archivo):
    N_, M_ = archivo.size
    N_ = N_ // 64
    M_ = M_ // 64
    replica = np.resize(archivo, (M_*64,N_*64))
    return (replica)

#Funcion para crear mascara circular
def crear_mascara_circular(shape, centro, radio):
    # Crear una imagen en blanco (negra) del tamaño especificado
    mascara = np.zeros(shape, dtype=np.uint8)
    
    # Dibujar el círculo en la máscara
    cv2.circle(mascara, centro, radio, (255, 255, 255), -1)
    
    return mascara

def ajuste_tamano1(archivo):
    N_, M_ = archivo.shape
    N_ = N_ // 64
    M_ = M_ // 64
    vector = archivo.flatten()
    replica = np.resize(archivo, (N_*64,M_*64))
    return (replica)

def lectura_continua(direccion):
    cv_img = []
    arepa=glob.glob(direccion)
    archivos = sorted(arepa, key=lambda x: x, reverse=True)
    for img in archivos:
        print(img)
        n = Image.open(img).convert('L')
        cv_img.append(n)
    return (cv_img)

# Función para el guardado de la imagen
def guardado(name_out, matriz):
    
    resultado = Image.fromarray(matriz)
    resultado = resultado.convert('RGB')
    resultado.save("./"+str(name_out))

# Función para graficar y ponerle nombre a los ejes
def mostrar(matriz, titulo="a", ejex="b", ejey="c"):
    plt.imshow(matriz, cmap='gray')
    plt.title(str(titulo))
    plt.xlabel(str(ejex))
    plt.ylabel(str(ejey))
    plt.show()

# Calculo de la magnitud, pero hagamos esto en cuda
def amplitud(matriz):
    amplitud = np.abs(matriz)
    return (amplitud)



def intensidad(matriz):
    intensidad = np.abs(matriz)
    intensidad = np.power(intensidad, 2)
    return (intensidad)


def fase(matriz):
    fase = np.angle(matriz, deg=False)
    return (fase)


def dual_img(image1, image2, title):
    fig, (ax1, ax2) = plt.subplots(ncols=2)

    # Graficar las imágenes en cada subplot
    ax1.imshow(image1, cmap='gray')
    ax2.imshow(image2, cmap='gray')

    # Personalizar los subplots y la figura
    ax1.set_title('Entrada')
    ax2.set_title('Salida')
    fig.suptitle(str(title))

    # Mostrar el gráfico
    plt.show()


def dual_save(image1, image2, title):
    fig, (ax1, ax2) = plt.subplots(ncols=2)

    # Graficar las imágenes en cada subplot
    ax1.imshow(image1, cmap='gray')
    ax2.imshow(image2, cmap='gray')

    # Personalizar los subplots y la figura
    ax1.set_title('Entrada')
    ax2.set_title('Salida')
    fig.suptitle(str(title))

    # Mostrar el gráfico
    plt.savefig('./Imagenes/guardado.png', dpi=1000)

def Espectro_angular(entrada,z,lamb,fx,fy,P,Q):
    result = np.exp2(1j*z*np.pi*np.sqrt(np.power(1/lamb, 2) -
              (np.power(fx*P, 2) + np.power(Q*fy, 2))))
    result = entrada*result
    return result

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

def normalizar(matriz):
    
    min_val = np.min(matriz)
    max_val = np.max(matriz)
    matriz = 255*(matriz - min_val) / (max_val - min_val)
    return matriz