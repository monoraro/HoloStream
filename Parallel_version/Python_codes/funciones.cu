#include <math.h>
#include <stdio.h>
#include <cuComplex.h>
#include <limits.h>

//esta parte es para el propagador
              

                   /*FUNCIONES PARALELIZADAS DEL ALGORITMO*/

__global__ void fase_refe(cuComplex *holo, cuComplex *holo2, cuComplex *ref, float *m, float *n,  int N, int M, float k, float fox, float foy, float fx, float fy, float lamb, float dx, float dy)
{
    int fila = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int i2 = (col * N) + fila;

    // Precalcula senos y cosenos
    double temp_x = (fox - fx) * lamb / (M * dx);
    double temp_y = (foy - fy) * lamb / (N * dx);
    double tx = asin(temp_x);
    double ty = asin(temp_y);
    float sin_tx = sin(tx);
    float sin_ty = sin(ty);

    // Cálculo de la fase
    float temporal = k * ((sin_tx * m[i2] * dx) + (sin_ty * n[i2] * dy));
    float cos_temporal = cos(temporal);
    float sin_temporal = sin(temporal);

    // Guardar en memoria global
    ref[i2].x = cos_temporal;
    ref[i2].y = sin_temporal;

    // Multiplicación directa en memoria global
    holo2[i2] = cuCmulf(holo[i2], ref[i2]);

}
              

                   /*Función paralelizada para encontrar el máximo junto con su respectiva posiciAmplitudón
                
                   //Como creo vectores temporales jajaja
                    //Ya lo programaron en pycuda jajaja, Pero la teoria es interesante, es como un arbol genealógico al revéz
                    */
__global__ void coordenadas_maximo(float *matrix, int rows, int cols, int *max_position){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid == 0)
    {
        float max_value = -1e9;  // Valor inicial muy pequeño
        int max_index = -1;

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (matrix[i * cols + j] > max_value)
                {
                    max_value = matrix[i * cols + j];
                    max_index = i * cols + j;
                }
            }
        }

        *max_position = max_index;
    }
}
__global__ void reseteo(cuComplex *holo,int N, int M){
    //Estos son parametros necesarios para definición
    int fila = blockIdx.x*blockDim.x+threadIdx.x;
	int col = blockIdx.y*blockDim.y+threadIdx.y;
    int i2= ((fila*M)+col);
    holo[i2].x=0;
    holo[i2].y=0;           
}

                   /*FUNCION DE SHIFTEO CUANDO SE TIENE LA INFORMACIÓN EN 2 COMPLEX 64 */
__global__ void fft_shift(cuComplex *final,cuComplex *dest_gpu, int N, int M)
{
	int n2 = N / 2;
	int m2 = M / 2;

	//Descriptores de cada hilo
    int fila = (blockIdx.x*blockDim.x + threadIdx.x);
	int col = (blockIdx.y*blockDim.y+ threadIdx.y);
	int fila2 = fila + n2;
	int col2 = col + m2;
    
    final[(fila2*M + col2)].x = dest_gpu[(fila*M+col)].x;  //Guardo el primer cuadrante
	final[(fila*M+col)].x = dest_gpu[(fila2*M+col2)].x;  //en el primer cuadrante estoy poniendo lo que hay en el tercero
    final[(fila2*M + col2)].y = dest_gpu[(fila*M+col)].y;  //Lo mismo pero para los imaginarios
	final[(fila*M+col)].y = dest_gpu[(fila2*M+col2)].y;
    
    final[(fila*M + col2)].x = dest_gpu[(fila2*M+col)].x;  //Guardo el segundo cuadrante
	final[(fila2*M+col)].x = dest_gpu[(fila*M+col2)].x;  //en el segundo cuadrante estoy poniendo lo que hay en el tercer cuadrante
    final[(fila*M + col2)].y = dest_gpu[(fila2*M+col)].y;  //Lo mismo pero para los imaginarios
	final[(fila2*M+col)].y = dest_gpu[(fila*M+col2)].y;  
}

                   /*FUNCION DE SHIFTEO SE TIENE LA INFORMACIÓN NO COMPLEJA */              
__global__ void fft_shift_var_no_compleja(cuComplex *final,float *U_gpu, int N, int M)
{
	int n2 = N / 2;
	int m2 = M / 2;

	//Descriptores de cada hilo
    int fila = blockIdx.x*blockDim.x+ threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y;
	int fila2 = blockIdx.x*blockDim.x + threadIdx.x + n2;
	int col2 = blockIdx.y*blockDim.y + threadIdx.y + m2;
    final[fila*M+col].x = U_gpu[((fila2*M) + (col2))];
    final[fila*M+col].y = 0;
    final[fila2*M+col2].x = U_gpu[((fila*M) + (col))];
    final[fila2*M+col2].y = 0;
    final[fila*M+col2].x = U_gpu[((fila2*M) + (col))];
    final[fila*M+col2].y = 0;
    final[fila2*M+col].x =U_gpu[((fila*M) + (col2))];
    final[fila2*M+col].y = 0;
}

__global__ void thresholding(float *image, int N, int M, float threshold)
{
    int fila = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
                   
    int i2= ((fila*M)+col);
    image[i2] = (image[i2] > threshold) ? 1.0 : 0.0;
}
                   /*mascaras para una imagen dada */                  
__global__ void mascara_1er_cuadrante(cuComplex *final,cuComplex *U_gpu,float *mascara, int N, int M)
{

	//Descriptores de cada hilo
    int fila = blockIdx.x*blockDim.x+ threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y;

    final[fila*M+(col)].x = U_gpu[fila*M+(col)].x * mascara[fila*M+(col)];
    final[fila*M+(col)].y = U_gpu[fila*M+(col)].y * mascara[fila*M+(col)];

} 
                   /*ESTA PARTE ES MERAMENTE PARA NORMALIZAR UNA MATRIZ */

                   
__global__ void Normalizar(float *U_gpu, int N, int M, float *minimo, float *maximo)
{
	//Descriptores de cada hilo
    int fila = (blockIdx.x*blockDim.x + threadIdx.x);
	int col = (blockIdx.y*blockDim.y+ threadIdx.y);
    float mini=minimo[0];
    float maxi=maximo[0];
    U_gpu[(fila*M + col)] = (U_gpu[(fila*M + col)]-mini)/(maxi-mini); //Calculo la amplitud
}           
                   /*ESTA PARTE ES PARA LA RECONSTRUCCIÓN EN FASE, AMPLITUD O INSTENSIDAD*/
   
__global__ void Amplitud(float *U_gpu, cuComplex *dest_gpu, int N, int M)
{
	//Descriptores de cada hilo
    int fila = (blockIdx.x*blockDim.x + threadIdx.x);
	int col = (blockIdx.y*blockDim.y+ threadIdx.y);
    
    U_gpu[(fila*M + col)] = sqrt(pow((dest_gpu[(fila*M+col)].x),2) + (pow((dest_gpu[(fila*M+col)].y),2)));  //Calculo la amplitud
}
__global__ void Amplitud_log(float *U_gpu, cuComplex *dest_gpu, int N, int M)
{
	//Descriptores de cada hilo
    int fila = (blockIdx.x*blockDim.x + threadIdx.x);
	int col = (blockIdx.y*blockDim.y+ threadIdx.y);
    
    U_gpu[(fila*M + col)] = logf(sqrt(pow((dest_gpu[(fila*M+col)].x),2) + (pow((dest_gpu[(fila*M+col)].y),2)))+1);  //Calculo la amplitud
}
                   
__global__ void Intensidad(float *U_gpu, cuComplex *dest_gpu, int N, int M)
{

	//Descriptores de cada hilo
    int fila = (blockIdx.x*blockDim.x + threadIdx.x);
	int col = (blockIdx.y*blockDim.y+ threadIdx.y);
    
    U_gpu[(fila*M + col)] = pow((dest_gpu[(fila*M+col)].x),2) + (pow((dest_gpu[(fila*M+col)].y),2));  //Calculo la intensidad
}

__global__ void Fase(float *U_gpu, cuComplex *dest_gpu, int N, int M){
                   
	//Descriptores de cada hilo
    int fila = (blockIdx.x*blockDim.x + threadIdx.x);
	int col = (blockIdx.y*blockDim.y+ threadIdx.y);
    
    U_gpu[(fila*M + col)] = atan2f(dest_gpu[(fila*M+col)].y, dest_gpu[(fila*M+col)].x); //Calculo la fase
}
__global__ void espectro_angular(cuComplex *final,cuComplex *dest_copia, cuComplex *dest_gpu, float *m_gpu, float *n_gpu, float fx, float fy, float z,float pi, int M, float lamb)
{

    int fila = blockIdx.x*blockDim.x+threadIdx.x;
	int col = blockIdx.y*blockDim.y+threadIdx.y;
    int i2= ((fila*M)+col);

    dest_copia[i2].x = __cosf(2*pi*z*pow(pow(1/lamb,2)-(pow(fx*n_gpu[i2],2)+pow(fy*m_gpu[i2],2)),0.5));
    dest_copia[i2].y = __sinf(2*pi*z*pow(pow(1/lamb,2)-(pow(fx*n_gpu[i2],2)+pow(fy*m_gpu[i2],2)),0.5));
    dest_gpu[i2].y = ((final[i2].y * dest_copia[i2].x) + (final[i2].x * dest_copia[i2].y));
    dest_gpu[i2].x = ((final[i2].x * dest_copia[i2].x) - (final[i2].y * dest_copia[i2].y));


}
