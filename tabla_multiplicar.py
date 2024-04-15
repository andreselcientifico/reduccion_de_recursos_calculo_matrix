# tablas_multiplicar = {i: {j: i * j for j in range(1, 11)} for i in range(1, 10)}

# # # Imprimir el array de diccionarios
# # for tabla in tablas_multiplicar:
# #     print(f"Tabla del {tabla['numero']}:")
# #     for multiplicador, resultado in tabla['multiplos'].items():
# #         print(f"{tabla['numero']} x {multiplicador} = {resultado}")
# #     print("\n")


# # # Función para buscar y mostrar la tabla de multiplicar
# # def mostrar_tabla(numero_busqueda):
# #     for tabla in tablas_multiplicar:
# #         if tabla['numero'] == numero_busqueda:
# #             print(f"Tabla del {tabla['numero']}:")
# #             for multiplicador, resultado in tabla['multiplos'].items():
# #                 print(f"{tabla['numero']} x {multiplicador} = {resultado}")
# #             return
# #     print(f"No se encontró la tabla del {numero_busqueda}.")

# # Solicitar al usuario un número para buscar
# # numero_ingresado = int(input("Ingresa un número para mostrar su tabla de multiplicar: "))

# # Llamar a la función para mostrar la tabla
# # mostrar_tabla(numero_ingresado)

# # Función para buscar el resultado de una operación de multiplicación

# def buscar_resultado_multiplicacion(numero1, numero2):
#     return tablas_multiplicar.get(numero1, {}).get(numero2)

# # # Solicitar al usuario dos números para la multiplicación
# # numero1 = int(input("Ingresa el primer número: "))
# # numero2 = int(input("Ingresa el segundo número: "))

# # # Llamar a la función para buscar el resultado
# # resultado = buscar_resultado_multiplicacion(numero1, numero2)

# # # Imprimir el resultado o un mensaje de error
# # print(resultado)


from numba import jit, int64,int32,prange,cuda,int16
import numpy as np
import cupy as cp
import time
import psutil

# # Crear dos matrices de 50x50 con valores aleatorios
# matriz1 = np.random.randint(1, 10, size=(1000, 1000))
# matriz2 = np.random.randint(1, 10, size=(1000, 1000))

# # Registrar el tiempo antes de la multiplicación
# tiempo_inicio = time.time()

# # Realizar la multiplicación de matrices
# resultado = np.dot(matriz1, matriz2)

# # Registrar el tiempo después de la multiplicación
# tiempo_fin = time.time()

# # Calcular el tiempo total
# tiempo_total = tiempo_fin - tiempo_inicio

# # Imprimir el resultado y el tiempo total
# print(f"Tiempo de ejecución: {tiempo_total:.16f} segundos con numpy")

# @jit(nopython=True)
# def multiplicar_matrices(matriz1, matriz2):
#     filas_matriz1, columnas_matriz1 = matriz1.shape
#     filas_matriz2, columnas_matriz2 = matriz2.shape

#     if columnas_matriz1 != filas_matriz2:
#         raise ValueError("El número de columnas de la matriz1 debe ser igual al número de filas de la matriz2")

#     resultado = np.zeros((filas_matriz1, columnas_matriz2))

#     for i in range(filas_matriz1):
#         for j in range(columnas_matriz2):
#             for k in range(filas_matriz2):
#                 resultado[i, j] += matriz1[i, k] * matriz2[k, j]

#     return resultado


# # Registrar el tiempo antes de la multiplicación
# tiempo_inicio = time.time()

# # Realizar la multiplicación de matrices
# resultado = multiplicar_matrices(matriz1, matriz2)

# # Registrar el tiempo después de la multiplicación
# tiempo_fin = time.time()

# # Calcular el tiempo total
# tiempo_total = tiempo_fin - tiempo_inicio

# # Imprimir el resultado y el tiempo total
# print(f"Tiempo de ejecución: {tiempo_total:.16f} segundos con funcion manual")

# def multiplicar_matrices(matriz1, matriz2):
#     return np.array([[sum(int(buscar_resultado_multiplicacion(matriz1[i][k], matriz2[k][j])) for k in range(len(matriz2))) for j in range(len(matriz2[0]))] for i in range(len(matriz1))])

# # Registrar el tiempo antes de la multiplicación
# tiempo_inicio = time.time()

# # Realizar la multiplicación de matrices
# resultado = multiplicar_matrices(matriz1, matriz2)

# # Registrar el tiempo después de la multiplicación
# tiempo_fin = time.time()

# # Calcular el tiempo total
# tiempo_total = tiempo_fin - tiempo_inicio

# # Imprimir el resultado y el tiempo total
# print(f"Tiempo de ejecución: {tiempo_total:.16f} segundos con funcion manual y usos de busqueda")

# @jit(int32[:, :](), nopython=True)
# def generar_tablas_multiplicar():
#     tablas_multiplicar = np.zeros((9, 10), dtype=np.int32)

#     for i in range(1, 10):
#         for j in range(1, 11):
#             tablas_multiplicar[i-1, j-1] = i * j

#     return tablas_multiplicar

# @jit(int64(int64[:, :], int64, int64), nopython=True)
# def buscar_resultado_multiplicacion(tablas_multiplicar, numero1, numero2):
#     if 1 <= numero1 <= 9 and 1 <= numero2 <= 10:
#         return tablas_multiplicar[numero1-1, numero2-1]
#     else:
#         return 0  # O cualquier valor predeterminado si los números están fuera del rango

# @jit(int32[:, :](int32[:, :], int32[:, :], int32[:, :]), nopython=True)
# def multiplicar_matrices(matriz1, matriz2, tablas_multiplicar):
#     filas_matriz1, columnas_matriz1 = matriz1.shape
#     filas_matriz2, columnas_matriz2 = matriz2.shape

#     if columnas_matriz1 != filas_matriz2:
#         raise ValueError("El número de columnas de la matriz1 debe ser igual al número de filas de la matriz2")

#     resultado = np.zeros((filas_matriz1, columnas_matriz2), dtype=np.int32)

#     for i in range(filas_matriz1):
#         for j in range(columnas_matriz2):
#             for k in range(filas_matriz2):
#                 resultado[i, j] += tablas_multiplicar[matriz1[i, k]-1, matriz2[k, j]-1]
#     return resultado

# tablas_multiplicar = generar_tablas_multiplicar()
# # Registrar el tiempo antes de la multiplicación
# tiempo_inicio = time.time()

# # Realizar la multiplicación de matrices
# resultado = multiplicar_matrices(matriz1, matriz2,tablas_multiplicar)

# # Registrar el tiempo después de la multiplicación
# tiempo_fin = time.time()

# # Calcular el tiempo total
# tiempo_total = tiempo_fin - tiempo_inicio

# # Imprimir el resultado y el tiempo total
# print(f"Tiempo de ejecución: {tiempo_total:.16f} segundos con numba y usos de busqueda")

# def generar_tablas_multiplicar():
#     tablas_multiplicar = [[0] * 10 for _ in range(9)]

#     for i in range(1, 10):
#         for j in range(1, 11):
#             tablas_multiplicar[i-1][j-1] = i * j

#     return tablas_multiplicar

# def buscar_resultado_multiplicacion(tablas_multiplicar, numero1, numero2):
#     if 1 <= numero1 <= 9 and 1 <= numero2 <= 10:
#         return tablas_multiplicar[numero1-1][numero2-1]
#     else:
#         return 0

# def multiplicar_matrices(matriz1, matriz2, tablas_multiplicar):
#     filas_matriz1 = len(matriz1)
#     columnas_matriz1 = len(matriz1[0])
#     filas_matriz2 = len(matriz2)
#     columnas_matriz2 = len(matriz2[0])

#     if columnas_matriz1 != filas_matriz2:
#         raise ValueError("El número de columnas de la matriz1 debe ser igual al número de filas de la matriz2")

#     resultado = [[0] * columnas_matriz2 for _ in range(filas_matriz1)]

#     for i in range(filas_matriz1):
#         for j in range(columnas_matriz2):
#             for k in range(filas_matriz2):
#                 resultado[i][j] += tablas_multiplicar[matriz1[i][k]-1][matriz2[k][j]-1]
#     return resultado

# tablas_multiplicar = generar_tablas_multiplicar()
# # Registrar el tiempo antes de la multiplicación
# tiempo_inicio = time.time()

# # Realizar la multiplicación de matrices
# resultado = multiplicar_matrices(matriz1, matriz2,tablas_multiplicar)

# # Registrar el tiempo después de la multiplicación
# tiempo_fin = time.time()

# # Calcular el tiempo total
# tiempo_total = tiempo_fin - tiempo_inicio

# # Imprimir el resultado y el tiempo total
# print(f"Tiempo de ejecución: {tiempo_total:.16f} segundos con python y usos de busquedda")

# def generar_tablas_multiplicar():
#     return [[i * j for j in range(1, 11)] for i in range(1, 10)]

# def buscar_resultado_multiplicacion(tablas_multiplicar, numero1, numero2):
#     return tablas_multiplicar[numero1-1][numero2-1] if 1 <= numero1 <= 9 and 1 <= numero2 <= 10 else 0

# def multiplicar_matrices(matriz1, matriz2, tablas_multiplicar):
#     if len(matriz1[0]) != len(matriz2):
#         raise ValueError("El número de columnas de la matriz1 debe ser igual al número de filas de la matriz2")

#     return [
#         [
#             sum(tablas_multiplicar[matriz1[i][k]-1][matriz2[k][j]-1] for k in range(len(matriz2)))
#             for j in range(len(matriz2[0]))
#         ]
#         for i in range(len(matriz1))
#     ]


# tablas_multiplicar = generar_tablas_multiplicar()
# # Registrar el tiempo antes de la multiplicación
# tiempo_inicio = time.time()

# # Realizar la multiplicación de matrices
# resultado = multiplicar_matrices(matriz1, matriz2,tablas_multiplicar)

# # Registrar el tiempo después de la multiplicación
# tiempo_fin = time.time()

# # Calcular el tiempo total
# tiempo_total = tiempo_fin - tiempo_inicio

# # Imprimir el resultado y el tiempo total
# print(f"Tiempo de ejecución: {tiempo_total:.16f} segundos python comprehensions y uso de busqueda")

# # import subprocess

# # # Compilar el código Fortran usando el compilador Fortran
# # subprocess.run(["gfortran", "-o", "fortran", "fortran_code.f90", "-mconsole"])

# # # Ejecutar el programa Fortran compilado
# # subprocess.run(["./fortran"])

matriz1 = np.random.randint(1, 10, size=(1000, 1000))
matriz2 = np.random.randint(1, 10, size=(1000, 1000))

# @jit(int32[:, :](), nopython=True)
# def generar_tablas_multiplicar():
#     tablas_multiplicar = np.zeros((9, 10), dtype=np.int32)

#     for i in range(1, 10):
#         for j in range(1, 11):
#             tablas_multiplicar[i-1, j-1] = i * j

#     return tablas_multiplicar

# @jit(int64(int64[:, :], int64, int64), nopython=True)
# def buscar_resultado_multiplicacion(tablas_multiplicar, numero1, numero2):
#     if 1 <= numero1 <= 9 and 1 <= numero2 <= 10:
#         return tablas_multiplicar[numero1-1, numero2-1]
#     else:
#         return 0  # O cualquier valor predeterminado si los números están fuera del rango

# @jit(int32[:, :](int32[:, :], int32[:, :], int32[:, :]), nopython=True)
# def multiplicar_matrices(matriz1, matriz2, tablas_multiplicar):
#     filas_matriz1, columnas_matriz1 = matriz1.shape
#     filas_matriz2, columnas_matriz2 = matriz2.shape

#     if columnas_matriz1 != filas_matriz2:
#         raise ValueError("El número de columnas de la matriz1 debe ser igual al número de filas de la matriz2")

#     resultado = np.zeros((filas_matriz1, columnas_matriz2), dtype=np.int32)

#     for i in range(filas_matriz1):
#         for j in range(columnas_matriz2):
#             for k in range(filas_matriz2):
#                 resultado[i, j] += tablas_multiplicar[matriz1[i, k]-1, matriz2[k, j]-1]
#     return resultado

# tablas_multiplicar = generar_tablas_multiplicar()
# # Registrar el tiempo antes de la multiplicación
# tiempo_inicio = time.time()

# # Realizar la multiplicación de matrices
# resultado = multiplicar_matrices(matriz1, matriz2,tablas_multiplicar)

# # Registrar el tiempo después de la multiplicación
# tiempo_fin = time.time()

# # Calcular el tiempo total
# tiempo_total = tiempo_fin - tiempo_inicio

# # Imprimir el resultado y el tiempo total
# print(f"Tiempo de ejecución: {tiempo_total:.16f} segundos con numba, tecnica de busqueda")


@jit(int32[:, :](), nopython=True, parallel=True)
def generar_tablas_multiplicar():
    tablas_multiplicar = np.zeros((9, 10), dtype=np.int32)

    for i in range(1, 10):
        tablas_multiplicar[i-1, :] = np.arange(1, 11) * i

    return set(tablas_multiplicar)

@jit(int32[:, :](int32[:, :], int32[:, :], int32[:, :]), nopython=True, parallel=True)
def multiplicar_matrices(matriz1, matriz2, tablas_multiplicar):
    filas_matriz1, columnas_matriz1 = matriz1.shape
    filas_matriz2, columnas_matriz2 = matriz2.shape

    if columnas_matriz1 != filas_matriz2:
        raise ValueError("El número de columnas de la matriz1 debe ser igual al número de filas de la matriz2")

    resultado = np.zeros((filas_matriz1, columnas_matriz2), dtype=np.int32)

    for i in prange(filas_matriz1):
        for j in range(columnas_matriz2):
            for k in range(filas_matriz2):
                resultado[i, j] += tablas_multiplicar[matriz1[i, k]-1, matriz2[k, j]-1]
    return resultado

tablas_multiplicar = generar_tablas_multiplicar()

import sys 

tamaño_en_memoria = sys.getsizeof(tablas_multiplicar)
print(f"Tamaño en memoria de la Tabla: {tamaño_en_memoria} bytes")

# Registrar el tiempo antes de la multiplicación
tiempo_inicio = time.time()

# Realizar la multiplicación de matrices
resultado = multiplicar_matrices(matriz1, matriz2,tablas_multiplicar)

# Obtener el uso de CPU después de la multiplicación
cpu_fin = psutil.cpu_percent()

# Registrar el tiempo después de la multiplicación
tiempo_fin = time.time()

# Calcular el tiempo total
tiempo_total = tiempo_fin - tiempo_inicio

# Imprimir el resultado y el tiempo total
print(f"Tiempo de ejecución: {tiempo_total:.16f} segundos con numba, tecnica de busqueda, tecnicas de local buffer, y calculo por bloques")
print(f"Uso de CPU (%): {cpu_fin:.2f}")

@jit(int32[:, :](int32[:, :], int32[:, :], int32[:, :]), nopython=True, parallel=True)
def multiplicar_matrices(matriz1, matriz2, tablas_multiplicar):
    filas_matriz1, columnas_matriz1 = matriz1.shape
    filas_matriz2, columnas_matriz2 = matriz2.shape

    if columnas_matriz1 != filas_matriz2:
        raise ValueError("El número de columnas de la matriz1 debe ser igual al número de filas de la matriz2")

    resultado = np.zeros((filas_matriz1, columnas_matriz2), dtype=np.int32)

    for j in prange(columnas_matriz2):
        for i in range(filas_matriz1):
            for k in range(filas_matriz2):
                resultado[i, j] += tablas_multiplicar[matriz1[i, k]-1, matriz2[k, j]-1]
    return resultado

tablas_multiplicar = generar_tablas_multiplicar()

# Registrar el tiempo antes de la multiplicación
tiempo_inicio = time.time()

# Realizar la multiplicación de matrices
resultado = multiplicar_matrices(matriz1, matriz2,tablas_multiplicar)

# Obtener el uso de CPU después de la multiplicación
cpu_fin = psutil.cpu_percent()

# Registrar el tiempo después de la multiplicación
tiempo_fin = time.time()

# Calcular el tiempo total
tiempo_total = tiempo_fin - tiempo_inicio

# Imprimir el resultado y el tiempo total
print(f"Tiempo de ejecución: {tiempo_total:.16f} segundos con numba, tecnica de busqueda, tecnicas de local buffer, calculo por bloques y matriz transpuesta")
print(f"Uso de CPU (%): {cpu_fin:.2f}")

# Registrar el tiempo antes de la multiplicación
tiempo_inicio = time.time()

# Realizar la multiplicación de matrices
resultado = np.dot(matriz1, matriz2)

# Obtener el uso de CPU después de la multiplicación
cpu_fin = psutil.cpu_percent()

# Registrar el tiempo después de la multiplicación
tiempo_fin = time.time()

# Calcular el tiempo total
tiempo_total = tiempo_fin - tiempo_inicio

# Imprimir el resultado y el tiempo total
print(f"Tiempo de ejecución: {tiempo_total:.16f} segundos con numpy")
print(f"Uso de CPU (%): {cpu_fin:.2f}")

@jit(int32[:, :](int32[:, :], int32[:, :]), nopython=True, parallel=True)
def multiplicar_matrices(matriz1, matriz2):
    filas_matriz1, columnas_matriz1 = matriz1.shape
    filas_matriz2, columnas_matriz2 = matriz2.shape

    if columnas_matriz1 != filas_matriz2:
        raise ValueError("El número de columnas de la matriz1 debe ser igual al número de filas de la matriz2")

    resultado = np.zeros((filas_matriz1, columnas_matriz2), dtype=np.int32)

    for i in prange(filas_matriz1):
        for j in range(columnas_matriz2):
            for k in range(filas_matriz2):
                resultado[i, j] += matriz1[i, k]-1 * matriz2[k, j]-1
    return resultado


# Registrar el tiempo antes de la multiplicación
tiempo_inicio = time.time()

# Realizar la multiplicación de matrices
resultado = multiplicar_matrices(matriz1, matriz2)

# Registrar el tiempo después de la multiplicación
tiempo_fin = time.time()

# Calcular el tiempo total
tiempo_total = tiempo_fin - tiempo_inicio

# Imprimir el resultado y el tiempo total
print(f"Tiempo de ejecución: {tiempo_total:.16f} segundos con numba multiplicacion")


@cuda.jit('void(int32[:, :], int32[:, :], int32[:, :])', device=True)
def gpu_multiplicar_kernel(matriz1, matriz2, resultado):
    filas_matriz1, columnas_matriz1 = matriz1.shape
    filas_matriz2, columnas_matriz2 = matriz2.shape

    # Inicializar el array resultado en el dispositivo CUDA
    for i in range(filas_matriz1):
        for j in range(columnas_matriz2):
            resultado[i, j] = 0

    # Realizar la multiplicación de matrices
    for i in range(filas_matriz1):
        for j in range(columnas_matriz2):
            for k in range(filas_matriz2):
                resultado[i, j] += matriz1[i, k] * matriz2[k, j]

@cuda.jit
def gpu_multiplicar(matriz1, matriz2, resultado):
    i, j = cuda.grid(2)
    if i < resultado.shape[0] and j < resultado.shape[1]:
        for k in range(matriz1.shape[1]):
            resultado[i, j] += matriz1[i, k] * matriz2[k, j]

# Definir las dimensiones de las matrices
filas_matriz1, columnas_matriz1 = 1000, 1000
filas_matriz2, columnas_matriz2 = 1000, 1000

# Crear un array en el dispositivo CUDA para almacenar el resultado
resultado_device = cuda.device_array((filas_matriz1, columnas_matriz2), dtype=np.int32)

# Configurar la cuadrícula y los bloques
blockdim = (16, 16)
griddim = (filas_matriz1 // blockdim[0] + 1, columnas_matriz2 // blockdim[1] + 1)

# Registrar el tiempo antes de la multiplicación
tiempo_inicio = time.time()

# Llamar a la función del dispositivo CUDA
gpu_multiplicar[griddim, blockdim](resultado_device, resultado_device, resultado_device)

# Registrar el tiempo después de la multiplicación
tiempo_fin = time.time()

# Copiar el resultado de vuelta al host (opcional si es necesario)
resultado_host = resultado_device.copy_to_host()

# Calcular el tiempo total
tiempo_total = tiempo_fin - tiempo_inicio

# Imprimir el resultado y el tiempo total
print(f"Tiempo de ejecución: {tiempo_total:.16f} segundos con numba en gpu cuda")

# # Definir las dimensiones de las matrices
# filas_matriz1, columnas_matriz1 = 1000, 1000
# filas_matriz2, columnas_matriz2 = 1000, 1000

# # Crear matrices aleatorias en la GPU
# matriz1_gpu = cp.random.randint(1, 10, (filas_matriz1, columnas_matriz1), dtype=cp.int32)
# matriz2_gpu = cp.random.randint(1, 10, (filas_matriz2, columnas_matriz2), dtype=cp.int32)

# # Registrar el tiempo antes de la multiplicación
# tiempo_inicio = time.time()

# # Realizar la multiplicación de matrices en la GPU con CuPy
# resultado_gpu = cp.dot(matriz1_gpu, matriz2_gpu)

# # Registrar el tiempo después de la multiplicación
# tiempo_fin = time.time()

# # Transferir el resultado de vuelta al host si es necesario
# resultado_cpu = cp.asnumpy(resultado_gpu)

# # Calcular el tiempo total
# tiempo_total = tiempo_fin - tiempo_inicio

# # Imprimir el resultado y el tiempo total
# print(f"Tiempo de ejecución: {tiempo_total:.16f} segundos con CuPy en GPU")

@jit(int32[:, :](int32[:, :], int32[:, :], int32[:, :]), nopython=True, parallel=True)
def multiplicar_matrices(matriz1, matriz2, tablas_multiplicar):
    filas_matriz1, columnas_matriz1 = matriz1.shape
    filas_matriz2, columnas_matriz2 = matriz2.shape

    if columnas_matriz1 != filas_matriz2:
        raise ValueError("El número de columnas de la matriz1 debe ser igual al número de filas de la matriz2")

    resultado = np.zeros((filas_matriz1, columnas_matriz2), dtype=np.int32)

    for i in prange(filas_matriz1):
        for j in range(columnas_matriz2):
            for k in range(filas_matriz2):
                resultado[i, j] += tablas_multiplicar[matriz1[i, k]-1, matriz2[k, j]-1] if matriz1[i, k] > matriz2[k, j] else tablas_multiplicar[matriz2[k, j]-1, matriz1[i, k]-1]
    return resultado

# Registrar el tiempo antes de la multiplicación
tiempo_inicio = time.time()

# Realizar la multiplicación de matrices
resultado = multiplicar_matrices(matriz1, matriz2,tablas_multiplicar)

# Registrar el tiempo después de la multiplicación
tiempo_fin = time.time()

# Calcular el tiempo total
tiempo_total = tiempo_fin - tiempo_inicio

# Imprimir el resultado y el tiempo total
print(f"Tiempo de ejecución: {tiempo_total:.16f} segundos con numba, tecnica de busqueda, matrix tabla de multiplicar")

@jit(int32[:, :](int32[:, :], int32[:, :]), nopython=True, parallel=True)
def multiplicar_matrices(matriz1, matriz2):
    filas_matriz1, columnas_matriz1 = matriz1.shape
    filas_matriz2, columnas_matriz2 = matriz2.shape

    if columnas_matriz1 != filas_matriz2:
        raise ValueError("El número de columnas de la matriz1 debe ser igual al número de filas de la matriz2")

    resultado = np.zeros((filas_matriz1, columnas_matriz2), dtype=np.int32)

    for i in prange(filas_matriz1):
        for j in range(columnas_matriz2):
            for k in range(filas_matriz2):
                resultado[i, j] += matriz1[i, k] * matriz2[k, j]
    return resultado


# Registrar el tiempo antes de la multiplicación
tiempo_inicio = time.time()

# Realizar la multiplicación de matrices
resultado = multiplicar_matrices(matriz1, matriz2)

# Registrar el tiempo después de la multiplicación
tiempo_fin = time.time()

# Calcular el tiempo total
tiempo_total = tiempo_fin - tiempo_inicio

# Imprimir el resultado y el tiempo total
print(f"Tiempo de ejecución: {tiempo_total:.16f} segundos con numba y calculo de matriz normal")

@jit(int32[:, :](int32[:, :], int32[:, :], int32[:, :]), nopython=True, parallel=True)
def multiplicar_matrices(matriz1, matriz2, tablas_multiplicar):
    filas_matriz1, columnas_matriz1 = matriz1.shape
    filas_matriz2, columnas_matriz2 = matriz2.shape

    if columnas_matriz1 != filas_matriz2:
        raise ValueError("El número de columnas de la matriz1 debe ser igual al número de filas de la matriz2")

    resultado = np.zeros((filas_matriz1, columnas_matriz2), dtype=np.int32)

    for i in prange(filas_matriz1):
        for j in range(columnas_matriz2):
            suma = 0
            for k in range(filas_matriz2):
                suma += tablas_multiplicar[matriz1[i, k]-1, matriz2[k, j]-1]
            resultado[i, j] = suma
    return resultado

# Registrar el tiempo antes de la multiplicación
tiempo_inicio = time.time()

# Realizar la multiplicación de matrices
resultado = multiplicar_matrices(matriz1, matriz2,tablas_multiplicar)

# Registrar el tiempo después de la multiplicación
tiempo_fin = time.time()

# Calcular el tiempo total
tiempo_total = tiempo_fin - tiempo_inicio

# Imprimir el resultado y el tiempo total
print(f"Tiempo de ejecución: {tiempo_total:.16f} segundos con numba, tecnica de busqueda, tecnicas de local buffer")

@jit(int32[:, :](int32[:, :], int32[:, :], int32[:, :]), nopython=True, parallel=True)
def multiplicar_matrices(matriz1, matriz2, tablas_multiplicar):
    filas_matriz1, columnas_matriz1 = matriz1.shape
    filas_matriz2, columnas_matriz2 = matriz2.shape

    if columnas_matriz1 != filas_matriz2:
        raise ValueError("El número de columnas de la matriz1 debe ser igual al número de filas de la matriz2")

    resultado = np.zeros((filas_matriz1, columnas_matriz2), dtype=np.int32)
    flattened_tablas = tablas_multiplicar.flatten()
    for i in prange(filas_matriz1):
        for j in range(columnas_matriz2):
            for k in range(filas_matriz2):
                resultado[i, j] += flattened_tablas[(matriz1[i, k] - 1) * tablas_multiplicar.shape[1] + (matriz2[k, j] - 1)]
    return resultado

# Registrar el tiempo antes de la multiplicación
tiempo_inicio = time.time()

# Realizar la multiplicación de matrices
resultado = multiplicar_matrices(matriz1, matriz2,tablas_multiplicar)

# Registrar el tiempo después de la multiplicación
tiempo_fin = time.time()

# Calcular el tiempo total
tiempo_total = tiempo_fin - tiempo_inicio

# Imprimir el resultado y el tiempo total
print(f"Tiempo de ejecución: {tiempo_total:.16f} segundos con numba, matriz aplanada, tecnica de busqueda")

@jit(int32[:, :](int32[:, :], int32[:, :], int32[:, :]), nopython=True, parallel=True)
def multiplicar_matrices(matriz1, matriz2, tablas_multiplicar):
    filas_matriz1, columnas_matriz1 = matriz1.shape
    filas_matriz2, columnas_matriz2 = matriz2.shape

    if columnas_matriz1 != filas_matriz2:
        raise ValueError("El número de columnas de la matriz1 debe ser igual al número de filas de la matriz2")
    flattened = tablas_multiplicar.flatten()
    resultado = np.zeros((filas_matriz1, columnas_matriz2), dtype=np.int32)
    for j in prange(columnas_matriz2):
        for i in range(filas_matriz1):
            for k in range(filas_matriz2):
                resultado[i, j] += flattened[(matriz1[i, k] - 1) * tablas_multiplicar.shape[1] + (matriz2[k, j] - 1)]
    return resultado



# Registrar el tiempo antes de la multiplicación
tiempo_inicio = time.time()

# Realizar la multiplicación de matrices
resultado = multiplicar_matrices(matriz1, matriz2,tablas_multiplicar)

# Obtener el uso de CPU despues de la multiplicación
cpu_fin = psutil.cpu_percent()

# Registrar el tiempo después de la multiplicación
tiempo_fin = time.time()

# Calcular el tiempo total
tiempo_total = tiempo_fin - tiempo_inicio

# Imprimir el resultado y el tiempo total
print(f"Tiempo de ejecución: {tiempo_total:.16f} segundos con numba, matriz aplanada, tecnica de busqueda, matriz transpuesta")
print(f"Uso de CPU (%): {cpu_fin:.2f}")