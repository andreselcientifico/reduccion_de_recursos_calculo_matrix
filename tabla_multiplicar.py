tablas_multiplicar = {i: {j: i * j for j in range(1, 11)} for i in range(1, 10)}

# # Imprimir el array de diccionarios
# for tabla in tablas_multiplicar:
#     print(f"Tabla del {tabla['numero']}:")
#     for multiplicador, resultado in tabla['multiplos'].items():
#         print(f"{tabla['numero']} x {multiplicador} = {resultado}")
#     print("\n")


# # Función para buscar y mostrar la tabla de multiplicar
# def mostrar_tabla(numero_busqueda):
#     for tabla in tablas_multiplicar:
#         if tabla['numero'] == numero_busqueda:
#             print(f"Tabla del {tabla['numero']}:")
#             for multiplicador, resultado in tabla['multiplos'].items():
#                 print(f"{tabla['numero']} x {multiplicador} = {resultado}")
#             return
#     print(f"No se encontró la tabla del {numero_busqueda}.")

# Solicitar al usuario un número para buscar
# numero_ingresado = int(input("Ingresa un número para mostrar su tabla de multiplicar: "))

# Llamar a la función para mostrar la tabla
# mostrar_tabla(numero_ingresado)

# Función para buscar el resultado de una operación de multiplicación

def buscar_resultado_multiplicacion(numero1, numero2):
    return tablas_multiplicar.get(numero1, {}).get(numero2)

# # Solicitar al usuario dos números para la multiplicación
# numero1 = int(input("Ingresa el primer número: "))
# numero2 = int(input("Ingresa el segundo número: "))

# # Llamar a la función para buscar el resultado
# resultado = buscar_resultado_multiplicacion(numero1, numero2)

# # Imprimir el resultado o un mensaje de error
# print(resultado)


from numba import jit, int64,int32
import numpy as np
import time

# Crear dos matrices de 50x50 con valores aleatorios
matriz1 = np.random.randint(1, 10, size=(200, 100))
matriz2 = np.random.randint(1, 10, size=(100, 200))

# Registrar el tiempo antes de la multiplicación
tiempo_inicio = time.time()

# Realizar la multiplicación de matrices
resultado = np.dot(matriz1, matriz2)

# Registrar el tiempo después de la multiplicación
tiempo_fin = time.time()

# Calcular el tiempo total
tiempo_total = tiempo_fin - tiempo_inicio

# Imprimir el resultado y el tiempo total
print(f"Tiempo de ejecución: {tiempo_total:.16f} segundos")

@jit(nopython=True)
def multiplicar_matrices(matriz1, matriz2):
    filas_matriz1, columnas_matriz1 = matriz1.shape
    filas_matriz2, columnas_matriz2 = matriz2.shape

    if columnas_matriz1 != filas_matriz2:
        raise ValueError("El número de columnas de la matriz1 debe ser igual al número de filas de la matriz2")

    resultado = np.zeros((filas_matriz1, columnas_matriz2))

    for i in range(filas_matriz1):
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
print(f"Tiempo de ejecución: {tiempo_total:.16f} segundos")

def multiplicar_matrices(matriz1, matriz2):
    return np.array([[sum(int(buscar_resultado_multiplicacion(matriz1[i][k], matriz2[k][j])) for k in range(len(matriz2))) for j in range(len(matriz2[0]))] for i in range(len(matriz1))])

# Registrar el tiempo antes de la multiplicación
tiempo_inicio = time.time()

# Realizar la multiplicación de matrices
resultado = multiplicar_matrices(matriz1, matriz2)

# Registrar el tiempo después de la multiplicación
tiempo_fin = time.time()

# Calcular el tiempo total
tiempo_total = tiempo_fin - tiempo_inicio

# Imprimir el resultado y el tiempo total
print(f"Tiempo de ejecución: {tiempo_total:.16f} segundos")

@jit(int32[:, :](), nopython=True)
def generar_tablas_multiplicar():
    tablas_multiplicar = np.zeros((9, 10), dtype=np.int32)

    for i in range(1, 10):
        for j in range(1, 11):
            tablas_multiplicar[i-1, j-1] = i * j

    return tablas_multiplicar

@jit(int64(int64[:, :], int64, int64), nopython=True)
def buscar_resultado_multiplicacion(tablas_multiplicar, numero1, numero2):
    if 1 <= numero1 <= 9 and 1 <= numero2 <= 10:
        return tablas_multiplicar[numero1-1, numero2-1]
    else:
        return 0  # O cualquier valor predeterminado si los números están fuera del rango

@jit(int32[:, :](int32[:, :], int32[:, :], int32[:, :]), nopython=True)
def multiplicar_matrices(matriz1, matriz2, tablas_multiplicar):
    filas_matriz1, columnas_matriz1 = matriz1.shape
    filas_matriz2, columnas_matriz2 = matriz2.shape

    if columnas_matriz1 != filas_matriz2:
        raise ValueError("El número de columnas de la matriz1 debe ser igual al número de filas de la matriz2")

    resultado = np.zeros((filas_matriz1, columnas_matriz2), dtype=np.int32)

    for i in range(filas_matriz1):
        for j in range(columnas_matriz2):
            for k in range(filas_matriz2):
                resultado[i, j] += tablas_multiplicar[matriz1[i, k]-1, matriz2[k, j]-1]
    return resultado

tablas_multiplicar = generar_tablas_multiplicar()
# Registrar el tiempo antes de la multiplicación
tiempo_inicio = time.time()

# Realizar la multiplicación de matrices
resultado = multiplicar_matrices(matriz1, matriz2,tablas_multiplicar)

# Registrar el tiempo después de la multiplicación
tiempo_fin = time.time()

# Calcular el tiempo total
tiempo_total = tiempo_fin - tiempo_inicio

# Imprimir el resultado y el tiempo total
print(f"Tiempo de ejecución: {tiempo_total:.16f} segundos")

def generar_tablas_multiplicar():
    tablas_multiplicar = [[0] * 10 for _ in range(9)]

    for i in range(1, 10):
        for j in range(1, 11):
            tablas_multiplicar[i-1][j-1] = i * j

    return tablas_multiplicar

def buscar_resultado_multiplicacion(tablas_multiplicar, numero1, numero2):
    if 1 <= numero1 <= 9 and 1 <= numero2 <= 10:
        return tablas_multiplicar[numero1-1][numero2-1]
    else:
        return 0

def multiplicar_matrices(matriz1, matriz2, tablas_multiplicar):
    filas_matriz1 = len(matriz1)
    columnas_matriz1 = len(matriz1[0])
    filas_matriz2 = len(matriz2)
    columnas_matriz2 = len(matriz2[0])

    if columnas_matriz1 != filas_matriz2:
        raise ValueError("El número de columnas de la matriz1 debe ser igual al número de filas de la matriz2")

    resultado = [[0] * columnas_matriz2 for _ in range(filas_matriz1)]

    for i in range(filas_matriz1):
        for j in range(columnas_matriz2):
            for k in range(filas_matriz2):
                resultado[i][j] += tablas_multiplicar[matriz1[i][k]-1][matriz2[k][j]-1]
    return resultado

tablas_multiplicar = generar_tablas_multiplicar()
# Registrar el tiempo antes de la multiplicación
tiempo_inicio = time.time()

# Realizar la multiplicación de matrices
resultado = multiplicar_matrices(matriz1, matriz2,tablas_multiplicar)

# Registrar el tiempo después de la multiplicación
tiempo_fin = time.time()

# Calcular el tiempo total
tiempo_total = tiempo_fin - tiempo_inicio

# Imprimir el resultado y el tiempo total
print(f"Tiempo de ejecución: {tiempo_total:.16f} segundos")

def generar_tablas_multiplicar():
    return [[i * j for j in range(1, 11)] for i in range(1, 10)]

def buscar_resultado_multiplicacion(tablas_multiplicar, numero1, numero2):
    return tablas_multiplicar[numero1-1][numero2-1] if 1 <= numero1 <= 9 and 1 <= numero2 <= 10 else 0

def multiplicar_matrices(matriz1, matriz2, tablas_multiplicar):
    if len(matriz1[0]) != len(matriz2):
        raise ValueError("El número de columnas de la matriz1 debe ser igual al número de filas de la matriz2")

    return [
        [
            sum(tablas_multiplicar[matriz1[i][k]-1][matriz2[k][j]-1] for k in range(len(matriz2)))
            for j in range(len(matriz2[0]))
        ]
        for i in range(len(matriz1))
    ]


tablas_multiplicar = generar_tablas_multiplicar()
# Registrar el tiempo antes de la multiplicación
tiempo_inicio = time.time()

# Realizar la multiplicación de matrices
resultado = multiplicar_matrices(matriz1, matriz2,tablas_multiplicar)

# Registrar el tiempo después de la multiplicación
tiempo_fin = time.time()

# Calcular el tiempo total
tiempo_total = tiempo_fin - tiempo_inicio

# Imprimir el resultado y el tiempo total
print(f"Tiempo de ejecución: {tiempo_total:.16f} segundos")

# import subprocess

# # Compilar el código Fortran usando el compilador Fortran
# subprocess.run(["gfortran", "-o", "fortran", "fortran_code.f90", "-mconsole"])

# # Ejecutar el programa Fortran compilado
# subprocess.run(["./fortran"])