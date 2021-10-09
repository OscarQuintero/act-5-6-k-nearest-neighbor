# -*- coding: utf_8 -*-
import os
import random
import math
import pandas

#----------------VARIABLES---------------------------------------------

nombreArchivoCSV = "play_db.csv"
nombreClase = "Play"
porcentajeEntrenamiento = 0
numeroDeInstancias = 0
numeroDeInstanciasEntrenamiento = 0

#----------------FUNCTIONS---------------------------------------------

#----------------BEGIN-------------------------------------------------
print("Actividad 5.6 - Implementación ")
print("Algoritmo K Nearest Neighbor")
print("\n")

print("Cargando Conjunto de Datos desde ", nombreArchivoCSV, "...")
ConjuntoInicial = pandas.read_csv(nombreArchivoCSV)
df = ConjuntoInicial

print("CSV cargado")
porcentajeEntrenamiento = int(input("Introduzca el porcentaje de instancias de entrenamiento: "))
print("Conjunto de Datos: \n")

print(ConjuntoInicial)
print("\n")

listaIndicesAleatorios = []
numeroDeInstancias = len(ConjuntoInicial.index)
numeroDeInstanciasEntrenamiento = math.ceil(numeroDeInstancias * porcentajeEntrenamiento/100)

print("Se han encontrado ", numeroDeInstancias, " instancias en el Conjunto Inicial")
print("El porcentaje de instancias de entrenamiento es:\t", porcentajeEntrenamiento, "%")
print("El porcentaje de instancias de prueba es:       \t", 100-porcentajeEntrenamiento, "%")
print("En consecuencia se determina que")
print("Se usarán ", numeroDeInstanciasEntrenamiento," instancias para entrenamiento y")
print("se usarán ", numeroDeInstancias - numeroDeInstanciasEntrenamiento," instancias para prueba.")
print("\n")

k = 0
while len(listaIndicesAleatorios) < numeroDeInstanciasEntrenamiento:
	num = random.randint(0,numeroDeInstancias-1)
	
	if num not in listaIndicesAleatorios:
		listaIndicesAleatorios.append(num)
ConjuntoEntrenamiento = ConjuntoInicial.iloc[listaIndicesAleatorios]

print("Conjunto de Datos de Entrenamiento:")
print(ConjuntoEntrenamiento)
print("\n")
print("Conjunto de Datos de Prueba:")
# print(ConjuntoPrueba)
print("\n")







print("Algoritmo KNN ....")
print("Evaluar porcentaje de aciertos o error cuadrático medio")
print("Evaluar capacidad predicitiva")
print("\n")



