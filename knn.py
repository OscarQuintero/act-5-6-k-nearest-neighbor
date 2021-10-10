# -*- coding: utf_8 -*-
import os
import random
import math
import pandas
import numpy

#----------------VARIABLES---------------------------------------------

nombreArchivoCSV = "play_db.csv"
nombreClase = "Play"
porcentajeEntrenamiento = 0
numeroDeInstancias = 0
numeroDeInstanciasEntrenamiento = 0
K = 1
ConjuntoInicial = pandas.DataFrame()
ConjuntoEntrenamiento = pandas.DataFrame()
# ConjuntoPrueba = pandas.DataFrame()

#----------------FUNCTIONS---------------------------------------------

#----------------BEGIN-------------------------------------------------
print("Actividad 5.6 - Implementación ")
print("Algoritmo K Nearest Neighbor")
print("\n")

print("Cargando Conjunto de Datos desde ", nombreArchivoCSV, "...")
try: 
	ConjuntoInicial = pandas.read_csv(nombreArchivoCSV)
	df = ConjuntoInicial
except:
	print("Archivo CSV no cargado")
	print("Finalizando programa\n")
	exit()

print("CSV cargado")

print("Conjunto de Datos: \n")

print(ConjuntoInicial)
print("\n")


numeroDeInstancias = len(ConjuntoInicial.index)


print("Se han encontrado ", numeroDeInstancias, " instancias en el Conjunto Inicial")
print("Introduzca el porcentaje de instancias para entrenamiento (sin el signo %)")
porcentajeEntrenamiento = int(input("Porcentaje de instancias de entrenamiento: "))
numeroDeInstanciasEntrenamiento = math.ceil(numeroDeInstancias * porcentajeEntrenamiento/100)
print("El porcentaje de instancias de entrenamiento es:\t", porcentajeEntrenamiento, "%")
print("El porcentaje de instancias de prueba es:       \t", 100-porcentajeEntrenamiento, "%")
print("En consecuencia se determina que")
print("Se usarán ", numeroDeInstanciasEntrenamiento," instancias para entrenamiento y")
print("se usarán ", numeroDeInstancias - numeroDeInstanciasEntrenamiento," instancias para prueba.")
print("\n")

print("Direccion de memoria antes")
print(id(ConjuntoEntrenamiento))
print("\n")

def generarCoonjuntoEntrenamiento(ConjuntoI,numInstanciasE, ConjuntoE):
	print("generando conjunto entrenamiento")
	ConjuntoEntrenamiento = ConjuntoE
	print(id(ConjuntoE))
	print(id(ConjuntoEntrenamiento))
	print("\n")
	listaIndicesAleatorios = []
	k = 0
	while len(listaIndicesAleatorios) < numInstanciasE:
		num = random.randint(0,numeroDeInstancias-1)
		
		if num not in listaIndicesAleatorios:
			listaIndicesAleatorios.append(num)
	ConjuntoE = ConjuntoInicial.iloc[listaIndicesAleatorios]

	print(id(ConjuntoE))
	print(id(ConjuntoEntrenamiento))
	print("\n")
	
	print(ConjuntoE)


	listaIndicesPrueba = []
	for h in range(numeroDeInstancias):
		listaIndicesPrueba.append(h)

	for i in listaIndicesAleatorios:
		if i in listaIndicesPrueba:
			listaIndicesPrueba.remove(i)

	ConjuntoPrueba = ConjuntoInicial.iloc[listaIndicesPrueba]

	pass



generarCoonjuntoEntrenamiento(ConjuntoInicial, numeroDeInstanciasEntrenamiento, ConjuntoEntrenamiento.copy(False))

print("Direccion de memoria despues")
print(id(ConjuntoEntrenamiento))
print("\n")

print("Conjunto de Datos de Entrenamiento:")
print(ConjuntoEntrenamiento)
print("\n")
print("Conjunto de Datos de Prueba:")
# print(ConjuntoPrueba)
print("\n")


print("Algoritmo K Nearest Neighbor ....")
print("Hacer función del algoritmo K-NN por tupla")
print("Hacer predicciones para el conjunto de prueba")
print("Mostrar predicciones y valores reales")

print("Por definir mas funciones....")

print("\n")
print("Evaluar porcentaje de aciertos o error cuadrático medio")
print("Evaluar capacidad predicitiva")
print("\n")



