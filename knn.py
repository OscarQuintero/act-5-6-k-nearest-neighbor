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

def normalizar(valor, vMin, vMax):
	return (float(valor) - float(vMin))/(float(vMax) - float(vMin))

def distanciaEuclideana(tupla1, tupla2): #Con listas
	n = len(tupla1)
	if n != len(tupla2):
		return "Las dimensiones de las listas no coinciden"
	else:
		sumaCuadrados = 0
		for x in range(n):
			sumaCuadrados += (tupla2[x] - tupla1[x])**2
			
		return sumaCuadrados**0.5

def distanciaManhattan(tupla1, tupla2):
	n = len(tupla1)
	if n != len(tupla2):
		return "Las dimensiones de las listas no coinciden"
	else:
		sumaValoresAbsolutos = 0
		for x in range(n):
			sumaValoresAbsolutos += abs(tupla1[x] - tupla2[x])

		return sumaValoresAbsolutos

def distanciaHamming(tupla1, tupla2):
	n = len(tupla1)
	if n != len(tupla2):
		return "Las dimensiones de las listas no coinciden"
	else:
		sumaHamming = 0
		for x in range(n):
			if tupla1[x] == tupla2[x]:
				sumaHamming += 0
			else:
				sumaHamming += 1

		return sumaHamming
	 


#----------------BEGIN-------------------------------------------------
print("--------------------------------------------------------")
print("Actividad 5.6 - Implementación ")
print("Algoritmo K Nearest Neighbor")
print("--------------------------------------------------------")
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
try:
	porcentajeEntrenamiento = int(input("Porcentaje de instancias de entrenamiento: "))
	numeroDeInstanciasEntrenamiento = math.ceil(numeroDeInstancias * porcentajeEntrenamiento/100)
except ValueError:
	print("Valor no valido")
	print("Usar 70% como valor por defecto?")
	res = input("S/N: ")
	if res != 'S':
		print("Finalizando el programa.... :(")
		exit()
	else:
		porcentajeEntrenamiento = 70
		numeroDeInstanciasEntrenamiento = math.ceil(numeroDeInstancias * porcentajeEntrenamiento/100)

print("El porcentaje de instancias de entrenamiento es:\t", porcentajeEntrenamiento, "%")
print("El porcentaje de instancias de prueba es:       \t", 100-porcentajeEntrenamiento, "%")
print("En consecuencia se determina que")
print("Se usarán ", numeroDeInstanciasEntrenamiento," instancias para entrenamiento y")
print("se usarán ", numeroDeInstancias - numeroDeInstanciasEntrenamiento," instancias para prueba.")
print("\n")

listaIndicesAleatorios = []
k = 0
while len(listaIndicesAleatorios) < numeroDeInstanciasEntrenamiento:
	num = random.randint(0,numeroDeInstancias-1)
	
	if num not in listaIndicesAleatorios:
		listaIndicesAleatorios.append(num)
ConjuntoEntrenamiento = ConjuntoInicial.iloc[listaIndicesAleatorios]

listaIndicesPrueba = []
for h in range(numeroDeInstancias):
	listaIndicesPrueba.append(h)

for i in listaIndicesAleatorios:
	if i in listaIndicesPrueba:
		listaIndicesPrueba.remove(i)

ConjuntoPrueba = ConjuntoInicial.iloc[listaIndicesPrueba]

print("Conjunto de Datos de Entrenamiento:")
print(ConjuntoEntrenamiento)
print("\n")
print("Conjunto de Datos de Prueba:")
print(ConjuntoPrueba)
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

print("Prueba con distancias")
print(distanciaHamming([0,3,3,5],[8,90,3,1]))

