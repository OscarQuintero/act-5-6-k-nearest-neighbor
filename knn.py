# -*- coding: utf_8 -*-
import os
import random
import math
import pandas

#----------------VARIABLES---------------------------------------------

nombreArchivoCSV = "airfoil_self_noise.csv"
nombreClase = "Sound Level (TARGET)"
nombrePrediccion = 'Predicción'
porcentajeEntrenamiento = 0
numeroDeInstancias = 0
numeroDeInstanciasEntrenamiento = 0
numeroAtributos = 0
listaAtributos = []
K = 1
ConjuntoInicial = pandas.DataFrame()
ConjuntoEntrenamiento = pandas.DataFrame()
ConjuntoPrueba = pandas.DataFrame()

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
	 
def predecirKNN(tupla, ConjuntoE, nombreClase, k=1): #predice el valor de la clase
	print("-----")
	print("Tabla de distancias para ", tupla)
	TablaDeDistancias = ConjuntoE[listaAtributos]
	TablaDeDistancias['Distancia'] = TablaDeDistancias.apply(lambda fila: distanciaEuclideana(tupla,fila[listaAtributos].tolist()), axis=1)
	TablaDeDistancias = TablaDeDistancias.sort_values('Distancia')
	print(TablaDeDistancias)


	minDistancia = TablaDeDistancias['Distancia'].min()
	# print("Minimo: ", minDistancia)	

	
	KNearestNeighborsTable = TablaDeDistancias.head(k)
	print(KNearestNeighborsTable)
	ListaIndicesKNN = KNearestNeighborsTable.index.tolist()
	
	KNearestNeighborsTable = ConjuntoInicial.iloc[ListaIndicesKNN]
	print(KNearestNeighborsTable) #No se pudo desde ConjuntoE por desbordamiento
	
	print("-----")
	return KNearestNeighborsTable[nombreClase].mean()
	

def generarPrediccionesKNNEnConjunto(ConjuntoP, ConjuntoE, nombreClase, k=1):
	TablaConPredicciones = ConjuntoP
	TablaConPredicciones[nombrePrediccion] = TablaConPredicciones.apply(insertarPrediccionParaLaInstancia, axis=1)
	return TablaConPredicciones

def insertarPrediccionParaLaInstancia(fila):
	instancia = fila[listaAtributos].tolist()
	print(instancia)
	prediccion = predecirKNN(instancia, ConjuntoEntrenamiento, nombreClase, K)
	print(prediccion)
	print("\n")
	return prediccion

def errorCuadrado(fila):
	return (fila[nombrePrediccion] - fila[nombreClase])**2


def generarTablaComparacion(TablaP, nombreClase, nombrePrediccion):
	 return TablaP[[nombreClase, nombrePrediccion]]

def errorCuadraticoMedio(TablaC, nombreClase, nombrePrediccion):
	
	TablaC['Error Cuadrado'] = TablaC.apply(errorCuadrado, axis=1)
	
	return TablaC['Error Cuadrado'].mean()

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

listaAtributos = ConjuntoInicial.columns.tolist()
listaAtributos.remove(nombreClase)
numeroAtributos = len(listaAtributos)


print("Se han encontrado ", numeroDeInstancias, " instancias en el Conjunto Inicial")
print("Introduzca el porcentaje de instancias para entrenamiento (sin el signo %)")
try:
	porcentajeEntrenamiento = float(input("Porcentaje de instancias de entrenamiento: "))
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

print("Prueba con predicciones")
tupla = [800,0,0.3048,71.3,0.00266337] #debe resultar: 126.201
res = predecirKNN(tupla, ConjuntoEntrenamiento, nombreClase, 3)
print(res)
print("\n")
print(ConjuntoEntrenamiento)
print("\n")

print("Generar tabla de predicciones")
TablaDePredicciones = generarPrediccionesKNNEnConjunto(ConjuntoPrueba, ConjuntoEntrenamiento, nombreClase, K)
print(TablaDePredicciones)
print(ConjuntoPrueba)

print("Por definir mas funciones....")



print("\n")
print("Evaluar porcentaje de aciertos o error cuadrático medio")
print("Evaluar capacidad predicitiva")
print("\n")


TablaComparacion = generarTablaComparacion(TablaDePredicciones, nombreClase, nombrePrediccion)
MSE = errorCuadraticoMedio(TablaComparacion, nombreClase, nombrePrediccion)

print("La clase del conjunto de datos es numérica")
print(TablaComparacion)
print("\n")
print("El Error Cadrático Medio (MSE) es: ", MSE)

print("\n")
print("\n")





