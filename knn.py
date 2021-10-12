# -*- coding: utf_8 -*-
import os
import random
import math
import pandas
import numpy

#----------------VARIABLES---------------------------------------------

nombreArchivoCSV = "airfoil_self_noise.csv"
nombreClase = "Sound Level (TARGET)"
nombrePrediccion = 'Predicción'
claseNumerica = True
porcentajeEntrenamiento = 0
numeroDeInstancias = 0
numeroDeInstanciasEntrenamiento = 0
numeroAtributos = 0
listaAtributos = []
listaAtributosNumericos = []
K = 1
ConjuntoInicial = pandas.DataFrame()
ConjuntoEntrenamiento = pandas.DataFrame()
ConjuntoPrueba = pandas.DataFrame()
MSE = 0
PorcentajeAciertos = 0

#----------------FUNCTIONS---------------------------------------------
def limpiarPantalla():	
	if(os.name == "posix"):
		os.system("clear")
	elif(os.name == "ct" or os.name == "nt" or os.name == "dos"):
		os.system("cls")

def generarConjuntoEntrenamiento(ConjuntoI,numInstanciasE):
	print("generando conjunto entrenamiento")
	
	listaIndicesAleatorios = []
	k = 0
	while len(listaIndicesAleatorios) < numInstanciasE:
		num = random.randint(0,numeroDeInstancias-1)
		
		if num not in listaIndicesAleatorios:
			listaIndicesAleatorios.append(num)
	ConjuntoE = ConjuntoInicial.iloc[listaIndicesAleatorios]

	
	listaIndicesPrueba = []
	for h in range(numeroDeInstancias):
		listaIndicesPrueba.append(h)

	for i in listaIndicesAleatorios:
		if i in listaIndicesPrueba:
			listaIndicesPrueba.remove(i)

	ConjuntoPrueba = ConjuntoInicial.iloc[listaIndicesPrueba]
	ConjuntoP = ConjuntoPrueba
	ConjuntosResultantes = {'Entrenamiento': ConjuntoE, 'Prueba': ConjuntoP}

	return ConjuntosResultantes



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
	# TablaDeDistancias = TablaDeDistancias
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

	if claseNumerica:
		result = KNearestNeighborsTable[nombreClase].mean()
		return result
	else: 
		result = KNearestNeighborsTable[nombreClase].mode().head(1)
		return result.iloc[0]
	

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
def esAcierto(fila):
	
	if fila[nombreClase]==fila[nombrePrediccion]:
		return 100
	else:
		return 0

def porcentajeAciertos(TablaC, nombreClase, nombrePrediccion):
	TablaC['Acierto'] = TablaC.apply(esAcierto, axis=1)

	return TablaC['Acierto'].mean()

def normalizarColumna(Tabla, columna):
	vmin = Tabla[columna].min()
	vmax = Tabla[columna].max()
	print("Mínimo de la columna ", columna, ": ", vmin)
	print("Máximo de la columna ", columna, ": ", vmax)
	Tabla[columna] = Tabla.apply(lambda fila: normalizar(fila[columna],vmin,vmax), axis=1)
	return Tabla


def normalizarColumnasEspecificadas(Tabla, listaColumnas):
	for columna in listaColumnas:		
		Tabla = normalizarColumna(Tabla, columna) 

	return Tabla
def detectarColumnasNumericas():
	return
#----------------BEGIN-------------------------------------------------
limpiarPantalla()
print("--------------------------------------------------------")
print("Actividad 5.6 - Implementación ")
print("Algoritmo K Nearest Neighbor")
print("--------------------------------------------------------")
print("\n")

print("Cargando Conjunto de Datos desde ", nombreArchivoCSV, "...")
try: 
	# Se carga el CSV ignorando la segunda linea del csv que contiene los tipos de dato
	ConjuntoInicial = pandas.read_csv(nombreArchivoCSV, header=0, skiprows=[1])
	df = ConjuntoInicial

	# Se cargan los tipos de datos en un diccionario
	datatipes = pandas.read_csv(nombreArchivoCSV, header=0, nrows=1)
	
	diccTiposAtributos ={}
	for col in datatipes.columns.tolist():
		diccTiposAtributos[col] = datatipes[col].iloc[0]
	if diccTiposAtributos[nombreClase] != 'Numerical':
		claseNumerica = False


except:


	print("Archivo CSV no cargado")
	print("Finalizando programa\n")
	exit()


print("CSV cargado")
input("Presione una tecla para continuar...")
print("\n")
print("Conjunto de Datos: \n")

print(ConjuntoInicial)
print("\n")

numeroDeInstancias = len(ConjuntoInicial.index)

listaAtributos = ConjuntoInicial.columns.tolist()
listaAtributos.remove(nombreClase)
numeroAtributos = len(listaAtributos)

listaAtributosNumericos[:] = listaAtributos[:]
for atributo in listaAtributos:
	if diccTiposAtributos[atributo] != 'Numerical':
		listaAtributosNumericos.remove(atributo)
print(listaAtributosNumericos)
print("\n")

print("Normalizando Atributos Numéricos del Conjunto Iicial")
ConjuntoInicial = normalizarColumnasEspecificadas(ConjuntoInicial, listaAtributosNumericos)
print("Conjunto de Datos [Normalizado]:")
print(ConjuntoInicial)
print("\n")
print("Se han encontrado ", numeroDeInstancias, " instancias en el Conjunto Inicial")
print("Los tipos de dato para cada atributo son:\n")
print(datatipes)
print("\n")

print("Introduzca el porcentaje de instancias para entrenamiento (sin el signo %)")
try:
	porcentajeEntrenamiento = float(input("Porcentaje de instancias de entrenamiento: "))
	numeroDeInstanciasEntrenamiento = math.ceil(numeroDeInstancias * porcentajeEntrenamiento/100)
except ValueError:
	print("Valor no valido")
	print("Usar 70% como valor por defecto?")
	res = input("S/N: ")
	if res.upper() != 'S':
		print("Finalizando el programa.... :(")
		exit()
	else:
		porcentajeEntrenamiento = 70
		numeroDeInstanciasEntrenamiento = math.ceil(numeroDeInstancias * porcentajeEntrenamiento/100)
print("Introduzca el valor de vecinos cercanos a la tupla ")
try:
	K = int(input("K: "))
	
except ValueError:
	print("Valor no valido")
	print("Finalizando el programa.... :(")
	exit()	

print("El porcentaje de instancias de entrenamiento es:\t", porcentajeEntrenamiento, "%")
print("El porcentaje de instancias de prueba es:       \t", 100-porcentajeEntrenamiento, "%")
print("En consecuencia se determina que")
print("Se usarán ", numeroDeInstanciasEntrenamiento," instancias para entrenamiento y")
print("se usarán ", numeroDeInstancias - numeroDeInstanciasEntrenamiento," instancias para prueba.")
print("Se usarán K: ", K, " vecinos cercanos")
if(claseNumerica):
	tipoDeProblema = "Regresión"
else:
	tipoDeProblema = "Clasificación"
print("Se resolverá el problema por ", tipoDeProblema)

print("\n")
input("Presone ENTER para continuar...")

print("Direccion de memoria antes")
print(id(ConjuntoEntrenamiento))
print("\n")



ConjuntosResultantes = generarConjuntoEntrenamiento(ConjuntoInicial, numeroDeInstanciasEntrenamiento)
ConjuntoEntrenamiento = ConjuntosResultantes['Entrenamiento']
ConjuntoPrueba = ConjuntosResultantes['Prueba']

print("Direccion de memoria despues")
print(id(ConjuntoEntrenamiento))
print("\n")

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
tuplaI = [5.9,3.0,5.1,1.8]
tuplaA = [800,0,0.3048,71.3,0.00266337] #debe resultar: 126.201
res = predecirKNN(tuplaA, ConjuntoEntrenamiento, nombreClase, K)
print(res)
print("\n")
# print(ConjuntoEntrenamiento)
print("\n")


if False:
	
	print("Generar tabla de predicciones")
	TablaDePredicciones = generarPrediccionesKNNEnConjunto(ConjuntoPrueba, ConjuntoEntrenamiento, nombreClase, K)
	print(TablaDePredicciones)
	# print(ConjuntoPrueba)

	print("\n")
	print("Evaluando capacidad predicitiva...")
	print("\n")


	TablaComparacion = generarTablaComparacion(TablaDePredicciones, nombreClase, nombrePrediccion)
	if claseNumerica:
		MSE = errorCuadraticoMedio(TablaComparacion, nombreClase, nombrePrediccion)
	else:
		PorcentajeAciertos = porcentajeAciertos(TablaComparacion, nombreClase, nombrePrediccion)

	print("La clase del conjunto de datos es numérica")
	print(TablaComparacion)
	print("\n")
	print("El Error Cuadrático Medio (MSE) es: ", MSE)
	print("El porcentaje de Aciertos es: ", PorcentajeAciertos, " %")

	print("\n")
	print("\n")


	# Pendiente:
	
	# -Corregir copia de dataframes por copia y valor
	
	# -Atributos categoricos






