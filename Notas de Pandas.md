# Notas sobre pandas en Python

## Instalación:

	pip install pandas

## Uso:
### Incluirlo en el archivo

	import pandas

### Cargar del CSV
	pandas.read_csv("nombre_archivo")
	pandas.read_csv("nombre_archivo", index_col="columnaID") 
Esto devuelve un dataframe 

### Ver Data Frame
Si el dataframe es df, para imprimir todo en pantalla:
	
	print(df)

Primeras 5 filas:
	
	df.head()

Primeras n filas:

	df.head(n)

Últimas 5 filas:
	
	df.tail()

Últimas n filas
	
	df.tail(n)

En todos los casos es necesario usar el print()

Muestra datos estadisticos del dataset:

	df.describe()

	df.dropna()
	df.fillna(valor)
	df.fillna({"clumna1": 0, "columna2": 4})

### Filtrar:
Por columnas
	df["columnaX"]
	df[["columnaX", "columnaY"]]

Por filas

	df.iloc[0]
	df.iloc[2:4]
	df.iloc[[0,5,9]]

	df.loc[23442]
	*df.loc[]
	df.loc[[232333,233333,323334]]

Por filas y columnas
	
	df.loc[[3434, 3433], ["ColumnaX", "ColumnaY"]]

Por condiciones

	df[(df["ColumnaX"] > 400 & df["ColumnaY"] > 20)]
	df[ df["ColumnaX"].str.contains("cadena")]

	listaAtributos = ConjuntoInicial.columns.tolist()
