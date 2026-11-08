# Pipeline para ejecutar el análisis de preprocesamiento de señales electrofisiológicas y optogenética:

Herramiento computacional para el procesamiento de señales multiunitarias alineadas a un estímulo optogenético
para la identificación de espigas, reducción de dimensionalidad y clusterización de espigas.

El sistema procesa sesiones adquiridas con Open Ephys (formato binario .oebin), sincroniza señales analógicas continuas 
con eventos digitales TTL y genera representaciones peri-estímulo (matrices raster y PSTH) libres de artefactos por luz.

1. Estructura del proyecto:

El usuario necesita seguir los siguiente puntos para lograr ejecutar este código:
-README.md                    # Documentación e instrucciones de uso del programa
-requirements.txt             # Dependencias del entorno
-preprocess.py                # Script principal de ejecución
-functions_for_preprocess.py  # Funciones secundarias para preprocess

2. Requisitos del Sistema
-Sistema Operativo: Compatible con Windows 10/11, macOS y Linux.
-Lenguaje: Python 3.9 (Recomendado 3.10 o 3.11), de esta forma funciona mejor las librerias como numpy
-Memoria RAM: Mínimo 8 GB (Recomendado 16 GB para sesiones multicanal continuas de 30 a 60 minutos).
-Almacenamiento: Espacio suficiente para  los archivosgenerados a partir de la información cruda.

3. Descargar la carpeta con el programa y sus adicionales
-Abre tu terminal o tu interfaz de python preferida.
**En caso de hacerlo desde la terminal ingresa el siguiente comando en donde alojaste la carpeta:
> cd ruta/al/proyecto

4. Instalar dependencias:
**Ejecuta el siguiente comando con las paqueterias requeridas para el programa: 
> pip install open-ephys-python-tools numpy pandas matplotlib scipy scikit-learn

5. Ejecución

-Iniciar el pipeline:
> python preprocess.py

- Seleccionar la sesión de registro:
Se abrirá una ventana de selección de carpetas (select_folder). Selecciona el directorio del Record Node correspondiente a la grabación de Open Ephys (por ejemplo, .../Record Node 114)

-Procesamiento automatizado:
En este punto no te debes preocupar de nada, el programa ejecutará difernetes funciones como el pasabandas, umbralizar y comenzará 
el análisis de PCA y clustrización donde podrás ir visualizando por canales el número de cluster registrados en cada uno de los canales

-Curación Manual e Interactiva de Artefactos (Función artefact):
Para cada clúster detectado, el sistema desplegará una ventana gráfica de dos paneles:
a. Panel izquierdo (Template): Muestra todas las formas de onda recortadas de las espigas y la plantilla promedio.
b. Panel derecho (PSTH): Muestra el histograma de tiempo peri-estímulo centrado en la entrega del pulso LED (marcado con sombra gris de 0 a 250 ms).

Mientras la figura permanece en pantalla, la consola solicitará tu decisión. Introduce una de las siguientes opciones por teclado y presiona Enter:
n : Clasificar como NEURONA (conserva la unidad biológica).
m : Clasificar como ARTEFACTO (elimina el clúster por respuesta fotoeléctrica).
o : OMITIR (ignora la unidad ambigua).

Al ingresar la opción, la ventana se cerrará automáticamente y continuará con el siguiente clúster.

**Ojo si tienes poca RAM el programa puede tardar mas en generar las figuras interactivas, ten paciencia.

6. Finalización y Serialización:
Al terminar la curación de todas las unidades, el programa construirá las matrices binarias de eventos peri-estímulo y guardará los resultados serializados en la misma carpeta del archivo seleccionado.

7. Salidas Generadas

El sistema genera y guarda automáticamente dos archivos en formato .pkl en el directorio inmediatamente superior al Record Node:

1. metadata.pkl
Diccionario con los parámetros globales del registro:
sample_rate: Frecuencia de muestreo (Hz).
duration_minutes: Duración total de la sesión en minutos.
directory_path: Ruta del directorio procesado.
n_channels: Número total de canales analizados.
date_processed: Fecha y hora exacta de la ejecución.

2. register_output.pkl
Diccionario con las estructuras y variables analíticas procesadas:
data_filtered: Matriz continua 2D (muestras x canales) con la señal filtrada pasobanda.
all_spikes: Arreglo con las formas de onda de espigas extraídas.
spikes_per_channel: Índices de muestras donde se detectaron espigas agrupados por canal.
pulses_windows: DataFrame con los tiempos de inicio y fin de cada pulso de luz en segundos.
num_clusters: Estructura de espigas organizadas por canal y por clúster.
total_clusters_id: DataFrame estructurado (ClusterID, Events, Canal, ClusterLocal) con las unidades confirmadas como neuronas.
events_clusters: Lista de matrices binarias peri-estímulo de dimensiones (n_ensayos, tamaño_ventana) generadas por clusters_per_channel.
spike_clusters: Trazas y formas de onda validadas libres de artefactos.
