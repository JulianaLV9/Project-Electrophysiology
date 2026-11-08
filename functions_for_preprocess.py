
"""""
Created on Feb Tue 11 12:54:12 2025

@author: julianalozavaqueiro
"""

## Libraries

#Signal processing
from scipy.signal import butter, filtfilt, find_peaks

#Matplotlib
import matplotlib.pyplot as plt

# Python libraries
import numpy as np
import pandas as pd

#PCA analysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Clusterization analysis
from sklearn.cluster import KMeans
from kneed import KneeLocator

#Import files 
import subprocess

ce = (20/255, 48/255, 81/255)
cr = (118/255, 43/255, 39/255)

def select_folder():
    msg = "Select the folder where are your files for the preprocess (Choose the folder with the Record Node)"
    script = f'choose folder with prompt "{msg}"'
    
    try:
        # Ejecutamos el AppleScript
        cmd = ['osascript', '-e', f'POSIX path of ({script})']
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            directory = result.stdout.strip()
            print(f"Folder selected: {directory}")
            return directory
        else:
            print("Selection canceled")
            return None # Retornamos None para detener si cancelas
            
    except Exception as e:
        print(f"Error")
        return None


def band_pass_filter(data, sample_rate, low_cutoff=300, high_cutoff=6000, filter_order=3):
    
    """ 
    Funcion que permite agregar un filtro pasabanda
    
    Keyword argumets:
        data         = recibe el archivo con su informacion de los canales
        low_cutoff   = limite inferior del filtro pasa banda. Default es 300
        high_cutoff  = limite superior del filtro pasa banda. Default es 6000
        filter order = rigurosidad del filtro. Default es 3
        
    """
    # Normalizar las frecuencias de corte con respecto a la frecuencia de muestreo
    nyquist = 0.5 * sample_rate
    low     = low_cutoff / nyquist
    high    = high_cutoff / nyquist
    
    # Crear el filtro pasa-banda
    b, a = butter(filter_order, [low, high], btype='band')
    
    # Aplicar el filtro a los datos
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data

def select_train(pulses_windows):
    
    """ 
    Funcion que retorna las ventanas de la duracion total de  los trenes de estimulacion
    
    Keyword arguments:
        pulses_windows = DataFrame que contiene el Onset_Pulse y Off_Set de todos los pulsos
        
    """
    trains   = []
    n_pulses = pulses_windows.shape[0]
    
    for start in range(0, n_pulses, 5): 
        
        end = start + 5
        
        if end <= n_pulses:  # aseguramos que hay 5 pulsos completos
            onset_train  = pulses_windows.iloc[start]["Onset_Pulse"] - 0.250
            offset_train = pulses_windows.iloc[end-1]["Offset_Pulse"] + 0.250
            trains.append([onset_train, offset_train])
    
    train_windows = pd.DataFrame(trains, columns=["Onset_Train", "Offset_Train"])
    
    return train_windows

def channel_vision(data, timestamps, sample_rate, pulses_windows=None):
    """ 
    Funcion que permite la visualizacion de los canales de registro
    
    Keyword arguments:
        data           = recibe el archivo 
        timestamps     = estampas temporales 
        sample_rate    = frecuencia de muestreo. 
        pulses_windows = ventanas de los pulsos
        
    """
    
    num_canales_por_figura = 8  # Canales por figura
    total_canales          = 32  # Total de canales que esperas
    colors = plt.cm.get_cmap('tab20', num_canales_por_figura)  # Colores para 8 canales

    # Filtrar hasta qué número de pulso alcanzaron para graficarlo
    pulses = pulses_windows[pulses_windows['Onset_Pulse'] <= (len(data) / sample_rate)]

    # Crear 4 figuras, cada una con 8 canales
    for figure_num in range(4):
        fig, axs = plt.subplots(num_canales_por_figura, 1, figsize=(12, 12), sharex=True)

        # Calcular el rango de canales a mostrar
        inicio_canales = figure_num * num_canales_por_figura
        fin_canales    = min(inicio_canales + num_canales_por_figura, total_canales)

        # Verificar si hay canales suficientes para la figura actual
        if inicio_canales >= total_canales:
            break  # No hay más canales disponibles

        # Iterar sobre los canales del grupo correspondiente
        for i in range(inicio_canales, fin_canales):
            
            # Graficar los datos del canal
            axs[i - inicio_canales].plot(
                timestamps[:len(data)],
                data[:, i],
                label=f'Canal {i + 1}',
                color=colors(i - inicio_canales),
                linewidth=0.8
            )
            axs[i - inicio_canales].set_ylim(-500, 500)  # Establecer límites de -500 a 500 en Y

            # Agregar franjas sombreadas para los pulsos en cada canal
            for _, row in pulses.iterrows():
                onset_time  = row['Onset_Pulse']
                offset_time = row['Offset_Pulse']
                axs[i - inicio_canales].axvspan(
                    onset_time, offset_time,
                    color = (0.0, 0.0, 0.5),
                    alpha = 0.1  # Azul marino con transparencia
                )

            axs[i - inicio_canales].set_ylabel(f'Canal {i + 1}')
            axs[i - inicio_canales].legend(loc='upper right')

        # Etiqueta en el último subplot
        axs[-1].set_xlabel('Tiempo (s)')

        plt.suptitle(f'Señales de los canales {inicio_canales + 1}-{fin_canales}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

def threshold(data, multiple=5):
    
    """ 
    Funcion que establece el umbral para considerar un spike
    
    Keyword argumets:
        data     = recibe el archivo con su informacion de los canales
        multiple = multiplo que definira que tan estricto sera el umbral para detectar el spike. Default: 4 
        
    """
    
    thresholds = []
    
    for i in range(data.shape[1]):  
        median_val = np.median(data[:, i])  
        mad_val    = np.median(np.abs(data[:, i] - median_val))  # MAD para cada canal Median Absolute Deviation
        threshold  = median_val + multiple * mad_val  # Umbral positivo
        thresholds.append(threshold) 
        
    return np.array(thresholds)

def peaks(data, threshold_ch, sample_rate):
    
    """ 
    Funcion que localiza los indices de los spike dependiendo del umbral
    
    Keyword argumets:
        data        = recibe el archivo con su informacion de los canales
        threshold   = umbral definido 
        sample_rate = frecuencia de muestreo
        
    """
    
    num_samples, num_channels = data.shape
    binary_matrix             = np.zeros((num_samples, num_channels), dtype=int)  # Matriz binaria
    time_matrix               = np.zeros((num_samples, num_channels))  # Matriz de tiempos de picos
    spikes                    = np.zeros((num_samples, num_channels), dtype=int)  # Matriz binaria de picos
    spikes_channel            = []
    
    for i in range(num_channels): 
        
        # Encontrar los picos en el canal i
        peak_idx, _ = find_peaks(-data[:, i], height = threshold_ch[i], distance=60) #Se debe invertir la señal debido a que find_peaks solo detecta positivos
        spikes_channel.append(peak_idx)  # Guardar los índices de picos para cada canal
        
        # Marcar los picos en la matriz binaria y en spikes
        binary_matrix[peak_idx, i] = 1  
        spikes[peak_idx, i] = 1  
        
        # Guardar los tiempos en la matriz de tiempos (convertidos a segundos)
        time_matrix[peak_idx, i] = peak_idx / sample_rate  

    return np.array(spikes_channel, dtype=object), binary_matrix, time_matrix, spikes

def train_array(train_windows,spikes_data, sample_rate):
    """
    Funcion que va  a devolver el arreglo con el índice por canal de los spikes que ocurrieron en la ventana de los trenes
    
    Keyword arguments:
        train_window = Onset_train y offset_train de los trenes
        spikes_data  = matriz binaria de todos los eventos considerados spikes en todo el registro
        sample_rate  = tasa de muestreo
    
    """
    n_channels = spikes_data.shape[1]
    trains     = train_windows.to_numpy()
    n_train    = trains.shape[0]

    # duración fija de los trenes en muestras
    dur_window = train_windows.Offset_Train - train_windows.Onset_Train
    mean_dur   = np.mean(dur_window)
    length     = int(mean_dur * sample_rate)  

    # array 3D vacío
    train_array_3d = np.zeros((n_train, length, n_channels), dtype=int)
    
    for l in range(n_train):
        onset_sec, offset_sec = trains[l]
        onset_idx             = int(onset_sec * sample_rate)
        offset_idx            = int(offset_sec * sample_rate)
        
        # recorte de la ventana
        segment = spikes_data[onset_idx:offset_idx, :]

        # asegurar que tenga máximo 'length' muestras
        segment = segment[:length, :]

        # matriz de índices absolutos (posición en el registro)
        indices    = np.arange(onset_idx, onset_idx + segment.shape[0])[:, None]
        idx_matrix = segment * indices  # 0 donde no hay spike

        # Rellenar con ceros si el segmento fue más corto que 'length'
        padded                          = np.zeros((length, n_channels), dtype=int)
        padded[:idx_matrix.shape[0], :] = idx_matrix

        # guardar en el array 3D
        train_array_3d[l, :, :] = padded

    return train_array_3d
 
def spikes_array(data, train_array_3d, sample_rate):   
    
    """ 
    Genera un arreglo 4D de espigas por canal y por ensayo usando la información cruda de data.
    Cada espiga se alinea en la muestra 15 de una ventana de 60 muestras.

    Keyword arguments:
    - data: ndarray 2D (tiempo x canales) con datos crudos
    - train_array_3d: ndarray 3D (trenes x tiempo x canales) con 0 o tiempo real de espiga
    - sample_rate: Hz
    
    """
    
    #Ventana en al que se va a depositar el spike
    spike_start    = 0
    post_position  = 45
    pre_position   = 15
    
    #Cuantificacion de los loops
    n_trains   = train_array_3d.shape[0]   
    n_channels = train_array_3d.shape[2]   # 32 canales 

    # Sacar los índices de las spikes que están dentro de cada tren por canal 
    list_channel  = [] #se crea la lista por canal

    for ch in range(n_channels):   # ahora recorre 32 canales
       list_train = [] #Se crea la lista por tren
       
       for t in range(n_trains):
           column     = train_array_3d[t,:,ch] #selecciono el tren
           idx        = np.nonzero(column)[0] #busco los indices donde no es cero
           spikes     = column[idx]
           list_train.extend(spikes)
           
       list_channel.append(list_train)
       
    list_spike_channel = list_channel
    
    #Armar los array por spikes de canal entrando a las listas por canal
    all_channels_spikes = []
    
    for ch_ls in range(n_channels):
        spikes_ch    = list_spike_channel[ch_ls]
        spikes_array = []
        
        for spike in spikes_ch:    
           spike_start    = spike - pre_position
           spike_end      = spike + post_position
           complete_spike = data[spike_start:spike_end,ch_ls]
           
           #Revisamos que no salga de data
           if spike_start >= 0 and spike_end <= data.shape[0]:
            complete_spike = data[spike_start:spike_end, ch_ls]
            spikes_array.append(complete_spike)
            
        all_channels_spikes.append(np.array(spikes_array))
        all_spikes_channel = all_channels_spikes
    
    return list_spike_channel, all_spikes_channel

# clustering

"""
TAL vez esta función se vaya, es innecesaria

"""
def plot_spikes(all_spikes_array, sample_rate): 
    
    # Vector de tiempo para la ventana de cada spike
    dt = 1/sample_rate        # segundos por muestra
    dt_ms = dt * 1000         # convertir a ms
    window_size = 60
    time_vector = (np.arange(0, window_size) * dt_ms)  # desde 0 hasta 60 muestras

    for ch in range(len(all_spikes_array)):
        spike_array = np.array(all_spikes_array[ch])
        num_spikes = spike_array.shape[0]

        plt.figure(figsize=(12, 8))
        
        # Graficar todas las trayectorias
        for i in range(num_spikes):
            plt.plot(time_vector, spike_array[i], alpha=0.02,color='gray')  # transparencia

        # Línea vertical en t=0
        plt.axvline(time_vector[15], color='red', linestyle='--', alpha=0.7)  # si pre=15
        
        plt.xlabel('Tiempo (ms)')
        plt.ylabel('Voltaje (uV)')
        plt.title(f'Todas las trayectorias de {num_spikes} spikes - Canal {ch}')
        plt.ylim(spike_array.min() - 20, spike_array.max() + 20)
        plt.show()
        
def pca_analysis(all_spikes_array):
    
    """
    Realiza un análisis de PCA y clustering con KMeans sobre trenes de espigas por canal.

    Keyword arguments:
    all_spikes_array : list of ndarray
        Lista donde cada elemento corresponde a un canal. 
        Cada canal contiene un arreglo 2D de forma (num_spikes, num_features).

    Returns:
    all_labels : list
        Lista con los labels de clustering por canal (o None si no se pudo).
    all_pca : list
        Lista con las proyecciones PCA (X_pca) por canal.
    """

    all_labels = []  # guardamos los labels por canal
    all_pca    = []     # guardamos los PCA por canal

    for ch, spikes in enumerate(all_spikes_array):
        spike_array = np.array(spikes)  # (num_spikes, num_features)

        # Si el canal está vacío o tiene menos de 2 spikes, saltamos
        if spike_array.shape[0] < 2:
            print(f"[Canal {ch}] Muy pocos datos para PCA/KMeans.")
            all_pca.append(None)
            all_labels.append(None)
            continue

        # Escalar los datos
        X_scaled = StandardScaler().fit_transform(spike_array)

        # PCA
        n_components = min(2, X_scaled.shape[1], X_scaled.shape[0])
        pca          = PCA(n_components=n_components)
        X_pca        = pca.fit_transform(X_scaled)
        all_pca.append(X_pca)

        plt.figure(figsize=(6, 6))
        if n_components == 2:
            plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
            plt.xlabel("PC1")
            plt.ylabel("PC2")
        else:
            plt.scatter(X_pca[:, 0], np.zeros_like(X_pca[:, 0]), alpha=0.7)
            plt.xlabel("PC1")
            plt.ylabel("(no PC2)")
        plt.title(f"PCA Canal {ch}\n(varianza explicada: {np.sum(pca.explained_variance_ratio_[:n_components]):.2f})")
        plt.grid(True)
        plt.show()

        # Metodo del codo
        inertia = []
        K = range(1, min(10, len(X_pca) + 1))  # no más clusters que muestras
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_pca)
            inertia.append(kmeans.inertia_)

        kneedle   = KneeLocator(K, inertia, curve='convex', direction='decreasing')
        optimal_k = kneedle.knee
        print(f"[Canal {ch}] Número óptimo de clusters (k): {optimal_k}")

        plt.figure(figsize=(6, 4))
        plt.plot(K, inertia, 'o-', color='blue', label='Inercia')
        
        if optimal_k is not None:
            plt.vlines(optimal_k, plt.ylim()[0], plt.ylim()[1],
                       linestyles='dashed', colors='red', label=f'Codo k={optimal_k}')
        plt.xlabel("Número de clusters (k)")
        plt.ylabel("Inercia (Suma de distancias al cuadrado)")
        plt.title(f"Método del codo Canal {ch}")
        plt.legend()
        plt.grid(True)
        plt.show()

        # KMeans
        if optimal_k is not None and optimal_k > 1:
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            labels = kmeans.fit_predict(X_pca)
            centers = kmeans.cluster_centers_

            print(f"[Canal {ch}] Etiquetas de cada punto:", labels)
            print(f"[Canal {ch}] Centros de los clusters:\n", centers)

            plt.figure(figsize=(6, 6))
            if n_components == 2:
                plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
                plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X')
            else:
                plt.scatter(X_pca[:, 0], np.zeros_like(X_pca[:, 0]), c=labels, cmap='viridis', alpha=0.7)
                plt.scatter(centers[:, 0], np.zeros(optimal_k), c='red', s=200, marker='X')
            plt.xlabel("PC1")
            plt.ylabel("PC2" if n_components == 2 else "(no PC2)")
            plt.title(f"KMeans clustering Canal {ch}")
            plt.grid(True)
            plt.show()

            all_labels.append(labels)
        else:
            all_labels.append(None)

    return all_labels, all_pca

def clusters(list_spikes_ch,all_labels):
    
    """
    Agrupa los índices temporales de espigas (spikes) por canal y por clúster
    a partir de las etiquetas obtenidas en la reducción de dimensiones y clustering.

    Keyword arguments:
    list_spikes_ch : list of lists/arrays
        Lista indexada por canal que contiene los índices de muestra donde se
        detectó cada spike.
    all_labels : list of arrays or None
        Lista indexada por canal con los identificadores de cluster asignados a
        cada spike por el algoritmo de clasificación no supervisada (ej. KMeans).
        Si un canal no contiene espigas o clusters válidos, su valor es None.

    Returns:
    all_spikes_by_cluster : list of lists of lists
        Estructura anidada donde cada elemento corresponde a un canal, el cual
        contiene listas con los índices de los spikes agrupados para cada clúster local:
        all_spikes_by_cluster[canal][cluster_local] = [spike_indices].
    """
    
    all_spikes_by_cluster = []
    
    for ch, labels in enumerate(all_labels):
        list_spikes_channel = list_spikes_ch[ch]

        #Si no hay clusters en este canal
        if labels is None:
            all_spikes_by_cluster.append([])  # guardamos lista vacía
            continue

        clusters_ch        = sorted(set(labels))  # clusters únicos ordenados
        spikes_per_cluster = []

        for c in clusters_ch:
            
            # Selecciona los spikes del cluster c
            idx = [spike for spike, label in zip(list_spikes_channel, labels) if label == c]
            spikes_per_cluster.append(idx)
        
        all_spikes_by_cluster.append(spikes_per_cluster)
    
    return all_spikes_by_cluster

def ids(clusters):
    
    """
    Desempaqueta los clústeres por canal y genera un DataFrame
    asignando un identificador único (ClusterID) a cada unidad.

    Keyword arguments:
    clusters : list of lists of lists
        Estructura anidada devuelta por `clusters()`, organizada jerárquicamente
        como [canal][cluster_local][spikes].

    Returns:
    df_clusters : pandas.DataFrame
        DataFrame con los metadatos de las unidades neuronales identificadas.
        Columnas:
            - 'ClusterID': Identificador único para la sesión.
            - 'Events': Lista de índices de muestra de los spikes de la unidad.
            - 'Canal': Canal físico del electrodo donde se registró la unidad.
            - 'ClusterLocal': Índice ordinal del cluster dentro de ese canal específico (1-based).
    """
    lista_clusters = []

    cluster_id = 0
    for ch in range(len(clusters)):
        channel = clusters[ch]
        for c in range(len(channel)):
            num_cluster = channel[c]
            
            lista_clusters.append({
                "ClusterID": cluster_id,
                "Events": num_cluster,
                "Canal": ch,
                "ClusterLocal": c + 1,
            })
            
            cluster_id += 1
    
    # Convertir a DataFrame
    df_clusters = pd.DataFrame(lista_clusters)
    
    return df_clusters

def artefact(data_filter, info_units, sample_rate, pulses_windows):
    
    """
    Genera un resumen visual (Template y PSTH) para cada clúster y pide
    una decisión de depuración manual (neurona, artefacto, omitir).

    Keyword arguments:
    data_filter : ndarray 2D (tiempo x canales)
        Datos crudos (ej. tu 'data_filter').
    info_units : DataFrame
        Tu 'df_clusters' con 'Events', 'Canal', 'ClusterID'.
    sample_rate : float
        Tasa de muestreo (Hz) (ej. tu 'sample_rate').
    pulses_windows : DataFrame
        DataFrame con los tiempos de los pulsos del LED.
        Debe tener una columna 'Onset_Pulse' en segundos.

    Returns:
    tuple
        Una tupla de listas: (ids_de_neuronas, ids_de_artefactos)
    """

    # Configuración de la ventana de spike (de tu función)
    pre_position = 15
    post_position = 45

    ### Preparamos las listas para guardar decisiones
    ids_de_neuronas = []
    ids_de_artefactos = []
    
    ### Extraemos los tiempos de pulso (en segundos)
    all_led_pulses_sec = pulses_windows['Onset_Pulse'].to_numpy()

    n_clusters = info_units.shape[0]  # número total de clusters
    
    print(f" Iniciando Depuración Manual de {n_clusters} clústeres ")
    print("Para cada clúster, decide:")
    print(" (n) = NEURONA (guardar)")
    print(" (m) = ARTEFACTO (eliminar)")
    print(" (o) = OMITIR (ignorar)")
  

    for index, row in info_units.iterrows():
        
        # Obtenemos datos de la fila
        neuron_spikes = row['Events']   # eventos (índices de muestra)
        channel       = row['Canal']          # canal correspondiente
        cluster_id    = row['ClusterID']   # ID del clúster

        spikes_aligned     = []       # almacenar spikes recortados
        valid_spike_indices = []  # almacenar índices válidos para el PSTH

        ### 2. Creamos la figura con DOS paneles
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f'Análisis ClusterID: {cluster_id} (Canal: {channel})', fontsize=16)
        
        # Plot el template del spike
        
        for spike_idx in neuron_spikes:
            spike_start = int(spike_idx - pre_position)
            spike_end   = int(spike_idx + post_position)

            # Verificar que el spike esté dentro de los límites
            if spike_start < 0 or spike_end > data_filter.shape[0]:
                continue

            # Extraer el spike
            complete_spike = data_filter[spike_start:spike_end, channel]
            
            # Vector de tiempo en ms
            dt_ms          = (1 / sample_rate) * 1000
            time_vector_ms = np.arange(-pre_position, post_position) * dt_ms

            # Graficar spike
            ax1.plot(time_vector_ms, complete_spike, alpha=0.05, color='black')
            
            spikes_aligned.append(complete_spike)
            valid_spike_indices.append(spike_idx) # Guardamos el índice

        ax1.set_xlabel('Tiempo (ms)')
        ax1.set_ylabel('Amplitud')
        ax1.grid(True)
        
        if len(spikes_aligned) > 0:
            all_spikes_array = np.vstack(spikes_aligned)
            template = np.mean(all_spikes_array, axis=0)
            ax1.plot(time_vector_ms, template, color='black', linewidth=2, 
                     label=f'Template (n={len(spikes_aligned)})')
            ax1.set_ylim(all_spikes_array.min() - 20, all_spikes_array.max() + 20)

        # Línea vertical en tiempo cero (posición del spike)
        ax1.axvline(0, color='black', linestyle='--', alpha=0.7)
        ax1.set_title(f'Template (n={len(spikes_aligned)})')
        ax1.legend()
        
        # Plot PSTH
        
        # Definimos los límites de la ventana en segundos 
        pre_psth_sec  = 0.25     # 250 ms antes
        post_psth_sec = 0.25     # 250 ms después
        
        #Duracion del pulso 
        mean_pulse    = np.mean(pulses_windows.Offset_Pulse-pulses_windows.Onset_Pulse)
        pulse_sec = np.round(mean_pulse,3)
        
        # Calculamos el rango total para la búsqueda y el ploteo
        window_start_sec = -pre_psth_sec                            # -0.25 s
        window_end_sec = pulse_sec + post_psth_sec                  # +0.5 s (250ms pulso + 250ms post)
        
        # 3. Definimos los bins
        total_duration_sec = pre_psth_sec + pulse_sec + post_psth_sec # 0.75 s
        bin_size_ms = 5.0 # 10ms por bin
        bins_psth = int((total_duration_sec * 1000) / bin_size_ms) # 
        
        if len(valid_spike_indices) > 0:
            # Convertir índices a segundos
            spike_times_sec = np.array(valid_spike_indices) / float(sample_rate)
            
            relative_times_sec = []
            for pulse_time_sec in all_led_pulses_sec:
                
                # Buscamos spikes en la nueva ventana total
                spikes_en_ventana = spike_times_sec[
                    (spike_times_sec >= pulse_time_sec + window_start_sec) & # (t >= pulse - 0.25s)
                    (spike_times_sec <= pulse_time_sec + window_end_sec)   # (t <= pulse + 0.5s)
                ]
                relative_times_sec.extend(spikes_en_ventana - pulse_time_sec)

            if relative_times_sec:
                relative_times_ms = np.array(relative_times_sec) * 1000
                ax2.hist(relative_times_ms, 
                         bins=bins_psth, 
                         
                         # Rango en ms
                         range=(window_start_sec * 1000, window_end_sec * 1000), # -250 a +500
                         color='slategrey')
        
        ax2.set_title("PSTH")
        ax2.set_xlabel("Tiempo relativo al pulso (ms)")
        ax2.set_ylabel("Conteos de Spikes")
        
        # Sombra para el pulso de 0 a 250 ms
        ax2.axvspan(0, pulse_sec * 1000, color='lightgray', alpha=0.2, label='Pulso LED (0-250ms)')
        # Línea vertical para el inicio del pulso
        ax2.axvline(0, color='black', linestyle='--', alpha=0.7)
        
        ax2.legend()
        ax2.grid(True)
        # Ajustamos el límite X
        ax2.set_xlim(window_start_sec * 1000, window_end_sec * 1000) # -250 a +500    
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        plt.show(block=False) # Muestra la figura sin bloquear

        ### Pedir la decisión manual 
        decision = ""
        while decision not in ['m', 'n', 'o']:
            # El script se pausa AQUÍ, con el plot visible
            decision = input(f"¿Clúster {cluster_id} es ARTEFACTO (m), NEURONA (n), u OMITIR (o)? ")
            decision = decision.lower().strip()
            
        # DESPUÉS del input, se cierra la figura
        plt.close(fig) 
            
        if decision == 'm':
            ids_de_artefactos.append(cluster_id)
            print(f" Marcado como ARTEFACTO. ({cluster_id})")
        elif decision == 'n':
            ids_de_neuronas.append(cluster_id)
            print(f" Marcado como NEURONA. ({cluster_id})")
        else:
            print(f" OMITIDO. ({cluster_id})")
            pass # Se omite

    print("\n Depuración Terminada")
    print(f"Total de Artefactos identificados: {len(ids_de_artefactos)}")
    print(f"Total de Neuronas identificadas: {len(ids_de_neuronas)}")
    
    ### 4. Devolvemos las listas de decisión
    return ids_de_neuronas, ids_de_artefactos

def spikes_array_id(data, info_units, sample_rate):
    
    """
    Genera un arreglo de spikes por canal y por neurona a partir de datos crudos.
    Cada spike se alinea en la muestra 15 de una ventana de 60 muestras.

    Keyword arguments:
    data : ndarray 2D (tiempo x canales)
        Datos crudos.
    info_units : DataFrame
        Información de las neuronas/clusters con columnas:
        - 'Events': lista de spikes de cada neurona
        - 'Canal': canal correspondiente a cada neurona
        - 'ClusterID': identificador del cluster
    sample_rate : float
        Tasa de muestreo (Hz).

    RReturns:
    neurons : list
        Lista de listas. Cada elemento corresponde a una neurona y contiene
        los spikes alineados como arrays de tamaño (windowSpike,).
    """

    # Configuración de la ventana de spike
    windowSpike   = 60
    pre_position  = 15
    post_position = 45

    n_neurons = info_units.shape[0]  # número total de neuronas/clusters
    neurons = []

    for n in range(n_neurons):
        neuron_spikes = info_units.Events[n]   # eventos (spikes) de la neurona
        channel = info_units.Canal[n]          # canal correspondiente
        cluster_id = info_units.ClusterID[n]

        spikes_aligned = []  # almacenar spikes recortados para esta neurona

        plt.figure(figsize=(6, 4))  # figura por neurona

        for spike in neuron_spikes:
            spike_start = spike - pre_position
            spike_end = spike + post_position

            # Verificar que el spike esté dentro de los límites del data
            if spike_start < 0 or spike_end > data.shape[0]:
                continue

            # Extraer el spike del canal correspondiente
            complete_spike = data[spike_start:spike_end, channel]

            # Vector de tiempo en ms
            dt_ms = (1 / sample_rate) * 1000
            time_vector = np.arange(0, windowSpike) * dt_ms

            # Graficar spike
            plt.plot(time_vector, complete_spike, alpha=0.02, color=ce)
            spikes_aligned.append(complete_spike)

        # Línea vertical en tiempo cero (posición del spike)
        plt.axvline(time_vector[pre_position], color=ce, linestyle='--', alpha=0.7)
        plt.xlabel('Tiempo (ms)')
        plt.ylabel('Voltaje (uV)')
        plt.title(f'Todos los spikes de la unidad: Cluster {cluster_id}')

        # Ajustar límites de la figura si hay spikes
        if len(spikes_aligned) > 0:
            all_spikes_array = np.vstack(spikes_aligned)
            plt.ylim(all_spikes_array.min() - 20, all_spikes_array.max() + 20)

        plt.show()
        neurons.append(spikes_aligned)

    return neurons

def clusters_per_channel(total_spikes, pulses_windows, sample_rate, pre_time_window=250, post_time_window=250):
    """
    CMatrices binarias de eventos peri-estímulo para cada
    clúster depurado, alineando las espigas con respecto a las ventanas de
    pulsos de estimulación (LED).

    Keyword arguments:
    total_spikes : pandas.DataFrame
        DataFrame que contiene la información de las unidades neuronales filtradas, 
        con la columna 'Events' que almacena los índices de muestra
        donde ocurrió cada spike.
    pulses_windows : pandas.DataFrame
        DataFrame con los tiempos de inicio y fin de cada pulso en segundos.
        Contiene las columnas 'Onset_Pulse' y 'Offset_Pulse'.
    sample_rate : float
        Frecuencia de muestreo del registro electrofisiológico en Hertz (Hz).
    pre_time_window : float, opcional
        Ventana de tiempo antes del inicio del pulso ('Onset_Pulse') en milisegundos (ms).
        El valor por defecto es 250.
    post_time_window : float, opcional
        Ventana de tiempo después del final del pulso ('Offset_Pulse') en milisegundos (ms).
        El valor por defecto es 250.

    Returns:
    events_clusters : list of ndarray 2D
        Lista donde cada elemento corresponde a una matriz binaria para un clúster específico.
        Las dimensiones de cada matriz son (n_ensayos, tamaño_ventana_en_muestras), donde:
            - n_ensayos: Número total de pulsos/ensayos evaluados.
            - tamaño_ventana_en_muestras: Longitud total de la ventana peri-estímulo.
            - Valor 1: Indica la presencia de un potencial de acción en esa muestra temporal.
            - Valor 0: Indica ausencia de disparo.
    """
    
    pre_time_samples  = int((pre_time_window * sample_rate) / 1000)
    post_time_samples = int((post_time_window * sample_rate) / 1000)
    
    # Usamos reset_index para que el contador de ensayos siempre empiece en 0
    pulses          = pulses_windows.reset_index(drop=True).copy()
    pulses["Start"] = (pulses["Onset_Pulse"] * sample_rate) - pre_time_samples
    pulses["End"]   = (pulses["Offset_Pulse"] * sample_rate) + post_time_samples
    
    size_w          = int(np.ceil((pulses["End"] - pulses["Start"]).max()))
    events_clusters = []

    for s in range(len(total_spikes)):
        spike_indices   = np.array(total_spikes.iloc[s]['Events'])
        events_in_essay = np.zeros((len(pulses), size_w))
        
        # En lugar de iterar spike por spike, filtramos por ventana
        for i, row in pulses.iterrows():
            start_w = row["Start"]
            end_w   = row["End"]
            
            # Buscamos qué spikes caen en este rango de una sola vez (Vectorizado)
            valid_spikes = spike_indices[(spike_indices >= start_w) & (spike_indices < end_w)]
            positions    = (valid_spikes - start_w).astype(int)
            
            # Solo llenamos si están dentro del límite de la matriz
            positions                     = positions[positions < size_w]
            events_in_essay[i, positions] = 1
            
        events_clusters.append(events_in_essay)
    
    return events_clusters
