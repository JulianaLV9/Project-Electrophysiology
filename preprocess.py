#Phototagging
"""
Created on Feb Tue 11 12:37:57 2025

@author: julianalozavaqueiro


Codigo creado para procesar registros multiunitarios.
El usuario va a cargar su datos de tipo "oebin". A partir de la informacion
analogica se obtendrá la actividad neuronal y a partir de la digital se 
los eventos a partir de las entradas TTL del sistema de adquisición.  

"""


##  Libraries  ##
#Path
from pathlib import Path

#Sys 
import sys 

#Os
import os
#sys.path.append('/Users/julianalozavaqueiro/Documents/Electro_Codes/Phototagging')

#OpenEphys library
from open_ephys.analysis import Session

#Numpy
import numpy as np 

#Pandas
import pandas as pd

#OS
import os

#Pickle
import pickle

#Specific functions
from functions_for_preprocess import select_folder, band_pass_filter, select_train, channel_vision, threshold, peaks, train_array, spikes_array, pca_analysis, clusters,ids, artefact, spikes_array_id, clusters_per_channel


##  Directory of the data
directory = select_folder()
#directory = r'/Volumes/ReachingJLV/Registros_ratas_anestesiadas/CS/G1/G1_C7_1550um_22mW_250ms2025-09-11_18-11-51/Record Node 114'
session   = Session(directory)

# Record Node
recordings = session.recordings 
# print(session.recordnodes[0].recordings[0]) #En caso de que haya record nodes hay que identificarlos a partir de lo que habita aqui

# Binary record
recording = recordings[0] #Carga sample numbers, timestamps and metadata

# Continuous data (Analog signal of each channel)
all_data   = len(recording.continuous[0].timestamps)
data       = recording.continuous[0].get_samples(start_sample_index=0, end_sample_index=all_data)  #Cambiar el largo del "sample" cuando se requiera un segmento. 
metadata   = recording.continuous[0].metadata #Carga toda la metadata 
timestamps = recording.continuous[0].timestamps #Carga las estampas de tiempo

# Event data (Binary information preceding from TTL inputs (Pulses)
events = recording.events

# Synchronization with the main line of the register (Se sincroniza la informacion analogica y digital)
processor_id  = recording.events["processor_id"].iloc[0] #Si se pierde el processor ID buscar con: "recording.events.head()"
stream_name   = recording.events["stream_name"].iloc[0]
recording.add_sync_line(1,processor_id,stream_name, main=True)    # use as the main stream

# Sample rate
sample_rate = metadata['sample_rate']

# Apply a band pass to visualize the signal filtered
data_filtered = band_pass_filter(data, sample_rate)

# Aligning Time and Converting to Seconds
initial_signal             = timestamps[0]
aligned_timestamps         = timestamps - initial_signal #En segundos 
aligned_timestamps_minutes = aligned_timestamps / 60
duration_record            = aligned_timestamps[-1] / 60

# Timestamps events
recording.compute_global_timestamps()
timestamps_events             = events['global_timestamp']
initial_events                = timestamps_events[0] 
aligned_timestamps_events_sec = timestamps_events - initial_signal #En segundos 
aligned_timestamps_events_min = aligned_timestamps_events_sec / 60

# Array of timestamps
aligned_timestamps_events_sec = aligned_timestamps_events_sec.to_numpy()

print(f'Longitud de aligned_timestamps: {len(aligned_timestamps)}') #Longitud del registro en # muestras
print(f'Longitud de data: {data.shape[0]}')

# Cut timestamps at length of data
if len(aligned_timestamps) > data_filtered.shape[0]:
    aligned_timestamps = aligned_timestamps[:data_filtered.shape[0]]
    print("Timestamps recortados para que coincidan con la longitud de 'data'")

# Pulse trains (Trenes de estimulacion con el LED de optogenética)
diff_pulses = np.diff(aligned_timestamps_events_sec)

# Pulse windows
pulses_windows                = aligned_timestamps_events_sec.reshape(-1,2)
pulses_windows                = pd.DataFrame(pulses_windows, columns=["Onset_Pulse","Offset_Pulse"])
train_windows                 = select_train(pulses_windows)
size_pulse                    = round(np.mean(pulses_windows.Offset_Pulse-pulses_windows.Onset_Pulse),3)
  
# Visualize channels
begin       = int(30 * sample_rate) #Seconds
select_data = int(35 * sample_rate)
channel_vision(data_filtered[begin:select_data], aligned_timestamps[begin:select_data], sample_rate, pulses_windows)

# Treshold channel (Umbralizar)
threshold_data  = threshold(data_filtered)

# Select peaks
spikes_channel_data, binary_matrix_data, time_matrix_data, spikes_data = peaks(data_filtered, threshold_data, sample_rate)

# Array pulse trains
train_array_3d = train_array(train_windows,spikes_data, sample_rate) #array de tres dimensiones (#trenes,#tiempoVentanaTrenes,#Canales) donde se encuentra el indice en el que se detecto un  spike

# Spike per channel list
list_spikes_ch, all_spikes_array = spikes_array(data_filtered,train_array_3d, sample_rate)

## DIMENSION REDUCTION AND CLUSTERING

# PCA Analysis
all_labels, pca_results = pca_analysis(all_spikes_array) 

# Clusterization of the units (KMeans)
num_clusters = clusters(list_spikes_ch,all_labels)

# Clusters by identity
info_units    = ids(num_clusters)
ids_units     = info_units[["ClusterID", "Events"]].copy() #Tenemos información por identidad y momento en el que sucedio un spike
info_clusters = info_units[["ClusterID", "Canal","ClusterLocal"]].copy()

# Led artefact
ids_neurons, ids_artefacts = artefact(data_filtered, info_units, sample_rate, pulses_windows) #Aqui puedo meter ya la variable del pulso

# Filter the units without the artefact
spikes_filter = info_units['ClusterID'].isin(ids_neurons)
total_spikes  = info_units[spikes_filter].copy()
total_spikes  = total_spikes.reset_index(drop=True) 

# Plot all the spikes filtered without the artefact
spike_clusters = spikes_array_id(data_filtered, total_spikes, sample_rate)

# Raster per cluster
events_clusters = clusters_per_channel(total_spikes,pulses_windows, sample_rate, pre_time_window = 250, post_time_window = 250)

# %%

## Make our variables
metadata_info = {
    'sample_rate': sample_rate,
    'duration_minutes': duration_record,
    'directory_path': directory,
    'n_channels': data_filtered.shape[1] if data_filtered.ndim > 1 else 1,
    'date_processed': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }

register_output = {
    'data_filtered': data_filtered,       
    'all_spikes': all_spikes_array,      
    'spikes_per_channel': list_spikes_ch,
    'pulses_windows': pulses_windows,
    'num_clusters': num_clusters,
    'events_clusters': events_clusters,
    'total_clusters_id': total_spikes,
    'spike_clusters': spike_clusters
    }  

#Save this variables
separator      = '/Record Node'
idx_name       = directory.find(separator)
save_directory = directory[:idx_name]
file_name_m    = 'metadata.pkl'
file_name_r    = 'register_output.pkl'
#file_name_e    = 'events_clusters.pkl'
save_path_m    = os.path.join(save_directory,file_name_m)
save_path_r    = os.path.join(save_directory,file_name_r)
#save_path_e    = os.path.join(save_directory,file_name_e)

with open(save_path_m, 'wb') as f:
    pickle.dump(metadata_info, f) 

print(f"Metadata dictionary saved successfully at: {save_directory}")

with open(save_path_r, 'wb') as f:
    pickle.dump(register_output, f) 

print(f"Register variables dictionary saved successfully at: {save_directory}")

#with open(save_path_e, 'wb') as f:
  
   # pickle.dump(events_cluster, f) 
    #print(f"Events clusters saved successfully at: {save_directory}")