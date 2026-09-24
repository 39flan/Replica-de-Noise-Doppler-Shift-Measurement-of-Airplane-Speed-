import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
#fs, audio = wavfile.read("WhatsApp-Audio-2026-09-09-at-12.33.58.wav")
#fs, audio = wavfile.read("Grabación.wav")

fs, audio = wavfile.read("/media/flan/disco/python/TOPICOS1/AVION.wav")

if len(audio.shape) > 1:
    audio = audio[:, 0]
N = len(audio)
#w = 6*np.pi + 5
#fs = 50
#cos = np.cos(w*(np.arange(N)/fs))

#Como cos(i) = cos(1*t), sabiendo que en ondas cos(2pi*f*t) o cos(wt), esto signifnica que w = 1, osea 2pif = 1, o f = 1/2pi, dando como resultado 0.159, deberíamos obtener un valor de 0.159 approx

#print(cos)

#%%
print(N)
print(fs)
resultado_complejo = np.fft.fft(audio)
mag = np.abs(resultado_complejo)

mag = (mag/len(audio))*2

frec = np.fft.fftfreq(n=len(audio),d = 1/fs)
#%%

#%%
mitad = N//2


frec_reales = frec[:mitad]
mag_reales = mag[:mitad]

indice_maximo = np.argmax(mag_reales)

frecuencia_exacta = frec_reales[indice_maximo]

amplitud_exacta = mag_reales[indice_maximo]


error = fs/N

print(frecuencia_exacta,error)
plt.xlim(0,5000)
plt.plot(frec_reales,mag_reales)

print(amplitud_exacta)

#%%

# 5. Graficar
plt.figure(figsize=(10, 4))
plt.plot(frec_reales, mag_reales, color='purple')
plt.title("Espectro de Frecuencias de tu Audio de WhatsApp")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.grid(True, alpha=0.3)
plt.show()
plt.savefig("testowo")
#%%
import numpy as np
from scipy.io import wavfile
from scipy import signal

# 1. Cargar el audio
fs, audio = wavfile.read("/media/flan/disco/python/TOPICOS1/AVION.wav")
if len(audio.shape) > 1:
    audio = audio[:, 0]

# 2. Calcular el espectrograma (usamos un nperseg más grande para mejor definición en frecuencia)
frecuencias, tiempos, Sxx = signal.spectrogram(audio, fs, nperseg=16384)
# 3. Encontrar la frecuencia pico, pero SOLO en la zona del motor
# Definimos los límites (puedes jugar con estos números si miras el gráfico)
#limite_inferior = 200  # Ignoramos el viento bajo 100 Hz
#limite_superior = 450  # Ignoramos pájaros y siseos sobre 800 Hz

# Buscamos en qué posiciones (índices) de la regla están esos límites
#idx_min = np.where(frecuencias > limite_inferior)[0][0]
#idx_max = np.where(frecuencias < limite_superior)[0][-1]

# Recortamos la matriz del espectrograma para mirar SOLO esa franja horizontal
Sxx_motor = Sxx[idx_min:idx_max, :]

# Ahora buscamos el pico máximo verticalmente, pero solo en la franja del motor
indice_picos_recortado = np.argmax(Sxx_motor, axis=0)

# Le sumamos el idx_min para que el índice coincida con el arreglo original
frecuencias_dominantes = frecuencias[indice_picos_recortado + idx_min]

# 4. Aislar las zonas estables (Evitando los silencios del inicio y del final)
# Supongamos que el avión cruza por la mitad del audio:
total_bloques = len(tiempos)

# Zona de ACERCAMIENTO: Tomamos un bloque seguro en la primera mitad (ej. del 20% al 45%)
inicio_app = int(total_bloques * 0.35)
fin_app = int(total_bloques * 0.48)
f_app = np.median(frecuencias_dominantes[inicio_app:fin_app]) # Usamos mediana para ignorar ruido extremo

# Zona de ALEJAMIENTO: Tomamos un bloque seguro en la segunda mitad (ej. del 55% al 80%)
inicio_rec = int(total_bloques * 0.55)
fin_rec = int(total_bloques * 0.90)
f_rec = np.median(frecuencias_dominantes[inicio_rec:fin_rec])

# 5. Calcular la velocidad del avión con la fórmula de Doppler acústico
v_sonido = 339.09  # m/s a temperatura ambiente
velocidad_ms = v_sonido * np.abs(((f_app - f_rec) / (f_app + f_rec)))
velocidad_kmh = velocidad_ms * 3.6

print(f"Frecuencia de acercamiento (f_app): {f_app:.2f} Hz")
print(f"Frecuencia de alejamiento (f_rec): {f_rec:.2f} Hz")
print(f"----------------------------------------")
print(f"Velocidad del avión: {velocidad_ms:.2f} m/s ({velocidad_kmh:.1f} km/h)")
#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal

# 1. Cargar el audio
fs, audio = wavfile.read("/media/flan/disco/python/TOPICOS1/AVION.wav")
if len(audio.shape) > 1:
    audio = audio[:, 0]

# 2. Calcular el espectrograma
frecuencias, tiempos, Sxx = signal.spectrogram(audio, fs, nperseg=2048)

# 3. Graficar correctamente usando pcolormesh
plt.figure(figsize=(10, 5))
plt.pcolormesh(tiempos, frecuencias, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
plt.ylabel('Frecuencia [Hz]')
plt.xlabel('Tiempo [s]')
plt.title('Espectrograma - Aterrizaje de Avión')
plt.colorbar(label='Intensidad [dB]')
plt.show()
plt.savefig("testowo2")