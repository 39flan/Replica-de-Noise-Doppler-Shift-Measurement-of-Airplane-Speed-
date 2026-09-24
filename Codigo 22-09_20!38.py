import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


def desaturar_senal(audio, umbral_factor=0.98):
    # Repara crestas planas por saturación (clipping) usando interpolación cúbica
    audio_reparado = audio.copy().astype(float)  # copia para no alterar el original
    max_val = np.max(np.abs(audio_reparado))  # valor maximo
    umbral = max_val * umbral_factor  # umbral de deteccion

    # si algo supera el umbral se considera potencialmente saturado

    # Identificar las muestras que superan el umbral de saturación
    saturado = np.abs(audio_reparado) >= umbral

    if not np.any(saturado):
        return audio_reparado

    indices = np.arange(len(audio_reparado))  # hace un array con las posiciones de los audios
    limpios = indices[~saturado]  # selecciona solo los que no estan saturados
    valores_limpios = audio_reparado[~saturado]  # ordena los valores limpios

    # Se requieren suficientes puntos limpios para la interpolación cúbica (polinomio grado 3)
    if len(limpios) < 4:
        return audio_reparado  # si no hay suficientes devuelve el audio sin hacer nada

    # Ajustar un spline cúbico en las zonas sanas y evaluar en los puntos saturados
    cs = CubicSpline(limpios, valores_limpios)  # construye la correcion
    indices_saturados = indices[saturado]  # Obtiene las posiciones a arreglar
    audio_reparado[saturado] = cs(indices_saturados)  # Reemplaza los valores saturados

    # Normalizar para evitar nueva saturación tras la interpolación
    audio_reparado = audio_reparado / np.max(np.abs(audio_reparado))

    return audio_reparado


# Cargar el audio
fs, audio = wavfile.read("/media/flan/disco/python/TOPICOS1/uwu.wav")
if len(audio.shape) > 1:
    audio = audio[:, 0]

# Reparar saturación/clipping
audio = desaturar_senal(audio)

#  Calcular el espectrograma con el audio procesado
frecuencias, tiempos, Sxx = signal.spectrogram(audio, fs, nperseg=2048)

# 5. Visualización
plt.figure(figsize=(10, 5))
plt.pcolormesh(tiempos, frecuencias, 10 * np.log10(Sxx + 1e-10), shading='auto', cmap='inferno')

plt.ylim(0, 4000)
plt.ylabel('Frecuencia [Hz]')
plt.xlabel('Tiempo [s]')
plt.title('Análisis Doppler de Avión')
plt.colorbar(label='Intensidad [dB]')
plt.tight_layout()
plt.savefig("esp0.png")
plt.show()
# 1. Cargar el audio
fs, audio = wavfile.read("/media/flan/disco/python/TOPICOS1/uwu.wav")
if len(audio.shape) > 1:
    audio = audio[:, 0]

# 2. Desaturar la señal antes del análisis
audio = desaturar_senal(audio)

# 2. Calcular el espectrograma (usamos un nperseg más grande para mejor definición en frecuencia)
frecuencias, tiempos, Sxx = signal.spectrogram(audio, fs, nperseg=2048)

# 3. Encontrar la frecuencia pico en CADA instante de tiempo (columna por columna)
# axis=0 le dice a argmax que busque el valor máximo verticalmente en cada columna
indice_picos = np.argmax(Sxx, axis=0)
frecuencias_dominantes = frecuencias[indice_picos]

# 4. Aislar las zonas estables (Evitando los silencios del inicio y del final)
# Supongamos que el avión cruza por la mitad del audio:
total_bloques = len(tiempos)

# Zona de ACERCAMIENTO: Tomamos un bloque seguro en la primera mitad (ej. del 20% al 45%)
inicio_app = int(total_bloques * 0.45)
fin_app = int(total_bloques * 0.55)
frecs_app = frecuencias_dominantes[inicio_app:fin_app]  # Usamos mediana para ignorar ruido extremo
f_app = np.median(frecs_app)

# Zona de ALEJAMIENTO: Tomamos un bloque seguro en la segunda mitad (ej. del 55% al 80%)
inicio_rec = int(total_bloques * 0.75)
fin_rec = int(total_bloques * 0.85)
frecs_rec = frecuencias_dominantes[inicio_rec:fin_rec]
f_rec = np.median(frecs_rec)

# 5. Calcular la velocidad del avión con la fórmula de Doppler acústico
v_sonido = 339.09  # m/s a 13 C
velocidad_ms = v_sonido * np.abs(((f_app - f_rec) / (f_app + f_rec)))
velocidad_kmh = velocidad_ms * 3.6

print(f"Frecuencia de acercamiento (f_app): {f_app:.2f} Hz")
print(f"Frecuencia de alejamiento (f_rec): {f_rec:.2f} Hz")
print(f"----------------------------------------")
print(f"Velocidad del avión: {velocidad_ms:.2f} m/s ({velocidad_kmh:.1f} km/h)")

# 6. Calculo de los errores instrumentales de las frecuencias:

err_f = 44100/2048

print(f"Error instumental de frecuencia de aproximacion = {f_app:.2f} ± {err_f:.2f} Hz")
print(f"Error instumental de frecuencia de alejamiento = {f_rec:.2f} ± {err_f:.2f} Hz")

dv_dfapp = v_sonido * 2 * f_rec / (f_app + f_rec)**2
dv_dfreс = -v_sonido * 2 * f_app / (f_app + f_rec)**2

error_velocidad =np.sqrt(
    (dv_dfapp * err_f)**2 +
    (dv_dfreс * err_f)**2
)

# 7. Calculo de los errores instrumentales de las velocidades:

print(f"Velocidad = {velocidad_ms:.2f} ± {error_velocidad:.2f} m/s")
print(f"Velocidad = {velocidad_kmh:.1f} ± {error_velocidad*3.6:.1f} km/h")

# 8. Calculo de los errores estadisticos de las frecuencias:

err_f_app = np.std(frecs_app, ddof=1)
err_f_rec = np.std(frecs_rec, ddof=1)

print(f"Error estadistico de frecuencia de aproximacion = {f_app:.2f} ± {err_f_app:.2f} Hz")
print(f"Error estadistico de frecuencia de alejamiento = {f_rec:.2f} ± {err_f_rec:.2f} Hz")

dv_dfapp = v_sonido * 2 * f_rec / (f_app + f_rec)**2
dv_dfreс = -v_sonido * 2 * f_app / (f_app + f_rec)**2

error_velocidad =np.sqrt(
    (dv_dfapp * err_f_app)**2 +
    (dv_dfreс * err_f_rec)**2
)

# 9. Calculo de los errores estadisticos de las velocidades:

print(f"Velocidad = {velocidad_ms:.2f} ± {error_velocidad:.2f} m/s")
print(f"Velocidad = {velocidad_kmh:.1f} ± {error_velocidad*3.6:.1f} km/h")
