import numpy as np
import os

# Anlegen des Ordners für die Ergebnisse
if not os.path.exists("ergebnisse"):
    os.makedirs("ergebnisse")

# Funktion zum Schreiben der Inhalte in eine Datei
def write_array_to_dat_file(file_name, data_array):
    try:
        with open("ergebnisse/" + file_name +'.dat', 'w') as file:
            for item in data_array:
                file.write(str(item) + '\n')
        print(f"Daten wurden in '{file_name}' geschrieben.")
    except Exception as e:
        print(f"Fehler beim Schreiben in '{file_name}': {e}")

# Laden der Daten aus einer gegebenen Datei
data = np.loadtxt('dstexample.dat')

# Implementierung und Anwendung der DST-II
def discrete_sine_transform(data):
  N = len(data)
  dst = np.zeros(N)

  # k=1,...,N:
  for k in range(1, N + 1):
    sum_val = 0
    # Summe von n=1 bis N:
    for n in range(1, N + 1):
      # Berechnung des Terms innerhalb der Summe für jedes n:
      sum_val += data[n - 1] * np.sin((np.pi / N) * (n - 0.5) * k)
    # Zuweisung des k-ten Wertes (dst Index 0 entspricht k=1)
    dst[k - 1] = sum_val / np.sqrt(N)
  return dst

dst_data = discrete_sine_transform(data)

write_array_to_dat_file("transformierteDaten", dst_data)

# Herausfiltern bestimmter Bereiche aus der DST
def filter_dst(dst, m0, m1):
  # dst-Array beginnend ab Index 0, aber eigentlich für k=1,...,N - Korrektur:
  start_idx = m0 - 1
  end_idx = m1 - 1
  # Alle Werte außerhalb des definierten Bereiches auf 0 setzen
  filtered_dst = [data if start_idx <= i <= end_idx else 0 for i, data in enumerate(dst)]
  return filtered_dst

# Anwendung der Filteroperation auf alle notwendigen Bereiche
dst_1_2 = filter_dst(dst_data, 1, 2)
dst_3_6 = filter_dst(dst_data, 3, 6)
dst_7_10 = filter_dst(dst_data, 7, 10)
dst_1_N = filter_dst(dst_data, 1, len(dst_data))

# inverse DST
def inverse_discrete_sine_transform(dst_data):
  N = len(dst_data)
  data_reconstructed = np.zeros(N)

  # k=1,...,N (für jeden Datenpunkt):
  for k in range(1, N + 1):
    sum_val = 0
    # Summe von k=1 bis N-1:
    for n in range(1, N):
      # Berechnung des Terms innerhalb der Summe
      sum_val += dst_data[n - 1] * np.sin((np.pi / N) * (k - 0.5) * n)
    # Berechnungen mit dem zuvor ermittelten Summenterm
    sum_val = (2 * sum_val + (-1)**(k - 1) * dst_data[N - 1]) / np.sqrt(N)
    # Zuweisung des Wertes zum rekonstruierten Datensatz
    data_reconstructed[k - 1] = sum_val
  return data_reconstructed

# Rücktransformation der bearbeiteten Daten
reconstructedData1_2 = inverse_discrete_sine_transform(dst_1_2)
reconstructedData3_6 = inverse_discrete_sine_transform(dst_3_6)
reconstructedData7_10 = inverse_discrete_sine_transform(dst_7_10)
reconstructedData1_N = inverse_discrete_sine_transform(dst_1_N)

# Ausgabe der berechneten Listen
write_array_to_dat_file("rekonstruierteDaten_1_2", reconstructedData1_2)
write_array_to_dat_file("rekonstruierteDaten_3_6", reconstructedData3_6)
write_array_to_dat_file("rekonstruierteDaten_7_10", reconstructedData7_10)
write_array_to_dat_file("rekonstruierteDaten_1_N", reconstructedData1_N)