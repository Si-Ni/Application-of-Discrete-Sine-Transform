import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os

#Anlegen des Ordners für die Ergebnisse
if not os.path.exists("grafischeErgebnisse"):
    # Ordner erstellen
    os.makedirs("grafischeErgebnisse")

#Laden der Daten aus einer gegebenen Datei
data = np.loadtxt('dstexample.dat')

#Konfiguration der Schriftart für die Diagramme
plt.rc('font', family='serif')


#Section: Hilfsfunktionen für die Formatierung der Grafiken
def tickFormatter(value, pos):
    if value == 0:
        return ""
    else:
        return f"{value:.1f}"

def tickFormatterTruncate(value, pos):
    if value == 0:
        return ""
    else:
        return f"{value:.0f}"

def configureAxis(ax):
  ax.spines['bottom'].set_position('zero')
  ax.spines['left'].set_position('zero')

  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

def configureAxisForData(ax):
  ax.xaxis.set_major_locator(plt.MultipleLocator(50))
  ax.xaxis.set_minor_locator(plt.MultipleLocator(10))

  ax.set_xlabel('n')
  ax.set_ylabel('Daten (D_n)')

  ax.xaxis.set_major_formatter(FuncFormatter(tickFormatterTruncate))
  ax.yaxis.set_major_formatter(FuncFormatter(tickFormatter))

  ax.xaxis.set_label_coords(1.06, 0.51)

  configureAxis(ax)
#EndSection
  
#Section: erstes Diagramm zur Darstellung der Daten
fig1, ax1 = plt.subplots()
time = np.arange(1, len(data) + 1, 1)

plt.scatter(time, data, color='red', marker='o', s=4, zorder=3)

configureAxisForData(ax1)

ax1.yaxis.set_major_locator(plt.MultipleLocator(0.5))
ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

plt.savefig('grafischeErgebnisse/dataPoints.pdf', format='pdf')
plt.close(fig1)
#EndSection

#Section: zweites Diagramm zur Darstellung der Daten
fig2, ax2 = plt.subplots()
time = np.arange(0, len(data) * 0.01, 0.01)

plt.plot(time, data, color='red', marker='', linewidth=0.8, zorder=3)

ax2.xaxis.set_major_locator(plt.MultipleLocator(0.5))
ax2.xaxis.set_minor_locator(plt.MultipleLocator(0.1))

ax2.yaxis.set_major_locator(plt.MultipleLocator(0.5))
ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

ax2.set_xlabel('Zeit (Δt)')
ax2.set_ylabel('Daten (D_n)')

ax2.xaxis.set_major_formatter(FuncFormatter(tickFormatter))
ax2.yaxis.set_major_formatter(FuncFormatter(tickFormatter))

ax2.xaxis.set_label_coords(1.06, 0.51)

configureAxis(ax2)

plt.savefig('grafischeErgebnisse/dataFunction.pdf', format='pdf')
plt.close(fig2)
#EndSection

#Section: Implementierung und Anwendung der DST-II
def discrete_sine_transform(data):
  N = len(data)
  dst = np.zeros(N)

  #k=1,...,N:
  for k in range(1, N + 1):
    sum_val = 0
    #Summe von n=1 bis N:
    for n in range(1, N + 1):
      # Berechnung des Terms innerhalb der Summe für jedes n:
      sum_val += data[n - 1] * np.sin((np.pi / N) * (n - 0.5) * k)
    # Zuweisung des k-ten Wertes (dst Index 0 entspricht k=1)
    dst[k - 1] = sum_val / np.sqrt(N)
  return dst

dst_data = discrete_sine_transform(data)
time = np.arange(1, len(dst_data) + 1, 1)
#EndSection

#Section: Diagramm für die berechnete DST
fig3, ax3 = plt.subplots()
plt.scatter(time, dst_data, color='red', marker='o', s=8, zorder=2)

ax3.xaxis.set_major_locator(plt.MultipleLocator(50))
ax3.xaxis.set_minor_locator(plt.MultipleLocator(10))

ax3.yaxis.set_major_locator(plt.MultipleLocator(2))
ax3.yaxis.set_minor_locator(plt.MultipleLocator(0.5))

ax3.set_xlabel('k')
ax3.set_ylabel('DST(D)')

ax3.xaxis.set_major_formatter(FuncFormatter(tickFormatterTruncate))
ax3.yaxis.set_major_formatter(FuncFormatter(tickFormatterTruncate))

configureAxis(ax3)

plt.savefig('grafischeErgebnisse/dstData.pdf', format='pdf')
plt.close(fig3)
#EndSection

#Section: Herausfiltern bestimmter Bereiche aus der DST
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
#EndSection

#Section: inverse DST
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
#EndSection

#Section: Diagramm für die inverse DST k = 1, 2
fig4, ax4 = plt.subplots()
plt.scatter(time, inverse_discrete_sine_transform(dst_1_2), color='red', marker='o', s=4, zorder=3)

configureAxisForData(ax4)
ax4.yaxis.set_major_locator(plt.MultipleLocator(0.5))
ax4.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

plt.savefig('grafischeErgebnisse/reconstructedData1_2.pdf', format='pdf')
plt.close(fig4)
#EndSection

#Section: Diagramm für die inverse DST k = 3, 4, 5, 6
fig5, ax5 = plt.subplots()
plt.scatter(time, inverse_discrete_sine_transform(dst_3_6), color='red', marker='o', s=4, zorder=3)

configureAxisForData(ax5)
ax5.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax5.yaxis.set_minor_locator(plt.MultipleLocator(0.02))

plt.savefig('grafischeErgebnisse/reconstructedData3_6.pdf', format='pdf')
plt.close(fig5)
#EndSection

#Section: Diagramm für die inverse DST k = 7, 8, 9, 10
fig6, ax6 = plt.subplots()
plt.scatter(time, inverse_discrete_sine_transform(dst_7_10), color='red', marker='o', s=4, zorder=3)

configureAxisForData(ax6)
ax6.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax6.yaxis.set_minor_locator(plt.MultipleLocator(0.02))

plt.savefig('grafischeErgebnisse/reconstructedData7_10.pdf', format='pdf')
plt.close(fig6)
#EndSection

#Section: Diagramm für die inverse DST alle k
fig7, ax7 = plt.subplots()
plt.scatter(time, inverse_discrete_sine_transform(dst_1_N), color='red', marker='o', s=4, zorder=3)

configureAxisForData(ax7)
ax7.yaxis.set_major_locator(plt.MultipleLocator(0.5))
ax7.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

plt.savefig('grafischeErgebnisse/reconstructedData1_N.pdf', format='pdf')
plt.close(fig7)
#EndSection