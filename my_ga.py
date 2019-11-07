import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
import numpy.random as random
from deap import base, creator, tools
from operator import attrgetter
from math import ceil

class Individuo:
    def __init__(self):
        self.coordenadas = []
        self.fitness = 0


class Poblacion:
    def __init__(self):
        self.individuos = []


def check_limite(mapa, punto):
    flg_contains = mapa.geometry.contains(punto)
    return flg_contains.any()


def crear_coordenada(mapa):
    x_min, y_min, x_max, y_max = mapa.total_bounds
    while True:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        point = Point(x, y)

        if check_limite(mapa, point):
            break
    return point


def crear_individuo(n_repeat, mapa):
    individuo = Individuo()
    for i in range(n_repeat):
        punto = crear_coordenada(mapa)
        individuo.coordenadas.append(punto)
    return individuo

    # fig, ax = plt.subplots(figsize=(15, 15))
    # mapa.plot(ax=ax)
    # serie = ind
    # serie.plot(ax=ax, color='green', marker='+', label='aleatorio')
    # plt.show()


def calcular_nni(individuo):
    puntos = []
    for p in individuo.coordenadas:
        puntos.append([np.radians(p.x), np.radians(p.y)])
    nbrs = NearestNeighbors(n_neighbors=2, metric='haversine').fit(puntos)
    distances, indices = nbrs.kneighbors(puntos)
    distances = distances * 6371000 / 1000
    # este array contiene para c/ punto, la distancia con respecto a su vecino más próximo
    result = distances[:, 1]
    dobs = np.sum(result) / len(puntos)
    rn = dobs / (0.5 * np.sqrt((area / len(puntos))))
    ind.__setattr__('fitness', rn)


def seleccionar(individuos, k):
    ind_ordenados = sorted(individuos, key=attrgetter('fitness'), reverse=True)
    sum_fits = sum(getattr(ind, 'fitness') for ind in individuos)
    chosen = []
    for i in range(k):
        u = random.random() * sum_fits
        sum_ = 0
        for ind in ind_ordenados:
            sum_ += getattr(ind, 'fitness')
            if sum_ > u:
                chosen.append(ind)
                break

    return chosen


def cruza(ind1, ind2):
    tam = min(len(ind1), len(ind2))
    pto_cruza = random.randint(1, tam-1)
    ind1[pto_cruza:], ind2[pto_cruza:] = ind2[pto_cruza:], ind1[pto_cruza:]

    return ind1, ind2

barrios_caba = gpd.read_file(r'/home/cecilia/Dropbox/Tesis/Puntos Verdes/barrios.csv')
barrios_caba.crs = {'init': 'EPSG:4326'}
barrios_caba['area'] = np.around(barrios_caba['area'].astype(float) / 10 ** 6, decimals=3)  # paso a km2
area = barrios_caba['area'].sum()

nro_coordenadas = 50
nro_individuos = 10
nro_generaciones = 20
probabilidad_cruza = 0.2
probabilidad_mutacion = 0.2

print('Creando población inicial de {} individuos. \nCada individuo posee {} coordenadas'.format(nro_individuos, nro_coordenadas))
poblacion = Poblacion()
for i in range(0, nro_individuos):
    individuo = crear_individuo(nro_coordenadas, barrios_caba)
    poblacion.individuos.append(individuo)
print('\n ------------------- \n Población de 10 individuos: \n')


for i in range(0, nro_generaciones):
    # Calculo para cada individuo el NNI
    for ind in poblacion.individuos:
        calcular_nni(ind)

    #Selecciono
    offspring = seleccionar(poblacion.individuos, len(poblacion.individuos))

    # Hago variar a los individuos (cruza y mutación)
    #clono a los individuos
    offspring_clonado = poblacion.individuos.copy()

    for i in range(1, len(offspring_clonado), 2): # 1, 3, 5
        if random.random < probabilidad_cruza:
            pass
            #offspring_clonado[i-1], offspring_clonado[i] =


