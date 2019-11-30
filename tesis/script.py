import geopandas as gpd
import copy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
from shapely.geometry import Point
import random
from operator import attrgetter
from collections import Sequence
from itertools import repeat

barrios_caba = gpd.read_file(r'/home/cecilia/Dropbox/Tesis/Puntos Verdes/barrios.csv')
barrios_caba.crs = {'init': 'EPSG:4326'}
barrios_caba['area'] = np.around(barrios_caba['area'].astype(float) / 10 ** 6, decimals=3)  # paso a km2
area = barrios_caba['area'].sum()


class Individuo:
    # un arreglo de 1 eje, con forma (1, 50) una fila, 50 columnas
    def __init__(self):
        self.coordenadas = None
        self.fitness = 0

    def validar_coordenada(self, punto):
        return barrios_caba.contains(punto).any()

    def crear_coordenadas(self, q_coordenadas):
        puntos = []
        for i in range(0, q_coordenadas):
            x_min, y_min, x_max, y_max = barrios_caba.total_bounds
            while True:
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
                point = Point(x, y)
                if self.validar_coordenada(point):
                    puntos.append(point)
                    break
        self.coordenadas = gpd.GeoSeries(puntos)

    def calcular_nni(self):
        coord_radianes = copy.deepcopy(self.coordenadas)
        coord_radianes = coord_radianes.apply(lambda c: (np.radians(c.x), np.radians(c.y)))
        nbrs = NearestNeighbors(n_neighbors=2, metric='haversine').fit(coord_radianes.tolist())
        distances, indices = nbrs.kneighbors(coord_radianes.tolist())
        distances = distances * 6371000 / 1000
        result = distances[:, 1]
        dobs = np.sum(result) / len(coord_radianes.tolist())
        rn = dobs / (0.5 * np.sqrt((area / len(coord_radianes.tolist()))))
        self.fitness = rn


class Poblacion:
    def __init__(self):
        self.individuos = None

    def crear_individuos(self, q_individuos, q_coordenadas):
        individuos = []
        for i in range(0, q_individuos):
            ind = Individuo()
            ind.crear_coordenadas(q_coordenadas)
            ind.calcular_nni()
            individuos.append(ind)

        self.individuos = gpd.GeoSeries(individuos)


class AlgoritmoGenetico:
    def __init__(self):
        self.q_individuos = 10
        self.q_coordenadas = 50
        self.nro_generaciones = 20
        self.prob_cruza = 0.2
        self.prob_mut = 0.1
        self.prob_mut_alelo = 0.2
        self.mu = 0
        self.sigma = 1
        self.poblacion = None

    def inicio_ga(self):
        self.poblacion = Poblacion(self.q_individuos, self.q_coordenadas)
        self.poblacion.crear_individuos()

    def seleccion_ruleta(self):
        for ind in self.poblacion.individuos:
            print(type(ind))


pob = Poblacion()
pob.crear_individuos(10, 50)
for ind in pob.individuos:
    print('Individuo 1')
    print(ind)