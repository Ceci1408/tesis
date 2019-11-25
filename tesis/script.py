import geopandas as gpd
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
    def __init__(self, q_coordenadas):
        self.coordenadas = np.empty(shape=(1, q_coordenadas), dtype=Point)
        self.fitness = 0

    def validar_coordenada(self, punto):
        return barrios_caba.contains(punto).any()

    def crear_coordenadas(self):
        for i in range(0, self.coordenadas.size):
            x_min, y_min, x_max, y_max = barrios_caba.total_bounds
            while True:
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
                point = Point(x, y)
                if self.validar_coordenada(point):
                    self.coordenadas[0, i] = point
                    break

    def calcular_nni(self):
        to_radians = lambda c: (np.radians(c.x), np.radians(c.y))
        vectorized_to_radians = np.vectorize(to_radians)
        puntos = vectorized_to_radians(self.coordenadas)
        x = puntos[0].tolist()[0]
        y = puntos[1].tolist()[0]
        puntos_radianes = np.array(list(zip(x, y)))

        nbrs = NearestNeighbors(n_neighbors=2, metric='haversine').fit(puntos_radianes)
        distances, indices = nbrs.kneighbors(puntos_radianes)
        distances = distances * 6371000 / 1000
        # este array contiene para c/ punto, la distancia con respecto a su vecino más próximo
        result = distances[:, 1]
        dobs = np.sum(result) / len(puntos_radianes)
        rn = dobs / (0.5 * np.sqrt((area / len(puntos_radianes))))
        self.fitness = rn


class Poblacion:
    def __init__(self, q_individuos, q_coordenadas):
        self.individuos = np.empty(shape=(q_individuos, 1), dtype=Individuo)
        self.q_coordenadas = q_coordenadas

    def crear_individuos(self):
        for i in range(0, self.individuos.size):
            individuo = Individuo(self.q_coordenadas)
            individuo.crear_coordenadas()
            individuo.calcular_nni()
            self.individuos[i, 0] = individuo


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

    def inicio_ga(self):
        poblacion = Poblacion(self.q_individuos, self.q_coordenadas)
        poblacion.crear_individuos()
        return poblacion


ga = AlgoritmoGenetico()
ga.inicio_ga()