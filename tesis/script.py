import geopandas as gpd
import copy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
from shapely.geometry import Point
import random
from operator import attrgetter
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
        self.individuos = pd.Series(individuos)


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
        self.tipo_modelo = 'e'
        self.k_seleccion = 0

    def inicio_ga(self, tipo_modelo):
        self.poblacion = Poblacion()
        self.poblacion.crear_individuos(self.q_individuos, self.q_coordenadas)

        if tipo_modelo == 'g':
            copia = self.seleccion_generacional()

        else:
            #for gen in range(0, self.nro_generaciones):
            self.k_seleccion = int(len(self.poblacion.individuos)/2)
            # estacionario con, por ahora, selección ruleta
            offspring = self.seleccion_ruleta(self.k_seleccion)

            for i in range(1, len(offspring), 2):
                if np.random.random() < self.prob_cruza:
                    offspring[i-1], offspring[i] = self.cruzamiento(offspring[i-1], offspring[i])

            for i in range(len(offspring)):
                if np.random.random() < self.prob_mut:
                    offspring[i] = self.mutacion(offspring[i], self.mu, self.sigma, self.prob_mut_alelo)

            for ind in offspring:
                ind.calcular_nni()
                print(ind.fitness)
            #return


    # modelo generacional.
    def seleccion_generacional(self):
        copy_ind = copy.deepcopy(self.poblacion.individuos)
        return copy_ind

    # modelo estacionario
    def seleccion_ruleta(self, k):
        copy_ind = copy.deepcopy(self.poblacion.individuos)
        sum_fit = sum(getattr(ind, 'fitness') for ind in copy_ind)

        chosen = []
        for i in range(k):
            u = np.random.random() * sum_fit
            sum_ = 0
            for ind in copy_ind:
                sum_ += getattr(ind, 'fitness')
                if sum_> u:
                    chosen.append(ind)
                    break
        return chosen

    # http: // sedici.unlp.edu.ar / bitstream / handle / 10915 / 4060 / III_ - _Algoritmos_evolutivos.pdf?sequence = 7 & isAllowed = y
    def selRandom(self, individuos, k):
        return [random.choice(individuos) for i in range(k)]

    def seleccion_torneo(self, tam_torneo):
        copy_ind = copy.deepcopy(self.poblacion.individuos)
        chosen = []
        for i in range(len(copy_ind)):
            aspirants = self.selRandom(copy_ind, tam_torneo)
            chosen.append(max(aspirants, key=attrgetter('fitness')))
        return chosen

    def cruzamiento(self, ind1, ind2):
        tam = min(len(ind1.coordenadas), len(ind2.coordenadas))
        pto_cruce = np.random.randint(1, tam-1)

        ind1_copia = copy.deepcopy(ind1)
        ind1.coordenadas[pto_cruce:], ind2.coordenadas[pto_cruce:] = ind2.coordenadas[pto_cruce:], \
                                                                     ind1_copia.coordenadas[pto_cruce:]

        del ind1_copia
        return ind1, ind2

    def mutacion(self, ind1, mu, sigma, prob):
        '''Prob: probabilidad de que cada coordenada sea mutada.
        Se aplica la mutación con distribución de Gauss de media mu y una desv.
        estándar de sigma como input del individuo'''

        # repito mu tantas coordenadas tenga el individuo
        mu = repeat(mu, len(ind1.coordenadas))
        # repito sigma tantas coordenadas tenga el individuo
        sigma = repeat(sigma, len(ind1.coordenadas))

        for i, m, s in zip(range(len(ind1.coordenadas)), mu, sigma):
            if random.random() < prob:
                while True:
                    punto_mutado = Point(ind1.coordenadas[i].x + random.gauss(m, s),
                                         ind1.coordenadas[i].y + random.gauss(m, s))

                    if ind1.validar_coordenada(punto_mutado):
                        ind1.coordenadas = ind1.coordenadas.drop([i])
                        ind1.coordenadas.loc[i] = punto_mutado
                        break
        return ind1

ga = AlgoritmoGenetico()
ga.inicio_ga('e')
