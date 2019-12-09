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


def validar_coordenada(punto):
    return barrios_caba.contains(punto).any()


def crear_coordenadas(q_coordenadas):
    puntos = []
    for i in range(0, q_coordenadas):
        x_min, y_min, x_max, y_max = barrios_caba.total_bounds
        while True:
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            point = Point(x, y)
            if validar_coordenada(point):
                puntos.append(point)
                break
    return puntos


def calcular_nni(coordenadas):
    coord_radianes = copy.deepcopy(coordenadas)
    coord_radianes = coord_radianes.apply(lambda c: (np.radians(c.x), np.radians(c.y)))
    nbrs = NearestNeighbors(n_neighbors=2, metric='haversine').fit(coord_radianes.tolist())
    distances, indices = nbrs.kneighbors(coord_radianes.tolist())
    distances = distances * 6371000 / 1000
    result = distances[:, 1]
    dobs = np.sum(result) / len(coord_radianes.tolist())
    rn = dobs / (0.5 * np.sqrt((area / len(coord_radianes.tolist()))))
    return rn


def crear_individuos(q_individuos, q_coordenadas):
    individuos = []
    for i in range(0, q_individuos):
        individuos.append(crear_coordenadas(q_coordenadas))
    return individuos


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
        self.k_seleccion = 2

    def inicio_ga(self, tipo_modelo):
        self.poblacion = pd.DataFrame(crear_individuos(10, 50))
        self.poblacion['nni'] = self.poblacion.apply(calcular_nni, axis=1)

        if tipo_modelo == 'g':
            copia = self.seleccion_generacional()

        else:
            copia_poblacion = copy.deepcopy(self.poblacion)
            for gen in range(0, self.nro_generaciones):
                # estacionario con, por ahora, selección ruleta
                offspring = self.seleccion_ruleta(self.k_seleccion)

                for i in range(1, len(offspring), 2):
                    if np.random.random() < self.prob_cruza:
                        offspring[i-1], offspring[i] = self.cruzamiento(offspring[i-1], offspring[i])

                for i in range(len(offspring)):
                    if np.random.random() < self.prob_mut:
                        offspring[i] = self.mutacion(offspring[i], self.mu, self.sigma, self.prob_mut_alelo)

                # paso al offspring a df
                offspring_df = pd.DataFrame(offspring)
                offspring_df['nni'] = offspring_df.apply(calcular_nni, axis=1)

                # elimino al peor de la población inicial y lo reemplazo por el nuevo mejor
                print('Peor \n'+str(copia_poblacion['nni'].idxmin()))
                copia_poblacion = copia_poblacion.drop(copia_poblacion['nni'].idxmin())
                copia_poblacion = copia_poblacion.append(offspring_df.iloc[offspring_df['nni'].idxmax()],
                                                         ignore_index=True)
                print(copia_poblacion)
            return copia_poblacion

    # modelo generacional.
    def seleccion_generacional(self):
        copy_ind = copy.deepcopy(self.poblacion.individuos)
        return copy_ind

    # modelo estacionario
    def seleccion_ruleta(self, k):
        copy_pob = copy.deepcopy(self.poblacion)
        sum_fit = copy_pob['nni'].sum()
        copy_pob['cum_sum'] = copy_pob['nni'].cumsum()
        chosen = []
        for i in range(k):
            u = np.random.random() * sum_fit
            s = copy_pob[copy_pob.cum_sum > u].head(1).iloc[:, :50]
            chosen.append(s.values[0])
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
        tam = min(len(ind1), len(ind2))
        pto_cruce = np.random.randint(1, tam-1)

        ind1_copia = copy.deepcopy(ind1)

        ind1[pto_cruce:], ind2[pto_cruce:] = ind2[pto_cruce:], ind1_copia[pto_cruce:]

        del ind1_copia
        return ind1, ind2

    def mutacion(self, ind1, mu, sigma, prob):
        '''Prob: probabilidad de que cada coordenada sea mutada.
        Se aplica la mutación con distribución de Gauss de media mu y una desv.
        estándar de sigma como input del individuo'''

        # repito mu tantas coordenadas tenga el individuo
        mu = repeat(mu, len(ind1))
        # repito sigma tantas coordenadas tenga el individuo
        sigma = repeat(sigma, len(ind1))

        for i, m, s in zip(range(len(ind1)), mu, sigma):
            if random.random() < prob:
                while True:
                    punto_mutado = Point(ind1[i].x + random.gauss(m, s),
                                         ind1[i].y + random.gauss(m, s))

                    if validar_coordenada(punto_mutado):
                        ind1 = np.delete(ind1, [i])
                        ind1 = np.insert(ind1, [i], [punto_mutado])
                        break
        return ind1



ga = AlgoritmoGenetico()
ga.inicio_ga('e')