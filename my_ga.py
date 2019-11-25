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



nro_coordenadas = 50
nro_individuos = 10
nro_generaciones = 70
probabilidad_cruza = 0.2
probabilidad_mutacion = 0.1
probabilidad_coordenada_mutacion = 0.2
mu = 0
sigma = 1


class Individuo:
    def __init__(self, nro_coordenadas):
        self.coordenadas = []
        self.fitness = 0
        self.nro_coordenadas = nro_coordenadas

    def validar_coordenada(self, punto):
        return barrios_caba.contains(punto).any()

    def crear_coordenadas(self):
        for i in range(0, nro_coordenadas):
            x_min, y_min, x_max, y_max = barrios_caba.total_bounds
            while True:
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
                point = Point(x, y)
                if self.validar_coordenada(point):
                    self.coordenadas.append(point)
                    break

    def listar_coordenadas(self):
        for c in self.coordenadas:
            print("Coordenada --> Longitud: "+str(c.x)+" Latitud: "+str(c.y))

    def calcular_nni(self):
        puntos = []

        for p in self.coordenadas:
            puntos.append([np.radians(p.x), np.radians(p.y)])
        nbrs = NearestNeighbors(n_neighbors=2, metric='haversine').fit(puntos)
        distances, indices = nbrs.kneighbors(puntos)
        distances = distances * 6371000 / 1000
        # este array contiene para c/ punto, la distancia con respecto a su vecino más próximo
        result = distances[:, 1]
        dobs = np.sum(result) / len(puntos)
        rn = dobs / (0.5 * np.sqrt((area / len(puntos))))
        return rn

    def graficar_puntos(self):
        fig, ax = plt.subplots(figsize=(15, 15))
        barrios_caba.plot(ax=ax)
        ax.plot()
        for coord in self.coordenadas:
            x, y = coord.x, coord.y
            ax.plot(x, y, 'x', color='green')
        plt.show()


class Poblacion:
    def __init__(self, nro_individuos):
        self.individuos = []
        self.nro_individuos = nro_individuos

    def crear_individuos(self):
        for i in range(0, nro_individuos):
            individuo = Individuo(50)
            individuo.crear_coordenadas()
            #if not individuo.validar_coordenada():
            #    individuo.penalizar()
            self.individuos.append(individuo)

    def graficar_fitness(self):
        fig, axes = plt.subplots(len(self.individuos), 1, sharex='col', sharey='none', squeeze=False)
        print(axes)

class Algoritmo:
    def __init__(self, nro_gen, prob_cx, prob_mt, prob_mt_gen, mu, sigma):
        nro_gen = nro_gen
        prob_cx = prob_cx
        prob_mt = prob_mt
        prob_mt_gen = prob_mt_gen
        mu = mu
        sigma = sigma


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
    tam = min(len(ind1.coordenadas), len(ind2.coordenadas))
    pto_cruza = random.randint(1, tam-1)
    ind1.coordenadas[pto_cruza:], ind2.coordenadas[pto_cruza:] = ind2.coordenadas[pto_cruza:], ind1.coordenadas[pto_cruza:]

    return ind1, ind2

#http://sedici.unlp.edu.ar/bitstream/handle/10915/22639/Documento_completo.%20pdf?sequence=1
def mutacion(ind1, mu, sigma, prob):
    '''Prob: probabilidad de que cada coordenada sea mutada.
    Se aplica la mutación con distribución de Gauss de media mu y una desv.
    estándar de sigma como input del individuo'''
    temp_ind = ind1

    if not isinstance(mu, Sequence):
        mu = repeat(mu, len(temp_ind.coordenadas))
    elif len(mu) < len(temp_ind.coordenadas):
        raise IndexError('mu debe tener al menos, la longitud de la lista de coordenadas')

    if not isinstance(sigma, Sequence):
        sigma = repeat(sigma, len(temp_ind.coordenadas))
    elif len(sigma) < len(temp_ind.coordenadas):
        raise IndexError('Sigma debe tener al menos, la longitud de la lista de coordenadas')

    for i, m, s in zip(range(len(temp_ind.coordenadas)), mu, sigma):
        #while True:
        if random.random() < prob:
            while True:
                punto_mutado = Point(temp_ind.coordenadas[i].x+random.gauss(m, s),
                                     temp_ind.coordenadas[i].y+random.gauss(m, s))
                if temp_ind.validar_coordenada(punto_mutado):
                    del ind1.coordenadas[i]
                    ind1.coordenadas.insert(i, punto_mutado)
                    break
    return ind1


def algoritmo_genetico(poblacion, nro_generaciones):
    #poblacion_clonada = poblacion

    for gen in range(0, nro_generaciones):
        print('GENERACION NRO '+str(gen))
        # Calculo para cada individuo el NNI
        for ind in poblacion.individuos:
            ind.fitness = ind.calcular_nni()
        #Selecciono
        offspring = seleccionar(poblacion.individuos, len(poblacion.individuos))

        # Hago variar a los individuos (cruza y mutación)
        #clono a los individuos
        #offspring_clonado = offspring.copy()

        for i in range(1, len(offspring), 2): # 1, 3, 5
            if random.random() < probabilidad_cruza:
                offspring[i-1], offspring[i] = cruza(offspring[i-1], offspring[i])
                #offspring[i-1].fitness, offspring[i].fitness = None, None

        for i in range(len(offspring)):
            if random.random() < probabilidad_mutacion:
                offspring[i] = mutacion(offspring[i], mu, sigma, probabilidad_coordenada_mutacion)
                #offspring[i].fitness = None

        # Calculo el fitness para todos porque quedaron los nuevos sin fitness y penalizo
        #poblacion_clonada = Poblacion()
        #poblacion_clonada.individuos = offspring_clonado

        for ind in poblacion.individuos:
            ind.fitness = ind.calcular_nni()
            print(ind.fitness)
    return poblacion


print('Creando población inicial de {} individuos. \nCada individuo posee {} coordenadas'.format(nro_individuos, nro_coordenadas))
poblacion = Poblacion(10)
poblacion.crear_individuos()
print('\n ------------------- \n Población de 10 individuos: \n')

pob_final = algoritmo_genetico(poblacion, nro_generaciones)

row = []
for idx, ind in enumerate(pob_final.individuos):
    row.append([idx, ind.fitness])

df = pd.DataFrame.from_records(row, columns=['ind', 'fitness'])
df.plot(kind='bar', x='ind', y='fitness')
plt.show()
# data = []
# for idx, i in enumerate(pob_final.individuos):
#     data.append([idx, i.fitness])
# df = pd.DataFrame.from_records(data, columns=['ind', 'fitness'])
# df.plot(kind='bar', x='ind', y='fitness')
# plt.show()
#for ind in pob_final.individuos:
#    ind.graficar_puntos()

#pob_final.graficar_fitness()

# TODO crear una función de penalizacion externa! penalizar soluciones no factibles
# 1) que esté dentro de los límites
# 2) que el rn esté dentro del rango 0 - 2.15