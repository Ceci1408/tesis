from deap import base, creator, tools
import random
from shapely.geometry import Point
from shapely.ops import unary_union, nearest_points
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import geopandas as gpd
import pickle
from scoop import futures
from collections import Sequence
from itertools import repeat
from datetime import datetime
'''
    Carga y variables del mapa
'''
barrios_caba = gpd.read_file('/home/cecilia/Dropbox/Tesis/Puntos Verdes/barrios.csv')
#barrios_caba = gpd.read_file('barrios.csv')
barrios_caba.crs = {'init': 'EPSG:4326'}
barrios_caba['area_km'] = np.around(barrios_caba['area'].astype(float) * (10 ** -6), decimals=3)  # paso a km2
area = barrios_caba['area_km'].sum()

poligonos = [p for p in barrios_caba.geometry]
todo_junto = unary_union(poligonos)

'''
    Variables para el algoritmo.
'''
Q_IND = 100# 200
Q_COORD = 100
MUT_PROB = 0.01
CROSS_PROB = 0.5
MUT_ALELO_PROB = 0.01
TORNEO_TAM = 3
MU = 0
SIGMA = 1
'''
Q_IND = 25  # 200
Q_COORD = 100
MUT_PROB = 0.3
CROSS_PROB = 0.9
MUT_ALELO_PROB = 0.3
TORNEO_TAM = 3
'''
def evalOneMax(individual):
    df = pd.Series(individual)
    coord_radianes = df.apply(lambda c: (np.radians(c.x), np.radians(c.y)))
    nbrs = NearestNeighbors(n_neighbors=2, metric='haversine').fit(coord_radianes.tolist())
    #nbrs = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(coord_radianes.tolist())
    distances, indices = nbrs.kneighbors(coord_radianes.tolist())
    distances = distances * 6371000 / 1000
    result = distances[:, 1]
    dobs = np.sum(result) / Q_COORD
    rn = 2 * dobs * np.sqrt(Q_COORD / area)

    return rn,


def check_limites():
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if not validar_coordenada(child[i]):
                        puntos = nearest_points(child[i], todo_junto)
                        child[i] = puntos[1]
            return offspring
        return wrapper
    return decorator


def validar_coordenada(punto):
    return punto.within(todo_junto)


def crear_coordenadas():
    x_min, y_min, x_max, y_max = barrios_caba.total_bounds
    while True:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        point = Point(x, y)
        if validar_coordenada(point):
            break
    return point


'''
    Funciones customizadas de Crossover y Mutación
'''


def cxBlend_custom(ind1, ind2, alpha):
    for i, (x1, x2) in enumerate(zip(ind1, ind2)):
        gamma = (1. + 2. * alpha) * random.random() - alpha
        ind1_temp_x = (1. - gamma) * x1.x + gamma * x2.x
        ind2_temp_x = gamma * x1.x + (1. - gamma) * x2.x

        ind1_temp_y = (1. - gamma) * x1.y + gamma * x2.y
        ind2_temp_y = gamma * x1.y + (1. - gamma) * x2.y

        ind1[i] = Point(ind1_temp_x, ind1_temp_y)
        ind2[i] = Point(ind2_temp_x, ind2_temp_y)
    return ind1, ind2


def mutGaussian_custom(individual, mu, sigma, indpb):
    size = len(individual)
    if not isinstance(mu, Sequence):
        mu = repeat(mu, size)
    elif len(mu) < size:
        raise IndexError("mu must be at least the size of individual: %d < %d" % (len(mu), size))
    if not isinstance(sigma, Sequence):
        sigma = repeat(sigma, size)
    elif len(sigma) < size:
        raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma), size))

    for i, m, s in zip(range(size), mu, sigma):
        if random.random() < indpb:
            temp_x = individual[i].x + random.gauss(m, s)
            temp_y = individual[i].y + random.gauss(m, s)
            individual[i] = Point(temp_x, temp_y)

    return individual,


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individuo", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("coordenada", crear_coordenadas)
toolbox.register("individuo", tools.initRepeat, creator.Individuo, toolbox.coordenada, n=Q_COORD)
toolbox.register("poblacion", tools.initRepeat, list, toolbox.individuo)
toolbox.register("mate", cxBlend_custom)
toolbox.register("mutate", mutGaussian_custom, indpb=MUT_ALELO_PROB)
toolbox.register("select", tools.selTournament, tournsize=TORNEO_TAM)
#toolbox.register("select",selElitistAndTournament,frac_elitist=0.1, tournsize=3)
toolbox.register("evaluate", evalOneMax)
toolbox.register("map", futures.map)

toolbox.decorate("mate", check_limites())
toolbox.decorate("mutate", check_limites())

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

logbook = tools.Logbook()


def main():
    random.seed(128)
    pop = toolbox.poblacion(n=Q_IND)

    fitnesses = list(map(toolbox.evaluate, pop))

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    fits = [ind.fitness.values[0] for ind in pop]

    g = 0

    # Begin the evolution
    while max(fits) < 2.15:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSS_PROB:
                toolbox.mate(child1, child2, random.random())
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUT_PROB:
                toolbox.mutate(mutant, MU, SIGMA)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        record = stats.compile(pop)
        logbook.record(gen_nro=g, **record)
        print(record)

        i = 0
        dataframe = pd.DataFrame()
        for ind in pop:
            puntos_df = pd.DataFrame(ind)
            puntos_df['ind'] = i
            puntos_df['gen'] = g
            dataframe = dataframe.append(puntos_df, ignore_index=True)
            i += 1
        dataframe.to_csv('puntos_torneos_{}.csv'.format(datetime.strftime(datetime.now(), '%Y%m%d')), mode='a', header=False, index=False)

    pickle_file = 'pickle_file_stats_torneo_{}'.format(datetime.strftime(datetime.now(), '%Y%m%d'))
    outfile = open(pickle_file, 'wb')
    pickle.dump(logbook, outfile)
    outfile.close()


if __name__ == '__main__':
    main()