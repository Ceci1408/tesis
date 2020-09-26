from deap import base, creator, tools
import random
from shapely.geometry import Point
from shapely.ops import unary_union, nearest_points
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import geopandas as gpd
from scoop import futures
from collections import Sequence
from itertools import repeat, count
from conexion_bd import conexion_db

'''
    Carga y variables del mapa
'''
barrios_caba_gdf = gpd.read_file('/home/cecilia/Documentos/Ceci/Tesis_2019/Puntos Verdes/poligono_disuelto_caba.csv')
barrios_caba_gdf = barrios_caba_gdf.set_geometry('geometry')
barrios_caba_gdf.crs = {'init': 'EPSG:4326'}
barrios_caba_gdf['area_km'] = np.around(barrios_caba_gdf['area_total'].astype(float) * (10 ** -6), decimals=3)  # paso a km2

area = barrios_caba_gdf['area_km'].iloc[0]

poligonos_barrios = [p for p in barrios_caba_gdf.geometry]
poligono_caba = unary_union(poligonos_barrios)

'''
    Variables para el algoritmo.
'''

Q_IND = 25# 200
Q_COORD = 50
MUT_PROB = 0.3
CROSS_PROB = 0.9
MUT_ALELO_PROB = 0.3
TIPO_SELECCION = "TORNEO"
TORNEO_TAM = 3
MU_X = 0
SIGMA_X = 0.01
MU_Y = 0
SIGMA_Y = 0.01
GENERACION_PARADA = 3500


'''
Según /home/cecilia/Dropbox/Tesis/Lectura de soporte/information-10-00390.pdf
concluye que para poblaciones pequeñas, lo mejor es Dynamic Increasing of Low Mutation/Decreasing of High Crossover (ILM/DHC)

'''

def evalOneMax(individual):
    df = pd.Series(individual)
    coord_radianes = df.apply(lambda c: (np.radians(c.x), np.radians(c.y)))
    nbrs = NearestNeighbors(n_neighbors=2, metric='haversine').fit(coord_radianes.tolist())
    distances, indices = nbrs.kneighbors(coord_radianes.tolist())
    distances = distances * 6371000 / 1000
    result = distances[:, 1]
    dobs = np.sum(result) / Q_COORD
    dest = 0.5 / np.sqrt(Q_COORD / area)

    rn = dobs/dest

    return rn,


def check_limites():
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if not validar_coordenada(child[i]):
                        puntos = nearest_points(child[i], poligono_caba)
                        child[i] = puntos[1]
            return offspring
        return wrapper
    return decorator


def validar_coordenada(punto):
    return punto.within(poligono_caba)


def crear_coordenadas():
    x_min, y_min, x_max, y_max = barrios_caba_gdf.total_bounds

    while True:
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
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


def mutGaussian_custom(individual, mu_x, sigma_x, mu_y, sigma_y, indpb):
    size = len(individual)

    if not isinstance(mu_x, Sequence):
        mu_x = repeat(mu_x, size)
    elif len(mu_x) < size:
        raise IndexError("mu must be at least the size of individual: %d < %d" % (len(mu_x), size))
    if not isinstance(sigma_x, Sequence):
        sigma_x = repeat(sigma_x, size)
    elif len(sigma_x) < size:
        raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma_x), size))

    if not isinstance(mu_y, Sequence):
        mu_y = repeat(mu_y, size)
    elif len(mu_y) < size:
        raise IndexError("mu must be at least the size of individual: %d < %d" % (len(mu_y), size))
    if not isinstance(sigma_y, Sequence):
        sigma_y = repeat(sigma_y, size)
    elif len(sigma_y) < size:
        raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma_y), size))

    for i, mx, sx, my, sy in zip(range(size), mu_x, sigma_x, mu_y, sigma_y):
        r = random.random()
        if r < indpb:
            r_1 = random.gauss(mx, sx)
            r_2 = random.gauss(my, sy)
            temp_x = individual[i].x + r_1
            temp_y = individual[i].y + r_2
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
toolbox.register("evaluate", evalOneMax)
toolbox.register("map", futures.map)

toolbox.decorate("mate", check_limites())
toolbox.decorate("mutate", check_limites())


def main(conn):

    random.seed(128)

    config_id = conexion_db.insert_config(conn, Q_IND, Q_COORD, MUT_PROB, CROSS_PROB, MUT_ALELO_PROB, TIPO_SELECCION, TORNEO_TAM,
                              MU_X, MU_Y, SIGMA_X, SIGMA_Y, GENERACION_PARADA)

    pop = toolbox.poblacion(n=Q_IND)

    fitnesses = list(map(toolbox.evaluate, pop))

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    fits = [ind.fitness.values[0] for ind in pop]

    g = 0
    print('Generacion {}'.format(g))

    generacion_id = conexion_db.insert_generacion(conn, g, config_id)

    for i, ind in enumerate(pop):
        individuo_id = conexion_db.insert_individuo(conn, i, generacion_id, ind.fitness.values[0])

        list_puntos = zip(count(start=1), repeat(individuo_id), [x.wkt for x in ind])

        conexion_db.insert_punto(conn, list(list_puntos))


    # Begin the evolution
    while max(fits) < 2.15 and g < GENERACION_PARADA:
        # A new generation
        g = g + 1
        print('Generacion {}'.format(g))
        generacion_id = conexion_db.insert_generacion(conn, g, config_id)

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
                toolbox.mutate(mutant, MU_X, SIGMA_X, MU_Y, SIGMA_Y)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        for i, ind in enumerate(pop):
            individuo_id = conexion_db.insert_individuo(conn, i, generacion_id, ind.fitness.values[0])

            list_puntos = zip(count(start=1), repeat(individuo_id), [x.wkt for x in ind])

            conexion_db.insert_punto(conn, list(list_puntos))

    conexion_bd.close_connection(conn, config_id)


if __name__ == '__main__':
    conexion_bd = conexion_db.connect()

    main(conexion_bd)
