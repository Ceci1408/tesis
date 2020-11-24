from deap import base, creator, tools
import random
from shapely.geometry import Point
from shapely.ops import unary_union
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import geopandas as gpd
from scoop import futures
from collections import Sequence
from itertools import repeat, count
from tesis.conexion_bd import base_de_datos
from tesis.visualizacion import visualizacion_resultados as vr
import os

poligono_caba=None


def funcion_objetivo(individual, Q_COORD, area):
    df = pd.Series(individual)
    coord_radianes = df.apply(lambda c: (np.radians(c.y), np.radians(c.x)))
    nbrs = NearestNeighbors(n_neighbors=2, metric='haversine').fit(coord_radianes.tolist())
    distances, indices = nbrs.kneighbors(coord_radianes.tolist())
    distances = distances * 6371000 / 1000
    result = distances[:, 1]
    dobs = np.sum(result) / Q_COORD
    dest = 0.5 / np.sqrt(Q_COORD / area)

    rn = dobs / dest

    return rn,


def validar_coordenada(punto, poligono_caba):
    return punto.within(poligono_caba)


def crear_coordenadas(mapa='mapa', poligono='poligono'):
    x_min, y_min, x_max, y_max = mapa.total_bounds

    while True:
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        point = Point(x, y)
        if validar_coordenada(point, poligono):
            break
    return point


def validar_individuo(individual):
    global poligono_caba
    return all(validar_coordenada(punto, poligono_caba) for punto in individual)


'''
    Funciones customizadas de Crossover y Mutación
'''


def cxBlend_custom(individuo_1, individuo_2, alpha):
    for i, (ind1, ind2) in enumerate(zip(individuo_1, individuo_2)):
        gamma = (1. + 2. * alpha) * random.random() - alpha

        coord_1_x = (1. - gamma) * ind1.x + gamma * ind2.x
        coord_2_x = gamma * ind1.x + (1. - gamma) * ind2.x

        coord_1_y = (1. - gamma) * ind1.y + gamma * ind2.y
        coord_2_y = gamma * ind1.y + (1. - gamma) * ind2.y

        punto_nuevo_ind_1 = Point(coord_1_x, coord_1_y)
        punto_nuevo_ind_2 = Point(coord_2_x, coord_2_y)

        individuo_1[i] = punto_nuevo_ind_1
        individuo_2[i] = punto_nuevo_ind_2

    return individuo_1, individuo_2


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

            punto_mutado = Point(temp_x, temp_y)
            individual[i] = punto_mutado

    return individual,


def ejecucion_ga(conn, ga_params):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individuo", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("coordenada", crear_coordenadas, mapa=ga_params.get('barrios_caba_gdf'),
                     poligono=ga_params.get('poligonos_caba'))
    toolbox.register("individuo", tools.initRepeat, creator.Individuo, toolbox.coordenada,
                     n=ga_params.get('q_coord'))
    toolbox.register("poblacion", tools.initRepeat, list, toolbox.individuo)
    toolbox.register("mate", cxBlend_custom)
    toolbox.register("mutate", mutGaussian_custom, indpb=ga_params.get('mut_prob_alelo'))

    if ga_params.get('tipo_seleccion') == 1:
        toolbox.register("select", tools.selRoulette)
    else:
        toolbox.register("select", tools.selTournament, tournsize=ga_params.get('torneo_tam'))

    toolbox.register("evaluate", funcion_objetivo, Q_COORD = ga_params.get('q_coord'), area=ga_params.get('area'))
    toolbox.register("map", futures.map)

    toolbox.decorate("evaluate", tools.DeltaPenalty(validar_individuo, 0))

    print('Ejecutando experimento {}'.format(ga_params.get('tipo_seleccion')))

    '''
    config_id = base_de_datos.insert_config(conn, ga_params.get('q_ind'), ga_params.get('q_coord'),
                                            ga_params.get('mut_prob'), ga_params.get('cross_prob'),
                                            ga_params.get('mut_prob_alelo'), ga_params.get('tipo_seleccion'),
                                            ga_params.get('torneo_tam'), ga_params.get('mu_x'), ga_params.get('mu_y'),
                                            ga_params.get('sigma_x'), ga_params.get('sigma_y'),
                                            ga_params.get('generacion_parada'), ga_params.get('alpha'))
    '''
    pop = toolbox.poblacion(n=ga_params.get('q_ind'))

    fitnesses = list(map(toolbox.evaluate, pop))

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    fits = [ind.fitness.values[0] for ind in pop]

    g = 0
    print('Generacion {} - Max Fit: {} Avg Fit {}'.format(g, max(fits), np.average(fits)))

    # generacion_id = base_de_datos.insert_generacion(conn, g, config_id)

    '''
    for i, ind in enumerate(pop):
        individuo_id = base_de_datos.insert_individuo(conn, i, generacion_id, ind.fitness.values[0])
        list_puntos = zip(count(start=1), repeat(individuo_id), [x.wkt for x in ind])
        base_de_datos.insert_punto(conn, list(list_puntos))
    '''
    # Begin the evolution
    while max(fits) < 2.15 and g < 300:#ga_params.get('generacion_parada'):
        # A new generation
        g = g + 1
        print('Generacion {} - Max Fit: {} Avg Fit {}'.format(g, max(fits), np.average(fits)))

        #generacion_id = base_de_datos.insert_generacion(conn, g, config_id)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < ga_params.get('cross_prob'):
                toolbox.mate(child1, child2, ga_params.get('alpha'))  # random.random())
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < ga_params.get('mut_prob'):
                toolbox.mutate(mutant, ga_params.get('mu_x'), ga_params.get('sigma_x'),
                               ga_params.get('mu_y'), ga_params.get('sigma_y'))
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        '''
        for i, ind in enumerate(pop):
            individuo_id = base_de_datos.insert_individuo(conn, i, generacion_id, ind.fitness.values[0])

            list_puntos = zip(count(start=1), repeat(individuo_id), [x.wkt for x in ind])

            base_de_datos.insert_punto(conn, list(list_puntos))

    base_de_datos.update_config(conn, config_id)
    '''
    '''
        Termina la ejecución y limpieza de variables, clases y funciones
    '''

    toolbox.unregister('select')
    toolbox.unregister('poblacion')
    toolbox.unregister('individuo')

    del creator.Individuo
    del creator.FitnessMax


def main(conn):
    """
        Carga de variables
    """
    #os.chdir('..')
    archivo_parametros = os.path.join(os.getcwd(), 'archivo_parametros.csv')

    parametros_df = pd.read_csv(archivo_parametros, decimal=',')

    for i in range(len(parametros_df)):
        fila = parametros_df.iloc[i]
        ubicacion_mapa = fila.loc['Ubicacion_mapa']

        '''
            Preparación del mapa
        '''
        barrios_caba_gdf = gpd.read_file(os.path.join(os.getcwd(), ubicacion_mapa))
        barrios_caba_gdf = barrios_caba_gdf.set_geometry('geometry')
        barrios_caba_gdf.crs = {'init': 'EPSG:4326'}
        barrios_caba_gdf['area_km'] = np.around(barrios_caba_gdf['area_total'].astype(float) * (10 ** -6),
                                                decimals=3)  # paso a km2
        area = barrios_caba_gdf['area_km'].iloc[0]
        poligonos_barrios = [p for p in barrios_caba_gdf.geometry]
        global poligono_caba
        poligono_caba = unary_union(poligonos_barrios)

        random.seed(128)

        '''
            Preparación del algoritmo:
                Tipo de selección 1 = Ruleta
                Tipo de seleccion 2 = Torneo
        '''

        #armo dict con los parámetros

        param_ga = dict(
            barrios_caba_gdf=barrios_caba_gdf,
            area=area,
            poligonos_caba=poligono_caba,
            q_ind=int(fila.loc['q_ind']),
            q_coord=int(fila.loc['q_coord']),
            mut_prob=float(fila.loc['mut_prob']),
            cross_prob=float(fila.loc['mut_cruce']),
            mut_prob_alelo=float(fila.loc['mut_prob_alelo']),
            tipo_seleccion=int(fila.loc['tipo_seleccion']),
            torneo_tam=int(fila.loc['torneo_tam']),
            mu_x =float(fila.loc['mu_x']),
            sigma_x=float(fila.loc['sigma_x']),
            mu_y=float(fila.loc['mu_y']),
            sigma_y=float(fila.loc['sigma_y']),
            generacion_parada=int(fila.loc['gen_parada']),
            alpha=float(fila.loc['alpha'])
        )

        ejecucion_ga(conn, param_ga)


if __name__ == '__main__':
    conexion_bd = base_de_datos.connect()
    #main(conexion_bd)
    vr.generar_visualizaciones(conexion_bd)

    base_de_datos.close_connection(conexion_bd)

