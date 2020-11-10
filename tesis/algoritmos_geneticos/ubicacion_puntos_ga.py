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
from tesis.conexion_bd import base_de_datos
import os


'''
    Carga de variables
'''
os.chdir('..')
archivo_parametros = os.path.join(os.getcwd(), 'parametros_iniciales.csv')

tipo_datos = {'Ubicacion_mapa': 'object',
               'q_ind': np.int8,
               'q_coord': np.int8,
               'mut_prob': np.float64,
               'mut_cruce': np.float64,
               'mut_prob_alelo': np.float64,
               'tipo_seleccion': np.int8,
               'torneo_tam': np.int8,
               'mu_x': np.float64,
               'mu_y': np.float64,
               'sigma_x': np.float64,
               'sigma_y': np.float64,
               'gen_parada': np.int8,
              'alpha': np.float64}

parametros_df = pd.read_csv(archivo_parametros, decimal=',')
conexion_bd = base_de_datos.connect()


for i in range(len(parametros_df)):
    fila = parametros_df.iloc[i]
    ubicacion_mapa = fila.loc['Ubicacion_mapa']
    q_ind = fila.loc['q_ind']
    q_coord = fila.loc['q_coord']
    mut_prob = fila.loc['mut_prob']
    mut_cruce = fila.loc['mut_cruce']
    mut_prob_alelo = fila.loc['mut_prob_alelo']
    tipo_seleccion = fila.loc['tipo_seleccion']
    torneo_tam = fila.loc['torneo_tam']
    mu_x = fila.loc['mu_x']
    mu_y = fila.loc['mu_y']
    sigma_x = fila.loc['sigma_x']
    sigma_y = fila.loc['sigma_y']
    gen_parada = fila.loc['gen_parada']
    alpha = fila.loc['alpha']

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
    poligono_caba = unary_union(poligonos_barrios)

    random.seed(128)

    '''
        Preparación del algoritmo:
            Tipo de selección 1 = Ruleta
            Tipo de seleccion 2 = Torneo
    '''

    Q_IND = int(q_ind)
    Q_COORD = int(q_coord)
    MUT_PROB = float(mut_prob)
    CROSS_PROB = float(mut_cruce)
    MUT_ALELO_PROB = float(mut_prob_alelo)

    TIPO_SELECCION = 'Ruleta' if tipo_seleccion == 1 else 'Torneo'
    TORNEO_TAM = int(torneo_tam)
    MU_X = float(mu_x)
    SIGMA_X = float(sigma_x)
    MU_Y = float(mu_y)
    SIGMA_Y = float(sigma_y)
    GENERACION_PARADA = int(gen_parada)
    ALPHA = float(alpha)

    def funcion_objetivo(individual):
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


    def validar_individuo(individual):
        return all(validar_coordenada(punto) for punto in individual)


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


    def ejecucion_ga(conn):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individuo", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        toolbox.register("coordenada", crear_coordenadas)
        toolbox.register("individuo", tools.initRepeat, creator.Individuo, toolbox.coordenada, n=Q_COORD)
        toolbox.register("poblacion", tools.initRepeat, list, toolbox.individuo)
        toolbox.register("mate", cxBlend_custom)
        toolbox.register("mutate", mutGaussian_custom, indpb=MUT_ALELO_PROB)

        if tipo_seleccion == 1:
            toolbox.register("select", tools.selRoulette)
        else:
            toolbox.register("select", tools.selTournament, tournsize=TORNEO_TAM)

        toolbox.register("evaluate", funcion_objetivo)
        toolbox.register("map", futures.map)

        toolbox.decorate("evaluate", tools.DeltaPenalty(validar_individuo, 0))

        print('Ejecutando {}'.format(TIPO_SELECCION))

        config_id = base_de_datos.insert_config(conn, Q_IND, Q_COORD, MUT_PROB, CROSS_PROB, MUT_ALELO_PROB,
                                                TIPO_SELECCION, TORNEO_TAM, MU_X, MU_Y, SIGMA_X, SIGMA_Y,
                                                GENERACION_PARADA, ALPHA)

        pop = toolbox.poblacion(n=Q_IND)

        fitnesses = list(map(toolbox.evaluate, pop))

        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        fits = [ind.fitness.values[0] for ind in pop]

        g = 0
        print('Generacion {} - Max Fit: {} Avg Fit {}'.format(g, max(fits), np.average(fits)))

        generacion_id = base_de_datos.insert_generacion(conn, g, config_id)

        for i, ind in enumerate(pop):
            individuo_id = base_de_datos.insert_individuo(conn, i, generacion_id, ind.fitness.values[0])
            list_puntos = zip(count(start=1), repeat(individuo_id), [x.wkt for x in ind])
            base_de_datos.insert_punto(conn, list(list_puntos))

        # Begin the evolution
        while max(fits) < 2.15 and g < GENERACION_PARADA:
            # A new generation
            g = g + 1
            print('Generacion {} - Max Fit: {} Avg Fit {}'.format(g, max(fits), np.average(fits)))

            generacion_id = base_de_datos.insert_generacion(conn, g, config_id)

            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))

            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CROSS_PROB:
                    toolbox.mate(child1, child2, ALPHA)#random.random())
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
                individuo_id = base_de_datos.insert_individuo(conn, i, generacion_id, ind.fitness.values[0])

                list_puntos = zip(count(start=1), repeat(individuo_id), [x.wkt for x in ind])

                base_de_datos.insert_punto(conn, list(list_puntos))

        base_de_datos.update_config(conn, config_id)

        '''
            Termina la ejecución y limpieza de variables, clases y funciones
        '''

        toolbox.unregister('select')
        toolbox.unregister('poblacion')
        toolbox.unregister('individuo')

        del creator.Individuo
        del creator.FitnessMax

    ejecucion_ga(conexion_bd)



base_de_datos.close_connection(conexion_bd)
