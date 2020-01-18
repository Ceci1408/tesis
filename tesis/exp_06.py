"""
    En este experimento, se hace selección elitista y torneo
    De 200 individuos, se toma el 5% como la elite, redondeado para arriba
"""

import geopandas as gpd
import copy
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from shapely.geometry import Point
import random
from itertools import repeat
from datetime import datetime
import csv
import time
from math import ceil

'''
    Carga y variables del mapa
'''
#barrios_caba = gpd.read_file('/home/cecilia/Dropbox/Tesis/Puntos Verdes/barrios.csv')
barrios_caba = gpd.read_file('barrios.csv')
barrios_caba.crs = {'init': 'EPSG:4326'}
barrios_caba['area'] = np.around(barrios_caba['area'].astype(float) / 10 ** 6, decimals=3)  # paso a km2
area = barrios_caba['area'].sum()

'''
    Variables para el algoritmo.
'''
q_individuos = 200
q_coordenadas = 100
prob_cruza = 0.65
prob_mut = 0.0361019610035419
prob_mut_alelo = 0.02
mu = 0
sigma = 1
poblacion = None
fecha = datetime.now()
tipo_modelo = 'e'
# Originalmente, en la descripción de Holland, deberían ser dos los padres no más a seleccionar
k_seleccion = 2
torneo = 3
q_elitistas = ceil(5*(q_individuos/100))


'''
    NOMENCLATURA:
    -   Los primeros 2 dígitos serán destinados al nro de experimento
    -   El(los) siguiente(s) dígito(s) --> método de selección
        0: ruleta
        1: torneo
        3: elitista
'''
NRO_EXP = "06"
SELECCION ="31" # elitista combinado con torneo


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


def inicio_ga():
    max_id_ejecucion = 1

    with open('configuracion_{}{}.csv'.format(NRO_EXP, SELECCION), 'w', newline='') as csv_configuracion:
        spamwriter = csv.writer(csv_configuracion)
        spamwriter.writerow([q_individuos, q_coordenadas, prob_cruza, prob_mut, prob_mut_alelo, mu, sigma,
                            fecha, max_id_ejecucion, 'elitista_torneo'])

    poblacion = pd.DataFrame(crear_individuos(q_individuos, q_coordenadas))
    poblacion['nni'] = poblacion.apply(calcular_nni, axis=1)

    copia_poblacion = copy.deepcopy(poblacion)
    nro_gen = 0
    nni_max = 0

    # Comienzo del Algoritmo Genético
    while nni_max <= 2.15:
        mejores, offspring = seleccion_elitista(copia_poblacion)
        offspring = seleccion_torneo(torneo, offspring, q_individuos-1)

        #Cruza
        offspring['random_cruza'] = np.random.random(offspring.shape[0])
        offspring_inter = offspring[offspring['random_cruza'] < prob_cruza].iloc[:, :q_coordenadas]
        offspring = offspring.drop(offspring[offspring['random_cruza'] < prob_cruza].index)
        offspring = offspring.iloc[:, :q_coordenadas]
        offspring_inter = offspring_inter.reset_index(drop=True)

        for i in range(1, len(offspring_inter), 2):
            offspring_inter.iloc[i-1, :q_coordenadas], offspring_inter.iloc[i, :q_coordenadas] = cruzamiento(offspring_inter.iloc[i-1, :q_coordenadas], offspring_inter.iloc[i, :q_coordenadas])

        offspring = offspring.append(offspring_inter, ignore_index=True, verify_integrity=True)
        del offspring_inter

        # Mutación:
        offspring['random_mutacion'] = np.random.random(offspring.shape[0])
        offspring_inter = offspring[offspring['random_mutacion'] < prob_mut].iloc[:, :q_coordenadas]
        offspring_inter = offspring_inter.apply(mutacion, axis=1, args= (mu, sigma, prob_mut_alelo,))
        offspring = offspring.drop(offspring[offspring['random_mutacion'] < prob_mut].index).iloc[:, :q_coordenadas]
        offspring_inter = offspring_inter.reset_index(drop=True)
        offspring = offspring.append(offspring_inter, ignore_index=True, verify_integrity=True)
        del offspring_inter

        # calculo nuevamente el nni
        offspring['nni'] = offspring.apply(calcular_nni, axis=1)

        # reemplazo la poblacion entera
        copia_poblacion = offspring
        copia_poblacion = copia_poblacion.append(mejores, ignore_index=True, sort=False)

        copia_poblacion['gen'] = nro_gen
        copia_poblacion['id_ejecucion'] = max_id_ejecucion
        file_individuo = 'individuo_{}{}.csv'.format(NRO_EXP, SELECCION)
        copia_poblacion[['id_ejecucion', 'gen', 'nni']].to_csv(file_individuo, mode='a', header=False, index=False)
        copia_poblacion.apply(save_puntos, axis=1, args=(max_id_ejecucion, q_coordenadas))

        nni_max = copia_poblacion['nni'].max()
        print('Generación : {}\n NNI Max de la generación: {} '.format(nro_gen, nni_max))

        nro_gen += 1

    return copia_poblacion


def seleccion_ruleta(k, copia_poblacion, q_coordenadas):
    chosen_df = pd.DataFrame()

    sum_fit = copia_poblacion['nni'].sum()
    copia_poblacion = copia_poblacion.sort_values(by=['nni'], ascending=False)
    copia_poblacion['cum_sum'] = copia_poblacion['nni'].cumsum()
    for i in range(0, k):
        random_num = random.random()
        copia_poblacion['randomxsum_fit'] = random_num * sum_fit
        chosen = copia_poblacion[copia_poblacion.cum_sum > copia_poblacion.randomxsum_fit].head(1).iloc[:, :q_coordenadas]
        chosen_df = chosen_df.append(chosen, ignore_index=True)
    return chosen_df


# http: // sedici.unlp.edu.ar / bitstream / handle / 10915 / 4060 / III_ - _Algoritmos_evolutivos.pdf?sequence = 7 & isAllowed = y

def seleccion_torneo(tam_torneo, copia_poblacion, k):
    chosen = pd.DataFrame()
    for i in range(k):
        aspirantes = copia_poblacion.sample(n=tam_torneo)
        chosen = chosen.append(aspirantes.loc[aspirantes['nni'].idxmax()], ignore_index=True)
        del aspirantes
    return chosen


def seleccion_elitista(copia_poblacion):
    copia_poblacion = copia_poblacion.sort_values(by='nni', ascending=False)
    mejores, offspring = copia_poblacion.head(q_elitistas), copia_poblacion.tail(len(copia_poblacion) - q_elitistas)
    return mejores, offspring


def cruzamiento(ind1, ind2):
    tam = min(len(ind1), len(ind2))
    pto_cruce = np.random.randint(1, tam-1)

    ind1_copia = copy.deepcopy(ind1)
    ind1.iloc[pto_cruce:], ind2[pto_cruce:] = ind2.iloc[pto_cruce:], ind1_copia.iloc[pto_cruce:]

    del ind1_copia
    return ind1, ind2


def mutacion(ind1, mu, sigma, prob):
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
                punto_mutado = Point(ind1.iloc[i].x + random.gauss(m, s),
                                     ind1.iloc[i].y + random.gauss(m, s))

                if validar_coordenada(punto_mutado):
                    ind1 = ind1.drop([i], axis=0)
                    ind1 = ind1.append(pd.Series(punto_mutado), ignore_index=True)
                    break
    return ind1


def save_puntos(serie, id_ejecucion, q_coordenadas):
    df_temp = serie.iloc[:q_coordenadas].to_frame(name='punto')
    df_temp['punto'] = df_temp['punto'].apply(lambda x: x.wkt)
    df_temp['id_ejecucion'] = id_ejecucion
    df_temp['ind'] = serie.name
    df_temp['gen'] = serie['gen']
    file_coord = 'coordenadas_{}{}.csv'.format(NRO_EXP, SELECCION)
    df_temp.to_csv(file_coord, mode='a', header=False, index=False)
    time.sleep(2)


def main():
    inicio_ga()


if __name__ == "__main__":
    main()
