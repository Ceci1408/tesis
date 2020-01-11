import geopandas as gpd
import copy
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from shapely.geometry import Point
import random
from sqlalchemy import create_engine, Table, Column, Integer, MetaData, Float, UniqueConstraint, DateTime, sql, func
from geoalchemy2 import Geometry, WKTElement
from itertools import repeat
from datetime import datetime

from hyperopt import fmin, tpe, hp, Trials
from collections import OrderedDict

SPACE = OrderedDict([
    ('q_individuos', 30+hp.randint('q_individuos', 40)),
    ('q_coordenadas', 50+hp.randint('q_coordenadas', 100)),
    ('prob_cruza', hp.uniform('prob_cruza', 0, 0.5)),
    ('prob_mut', hp.uniform('prob_mut', 0, 0.05)),
    ('prob_mut_alelo', hp.uniform('prob_mut_alelo', 0, 0.05)),
    ('mu', hp.uniform('mu', 0, 1)),
    ('sigma', hp.uniform('sigma', 0, 1))
                    ])


'''
    Carga y variables del mapa
'''
barrios_caba = gpd.read_file(r'/home/cecilia/Dropbox/Tesis/Puntos Verdes/barrios.csv')
barrios_caba.crs = {'init': 'EPSG:4326'}
barrios_caba['area'] = np.around(barrios_caba['area'].astype(float) / 10 ** 6, decimals=3)  # paso a km2
area = barrios_caba['area'].sum()

'''
    Variables para el algoritmo genético
'''
# q_individuos = 10
# q_coordenadas = 100
# nro_generaciones = 20
# prob_cruza = 0.4
# prob_mut = 0.025
# prob_mut_alelo = 0.02
# mu = 0
# sigma = 1
# poblacion = None
# fecha = datetime.now()
# tipo_modelo = 'e'
# # Originalmente, en la descripción de Holland, deberían ser dos los padres no más a seleccionar
# k_seleccion = 2


'''
    BASE DE DATOS:
    Para crear la base de datos, y que pueda almacenar tipo de datos "Spatial", hay que agregarle la extensión:
        CREATE EXTENSION postgis;
'''
engine = create_engine('postgresql://cecilia:pipo1234@localhost/tesis_ga', echo=False)
metadata = MetaData()
configuracion = Table('ag_configuracion', metadata,
                      Column('q_individuos', Integer),
                      Column('q_coordenadas', Integer),
                      Column('prob_cruza', Float),
                      Column('prob_mutacion', Float),
                      Column('prob_mut_alelo', Float),
                      Column('mu', Float),
                      Column('sigma', Float),
                      Column('fecha', DateTime),
                      Column('id_ejecucion', Integer, primary_key=True, autoincrement=True),
                      extend_existing=True)
individuo = Table('individuo', metadata,
                  Column('id_ejecucion', Integer),
                  Column('gen', Integer),
                  Column('ind', Integer),
                  Column('nni', Float),
                  UniqueConstraint('id_ejecucion','gen', 'ind'), extend_existing=True)
coordenadas = Table('coordenadas', metadata,
                    Column('id_ejecucion', Integer),
                    Column('gen', Integer),
                    Column('ind', Integer),
                    Column('punto', Geometry(geometry_type='POINT', srid=4326)), extend_existing=True)

metadata.create_all(engine, checkfirst=True)


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


def inicio_ga(args):
    # q_individuos, q_coordenadas, prob_cruza, prob_mut, prob_mut_alelo, mu, sigma = args

    q_individuos = int(args.get('q_individuos'))
    q_coordenadas = int(args.get('q_coordenadas'))
    prob_cruza = args.get('prob_cruza')
    prob_mut = args.get('prob_mut')
    prob_mut_alelo = args.get('prob_mut_alelo')
    mu = args.get('mu')
    sigma = args.get('sigma')

    max_id_ejecucion = 1

    poblacion = pd.DataFrame(crear_individuos(q_individuos, q_coordenadas))
    poblacion['nni'] = poblacion.apply(calcular_nni, axis=1)


    copia_poblacion = copy.deepcopy(poblacion)

    k_seleccion = len(poblacion)

    #offspring = seleccion_ruleta(k_seleccion, copia_poblacion)
    offspring = seleccion_torneo(3, copia_poblacion, k_seleccion)
    # Offspring es mi "Mating Buffer".

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
    offspring_inter = offspring[offspring['random_mutacion'] < prob_mut].iloc[:, :q_coordenadas].apply(mutacion, axis=1, args= (mu, sigma, prob_mut_alelo,))
    offspring = offspring.drop(offspring[offspring['random_mutacion'] < prob_mut].index).iloc[:, :q_coordenadas]
    offspring = offspring.append(offspring_inter, ignore_index=True, verify_integrity=True)
    del offspring_inter

    # calculo nuevamente el nni
    offspring['nni'] = offspring.apply(calcular_nni, axis=1)

    # reemplazo la poblacion entera
    copia_poblacion = offspring
    copia_poblacion['id_ejecucion'] = max_id_ejecucion
    #copia_poblacion[['id_ejecucion', 'gen', 'nni']].to_sql('individuo', engine, if_exists='append', index_label=('ind'))
    #copia_poblacion.apply(save_puntos, axis=1, args=(max_id_ejecucion,))

    nni_max = copia_poblacion['nni'].max()
    print('NNI Max: {} '.format(nni_max))
    print('Individuos: {} - Coordenadas: {} - Prob Cruza: {} - Prob Mut: {} - Prob mut alelo: {} - Mu: {} - Sigma: {}'.
          format(q_individuos, q_coordenadas, prob_cruza, prob_mut, prob_mut_alelo, mu, sigma))
    return -nni_max


def seleccion_ruleta(k, copia_poblacion, q_coordenadas):
    chosen_df = pd.DataFrame()

    sum_fit = copia_poblacion['nni'].sum()
    copia_poblacion['cum_sum'] = copia_poblacion['nni'].cumsum()
    for i in range(k):
        copia_poblacion['random'] = np.random.random(copia_poblacion.shape[0])
        copia_poblacion['random'] = copia_poblacion['random']*sum_fit
        chosen = copia_poblacion[copia_poblacion.cum_sum > copia_poblacion.random].head(1).iloc[:, :q_coordenadas]
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
    df_temp['punto'] = df_temp['punto'].apply(lambda x: WKTElement(x.wkt, srid=4326))
    df_temp['id_ejecucion'] = id_ejecucion
    df_temp['ind'] = serie.name
    df_temp['gen'] = serie['gen']
    df_temp.to_sql('coordenadas', engine, if_exists='append', index=False, dtype={'punto': Geometry('POINT', srid=4326)})


#inicio_ga()
best = fmin(inicio_ga, space=SPACE, max_evals=300, trials=Trials(), rstate=None, show_progressbar=True, algo=tpe.suggest)
print(best)
print(hp.space_eval(SPACE, best))
#inicio_ga('g')
