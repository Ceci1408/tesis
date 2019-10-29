import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt
import psycopg2
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from deap import creator, base, tools, algorithms
from shapely.geometry import Point
from configparser import ConfigParser


def crear_punto_aleatorio(geo_df):
    # obtengo los mínimos y máximos de los ejes X e Y del mapa "geodf"
    x_min, y_min, x_max, y_max = geo_df.total_bounds
    # TODO no puedo generar semilla porque siempre me elige el mismo punto, que si está fuera del mapa en la recursividad,
    # utiliza siempre la misma coordenada
    # genero la semilla
    # np.random.seed(7)
    while True:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)

        point = Point(x, y)
        #point = gpd.GeoSeries(gpd.points_from_xy(x=[x], y=[y]))
        #point_within = point.within(geo_df.geometry)
        #point_within = point[point.within(geo_df.unary_union)]
        flg_contains = geo_df.geometry.contains(point)
        # Si el punto está dentro del mapa que devuelva el punto.
        if flg_contains.any():
            break
    return point


def calcular_nni(individual, area):
    puntos = []
    for p in individual:
        puntos.append([np.radians(p.x), np.radians(p.y)])
    nbrs = NearestNeighbors(n_neighbors=20, metric='haversine').fit(puntos)
    distances, indices = nbrs.kneighbors(puntos)
    distances = distances * 6371000 / 1000
    # este array contiene para c/ punto, la distancia con respecto a su vecino más próximo
    result = distances[:, 1]
    dobs = np.sum(result) / len(puntos)
    rn = dobs / (0.5 * np.sqrt((area / len(puntos))))
    return rn,


def ejecutar_ga(q_puntos, tam_poblacion, mapa, area, cxpb, mutpb, ngen):
    # Creo los tipos
    creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Inicializo
    toolbox = base.Toolbox()
    # -m scoop hay que correr el programa con esto antes para que sea multiprocesador (interpreter options)
    # https://deap.readthedocs.io/en/master/tutorials/basic/part4.html
    #toolbox.register("map", futures.map)
    toolbox.register("coordenada", crear_punto_aleatorio, geo_df=mapa, )
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.coordenada, q_puntos)
    toolbox.register("poblacion", tools.initRepeat, list, toolbox.individual, tam_poblacion)

    #ind1 = toolbox.individual()
    #ind1.fitness.values = calcular_nni(ind1, area)
    #print(ind1.fitness.valid)
    #print(ind1.fitness)

    #k_to_select = math.floor(tam_poblacion * 0.75)
    # Operadores
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    # In eaSimple, selection is made to replace the whole population with k=len(population).
    # #Evolutionary Computation 1 : Basic Algorithms and Operators (pág 167)
    toolbox.register("select", tools.selRoulette)#, fit_attr='FitnessMax')
    toolbox.register("evaluate", calcular_nni, area=area)

    #Algoritmo genético.
    pop = toolbox.poblacion()
    #hof = tools.HallOfFame(5)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats, #halloffame=hof,
                                       verbose=True)

    return pop, logbook#, hof


def config(filename='database.ini', section='postgresql'):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db


def connect():
    conn = None
    try:
        params = config()

        # connect to the PostgreSQL server
        print('Conectándose a la base de datos...')
        conn = psycopg2.connect(**params)


        return conn

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


def save_results(poblacion):
    conn = connect()
    cur = conn.cursor()

    cur.execute('INSERT INTO poblacion VALUES (%s)', [datetime.now()])
    cur.execute('SELECT MAX(id_poblacion) FROM poblacion;')
    max_id_poblacion = cur.fetchone()[0]

    lista_puntos = []
    for ind in poblacion:
        fitness = ind.fitness.values[0]
        cur.execute('INSERT INTO individuo (id_poblacion, fitness) VALUES (%s,%s)', [max_id_poblacion, fitness])
        cur.execute('SELECT MAX(id_individuo) FROM individuo;')
        max_id_individuo = cur.fetchone()[0]

        for i in range(0, len(ind)):
            lista_puntos.append((max_id_individuo, ind[i].wkt))
        cur.executemany("INSERT INTO punto (id_individuo, point) VALUES (%s, ST_GeomFromText(%s,4326))", lista_puntos)
        conn.commit()

    print('holis')


def main():
    # barrios CABA
    barrios_caba = gpd.read_file(r'/home/cecilia/Dropbox/Tesis/Puntos Verdes/barrios.csv')
    barrios_caba.crs = {'init': 'EPSG:4326'}
    barrios_caba['area'] = np.around(barrios_caba['area'].astype(float) / 10 ** 6, decimals=3)  # paso a km2
    area = barrios_caba['area'].sum()

    # densidad CABA (por ahora no lo uso!)
    densidad_caba = gpd.read_file(r'/home/cecilia/Dropbox/Tesis/Datos/Demografía/caba.geojson')
    densidad_caba.crs = {'init': 'EPSG:4326'}

    # Configuracion
    q_puntos = 50
    tam_poblacion = 20

    # Ejecuto algoritmo genético.
    # Statistics will automatically be computed on the population every generation.
    # cxpb = The probability of mating two individuals. (cross)
    cxpb = 0.5
    # mutpb = The probability of mutating an individual.
    mutpb = 0.2
    # ngen = número de generaciones
    ngen = 10000

    #pop, logbook, best = ejecutar_ga(q_puntos, tam_poblacion, barrios_caba, area, cxpb, mutpb, ngen)
    pop, logbook,  = ejecutar_ga(q_puntos, tam_poblacion, barrios_caba, area, cxpb, mutpb, ngen)
    print('POBLACIÓN FINAL')

    save_results(pop)

    #print('\nMEJOR INDIVIDUO')
    #print(best)

    print('\nLOGBOOK')
    print(logbook)


if __name__ == "__main__":
    main()


    #asdasda