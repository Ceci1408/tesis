from deap import base, creator, tools, algorithms
from shapely.geometry import Point
import random
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from sklearn.neighbors import NearestNeighbors


barrios_caba = gpd.read_file(r'/home/cecilia/Dropbox/Tesis/Puntos Verdes/barrios.csv')
barrios_caba.crs = {'init': 'EPSG:4326'}
barrios_caba['area'] = np.around(barrios_caba['area'].astype(float) / 10 ** 6, decimals=3)  # paso a km2
area = barrios_caba['area'].sum()
# obtengo los mínimos y máximos de los ejes X e Y del mapa "geodf"
x_min, y_min, x_max, y_max = barrios_caba.total_bounds
cxpb = 0.5
# mutpb = The probability of mutating an individual.
mutpb = 0.2
# ngen = número de generaciones
ngen = 20

def check_limite(mapa):
    def decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for ind in offspring:
                for coord in ind:
                    point = Point(coord[0], coord[1])
                    flg_contains = mapa.geometry.contains(point)
                    # Si el punto está dentro del mapa que devuelva el punto.
                    if not flg_contains.any():
                        ind.fitness.values = (0,)
            return offspring
        return wrapper
    return decorator


def crear_punto_aleatorio(geo_df, x_min, x_max, y_min, y_max):
    while True:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        point = Point(x, y)
        flg_contains = geo_df.geometry.contains(point)
        if flg_contains.any():
            break
    return [point.x, point.y]


def calcular_nni(individual, area):
    puntos = []
    for p in individual:
        #puntos.append([np.radians(p.x), np.radians(p.y)])
        puntos.append([np.radians(p[0]), np.radians(p[1])])
    nbrs = NearestNeighbors(n_neighbors=2, metric='haversine').fit(puntos)
    distances, indices = nbrs.kneighbors(puntos)
    distances = distances * 6371000 / 1000
    # este array contiene para c/ punto, la distancia con respecto a su vecino más próximo
    result = distances[:, 1]
    dobs = np.sum(result) / len(puntos)
    rn = dobs / (0.5 * np.sqrt((area / len(puntos))))
    return rn,


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

IND_SIZE = 50

toolbox = base.Toolbox()
toolbox.register("coordenada", crear_punto_aleatorio, geo_df=barrios_caba, x_min=x_min, x_max=x_max, y_min=y_min,
                 y_max=y_max)
toolbox.register("individuo", tools.initRepeat, creator.Individual, toolbox.coordenada, n=IND_SIZE)
toolbox.register("poblacion", tools.initRepeat, list, toolbox.individuo, 50)

toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutGaussian,  mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selRoulette)
toolbox.register("evaluate", calcular_nni, area=area)

toolbox.decorate("mate", check_limite(mapa=barrios_caba))
toolbox.decorate("mutate", check_limite(mapa=barrios_caba))



pop = toolbox.poblacion()
# hof = tools.HallOfFame(5)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)

pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats,  # halloffame=hof,
                                   verbose=True)


print('POBLACIÓN FINAL')
print(pop)
# TODO: esto es para dibujar las coordenadas en el mapa. por ahora no es importante.
#fig, ax = plt.subplots(figsize=(15, 15))
#barrios_caba.plot(ax=ax)
#ind1.plot(ax=ax, color='green', marker='+', label='aleatorio')
#plt.legend(prop={'size': 15})
#plt.show()