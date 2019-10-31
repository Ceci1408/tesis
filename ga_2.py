from deap import base, creator, tools
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


def crear_punto_aleatorio(geo_df):
    # obtengo los mínimos y máximos de los ejes X e Y del mapa "geodf"
    x_min, y_min, x_max, y_max = geo_df.total_bounds

    while True:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        point = Point(x, y)
        flg_contains = geo_df.geometry.contains(point)
        # Si el punto está dentro del mapa que devuelva el punto.
        if flg_contains.any():
            break
    return point


def calcular_nni(individual, area):
    puntos = []
    for p in individual:
        puntos.append([np.radians(p.x), np.radians(p.y)])
    nbrs = NearestNeighbors(n_neighbors=2, metric='haversine').fit(puntos)
    distances, indices = nbrs.kneighbors(puntos)
    distances = distances * 6371000 / 1000
    # este array contiene para c/ punto, la distancia con respecto a su vecino más próximo
    result = distances[:, 1]
    dobs = np.sum(result) / len(puntos)
    rn = dobs / (0.5 * np.sqrt((area / len(puntos))))
    return rn,


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gpd.GeoSeries, fitness=creator.FitnessMax)

IND_SIZE = 50


toolbox = base.Toolbox()
toolbox.register("coordenada", crear_punto_aleatorio, geo_df=barrios_caba, )
toolbox.register("individuo", tools.initRepeat, creator.Individual, toolbox.coordenada, n=IND_SIZE)
toolbox.register("poblacion", tools.initRepeat, list, toolbox.individuo, 50)

# toolbox.register("mate", tools.cxTwoPoint)
# toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
# toolbox.register("select", tools.selRoulette, fit_attr="fitness")
# toolbox.register("evaluate", calcular_nni, area=area)

ind1 = toolbox.individuo()
print(ind1)
mutant = toolbox.clone(ind1)
ind2, = tools.mutFlipBit(ind1, indpb=0.05)
del mutant.fitness.values
print(ind2)

# TODO: esto es para dibujar las coordenadas en el mapa. por ahora no es importante.
#fig, ax = plt.subplots(figsize=(15, 15))
#barrios_caba.plot(ax=ax)
#ind1.plot(ax=ax, color='green', marker='+', label='aleatorio')
#plt.legend(prop={'size': 15})
#plt.show()