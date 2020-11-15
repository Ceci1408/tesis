import pandas as pd
import geopandas as gpd
from tesis.conexion_bd import base_de_datos
import seaborn as sns
from shapely import wkt
import folium
from selenium import webdriver
import os
import time as t
import imageio
import re


def boxplot(conn):
    datos, columnas = base_de_datos.get_fitness(conn)
    df = pd.DataFrame(datos, columns=columnas)
    df['alpha'] = df['alpha'].fillna(0)
    g = sns.catplot(data=df, x='expe_id', y='fitness', hue='tipo_seleccion', col='alpha', kind='box')
    g.savefig('boxplot.png', transparent=False, )


def time(conn):
    datos, columnas = base_de_datos.get_duracion(conn)
    df = pd.DataFrame(datos, columns=columnas)
    df['alpha'] = df['alpha'].fillna('sin incializar')
    g = sns.catplot(x='expe_nro', y='duracion', hue='tipo_seleccion', col='alpha', kind='bar', data=df)
    g.savefig('time.png', transparent=False)


def fitness(conn):
    datos, columnas = base_de_datos.get_duracion(conn)
    df = pd.DataFrame(datos, columns=columnas)
    df['alpha'] = df['alpha'].fillna('N/A')

    g = sns.catplot(x="expe_nro", y="fitness_avg", data=df, kind='bar', hue='tipo_seleccion',
                    col='alpha')  # hue='tipo_seleccion', col='alpha',
    g.savefig('fitness.png')


def evolutivo_fitness(conn):
    datos, columnas = base_de_datos.evolutivo_fitness(conn)
    df = pd.DataFrame(datos, columns=columnas)
    df['alpha'] = df['alpha'].fillna('sin incializar')

    inicio = 1

    for i in range(int(df['expe_nro'].max() / 6)):
        print(i)
        g = sns.relplot(x="generacion_nro", y="fitness", hue='alpha',
                        col="expe_nro", col_wrap=3, estimator='mean', ci=None,
                        kind="line", data=df[(df['expe_nro'] >= inicio) & (df['expe_nro'] < inicio + 6)])
        inicio = inicio + 6
        g.savefig('evolutivo_{}.png'.format(i))
        print('paso por aca y en teoria se guardó')

    print('ya sali')


def promedio_fitness(conn):
    datos, columnas = base_de_datos.get_duracion(conn)
    df = pd.DataFrame(datos, columns=columnas)
    df['alpha'] = df['alpha'].fillna('sin incializar')

    g = sns.catplot(x='expe_nro', y='fitness_avg', hue='tipo_seleccion', col='alpha', data=df, kind='bar')
    g.savefig('promedio_fit.png')


def visualizacion_mapa(conn):
    datos, columnas = base_de_datos.mejor_resultado(conn)
    df = gpd.GeoDataFrame(datos, columns=columnas)
    df['punto_coordenada'] = df['punto_coordenada'].apply(wkt.loads)
    df.set_geometry('punto_coordenada', inplace=True, crs='EPSG:4326')
    coordenadas_aprox_centro_caba= [-34.615016, -58.441922]


    nro_generaciones = df['generacion_nro'].unique().tolist()
    for i in (nro_generaciones):
        df_temp = df[df['generacion_nro']==i]
        titulo = 'Generacion {}'.format(i)

        title_html = '''
                     <h3 align="center" style="font-size:16px"><b>{}</b></h3>
                     '''.format(titulo)

        map = folium.Map(coordenadas_aprox_centro_caba, zoom_start=12)
        map.get_root().html.add_child(folium.Element(title_html))

        for j in range(0, len(df_temp)):
            folium.Circle(
                location=[df_temp['punto_coordenada'].iloc[j].y, df_temp['punto_coordenada'].iloc[j].x],
                radius=15, fill=True, color='crimson',).add_to(map)
        del df_temp

        map.save('mapa_generacion_{}.html'.format(i))


def crear_html_png():
    delay = 5
    print(os.getcwd())

    for file in os.listdir(os.getcwd()):
        if file.endswith('.html'):
            input_dir = 'file://{path}/{mapfile}'.format(path=os.getcwd(), mapfile=file)
        # Open a browser window...
            browser = webdriver.Firefox(executable_path=os.path.abspath("geckodriver"))
            browser.get(input_dir)
            t.sleep(delay)
            curr_dir = os.getcwd()
            src_folder = os.path.join(curr_dir, 'test_png')

            if not os.path.exists(src_folder):
                os.mkdir(src_folder)

            pic = file.split(".")[0] + ".png"
            browser.save_screenshot(os.path.join(src_folder,pic))
            browser.quit()


def create_gif():
    images = []
    dir = os.path.join(os.getcwd(), 'test_png')
    archivos = os.listdir(dir)
    archivos.sort(key=lambda f: int(re.sub('\D', '', f)))
    for filename in archivos:
        print(filename)
        images.append(imageio.imread(os.path.join(dir, filename)))

    output_file = os.path.join(dir,'Gif-final.gif')
    imageio.mimsave(output_file, images, duration=5)


def generar_visualizaciones(conn):
    #visualizacion_mapa(conn)
    #crear_html_png()
    create_gif()
    #boxplot(conn)
    #time(conn)
    #fitness(conn)
    #evolutivo_fitness(conn)
    #promedio_fitness(conn)
