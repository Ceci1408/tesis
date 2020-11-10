import pandas as pd
from tesis.conexion_bd import base_de_datos
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

conn = base_de_datos.connect()


def box_plot():
    datos, columnas = base_de_datos.get_fitness(conn)
    df = pd.DataFrame(datos, columns=columnas)
    g = sns.catplot(data=df, x='expe_id', y='fitness', hue='tipo_seleccion', kind='box')
    g.set_xlabels('Experimento')
    g.set_ylabels('índice vecino más próximo')
    g.set_titles("{col_name}")

    return g


def time_and_fitness():
    datos, columnas = base_de_datos.get_duracion(conn)
    df = pd.DataFrame(datos, columns=columnas)
    # Create combo chart
    fig, ax1 = plt.subplots(figsize=(10, 6), ncols=2)

    color = 'tab:green'
    sns.barplot(x="expe_id", y="duracion", hue='tipo_seleccion',data=df, ax=ax1)
    ax1.set_xlabel('Experimento ID')
    ax1.set_ylabel('Ejecución (hs)')
    ax1.legend(loc=0)
    ax2 = ax1.twinx()
    ax2.legend(loc=1)
    sns.lineplot(data=df['fitness_avg'], marker='o', sort=False, ax=ax2, label='NNI promedio')
    ax2.set_ylabel('NNI promedio')
    ax2.tick_params(axis='y', color=color)
    plt.legend(loc='upper center')


    #g = sns.barplot(x="expe_nro", y="duracion", hue='tipo_seleccion', data=df)

    return fig, ax1, ax2


def evolutivo_finess():
    datos, columnas = base_de_datos.evolutivo_fitness(conn)
    df = pd.DataFrame(datos, columns=columnas)

    g = sns.relplot(x="generacion_nro", y="fitness",# hue="",
            col="expe_id", col_wrap=3, estimator='mean', ci=None,
            kind="line", data=df[(df['expe_id']>=19) & (df['expe_id']<=24)]);

    return g

time_and_fitness()
plt.show()