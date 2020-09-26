import psycopg2
from config import config
from datetime import datetime


def connect():
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            create_table(conn)
            return conn


def add_postgis(conn):
    cur = conn.cursor()
    cur_2 = conn.cursor()

    cur.execute('CREATE EXTENSION postgis;')
    cur_2.execute('ALTER EXTENSION postgis UPDATE;')

    conn.commit()
    cur.close()
    cur_2.close()


def create_table(conn):
    # create a cursor
    cur = conn.cursor()

    # execute a statement
    cur.execute('CREATE TABLE IF NOT EXISTS ga_config '
                '('
                '   ga_config_id SMALLSERIAL PRIMARY KEY , '
                '   fecha_hora_comienzo timestamp NOT NULL, '
                '   cant_individuos SMALLINT, '
                '   cant_coordenadas SMALLINT, '
                '   prob_mut REAL,'
                '   prob_cruce REAL,'
                '   prob_mut_alelo REAL,'
                '   tipo_seleccion VARCHAR(15) NOT NULL,'
                '   torneo_tam SMALLINT, '
                '   mu_x REAL,'
                '   mu_y REAL, '
                '   sigma_x REAL,'
                '   sigma_y REAL,'
                '   nro_gen_parada SMALLINT, '
                '   fecha_hora_fin timestamp'
                ');'
                ''
                'CREATE TABLE IF NOT EXISTS generacion'
                '('
                '   generacion_id SERIAL UNIQUE,'
                '   generacion_nro SMALLINT,'
                '   ga_config_id SMALLINT REFERENCES ga_config (ga_config_id),'
                '   '
                '   CONSTRAINT unique_generacion UNIQUE (generacion_id, generacion_nro, ga_config_id)'
                ');'
                ''
                'CREATE TABLE IF NOT EXISTS individuo'
                '('
                '   individuo_id BIGSERIAL UNIQUE, '
                '   individuo_nro SMALLINT,'
                '   generacion_id INT REFERENCES generacion(generacion_id),'
                '   fitness REAL,'
                '   '
                '   CONSTRAINT unique_ind_generacion UNIQUE (individuo_id, individuo_nro, generacion_id)'
                ');'
                ''
                'CREATE TABLE IF NOT EXISTS punto'
                '('
                '   punto_id BIGSERIAL UNIQUE ,'
                '   punto_nro SMALLINT,'
                '   individuo_id BIGINT REFERENCES individuo (individuo_id), '
                '   punto_coordenada geography(POINT) ,'
                ''
                '   CONSTRAINT unique_punto_ind UNIQUE (punto_id, punto_nro, individuo_id)'
                ');')

    conn.commit()

    # close the communication with the PostgreSQL
    cur.close()


def insert_config(conn, cant_ind, cant_coord, prob_mut, prob_cruce, prob_mut_alelo, tipo_seleccion, torneo_tam, mu_x,
                  mu_y, sigma_x, sigma_y, nro_gen_parada):

    hora_comienzo = datetime.now()

    sql = """INSERT INTO ga_config VALUES (DEFAULT, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s ) 
    RETURNING ga_config_id"""

    cursor = conn.cursor()

    cursor.execute(sql, (hora_comienzo, cant_ind, cant_coord, prob_mut, prob_cruce, prob_mut_alelo, tipo_seleccion,
                   torneo_tam, mu_x, mu_y, sigma_x, sigma_y, nro_gen_parada))

    ga_config_id = cursor.fetchone()[0]

    conn.commit()
    cursor.close()

    return ga_config_id


def insert_generacion(conn, gen_nro, config_id):
    sql = """INSERT INTO generacion VALUES (DEFAULT, %s, %s) RETURNING generacion_id"""
    cursor = conn.cursor()
    cursor.execute(sql, (gen_nro, config_id))
    generacion_id = cursor.fetchone()[0]

    conn.commit()
    cursor.close

    return generacion_id


def insert_individuo(conn, individuo_nro, generacion_id, fitness):
    sql = """INSERT INTO individuo VALUES (DEFAULT, %s, %s, %s ) RETURNING individuo_id"""
    cursor = conn.cursor()
    cursor.execute(sql, (individuo_nro, generacion_id, fitness))
    individuo_id = cursor.fetchone()[0]

    conn.commit()
    cursor.close

    return individuo_id


def insert_punto(conn, lista_puntos):
    sql = """INSERT INTO punto (punto_nro, individuo_id, punto_coordenada) 
    VALUES (%s, %s, ST_GeomFromText(%s, 4326))"""
    cursor = conn.cursor()
    cursor.executemany(sql, lista_puntos)

    conn.commit()
    cursor.close


def close_connection(conn, config_id):
    hora_fin = datetime.now()

    sql = """UPDATE TABLE ga_config SET fecha_hora_fin = %s WHERE ga_config_id =%s"""
    cursor = conn.cursor()
    cursor.execute(sql, (hora_fin, config_id))
    cursor.close()
    conn.close()