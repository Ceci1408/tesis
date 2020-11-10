import psycopg2
from .config import config
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
                '   alpha REAL,'
                '   fecha_hora_fin timestamp,'
                '   flg_valido SMALLINT'
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
                  mu_y, sigma_x, sigma_y, nro_gen_parada, alpha):

    hora_comienzo = datetime.now()

    sql = """INSERT INTO ga_config 
    (
        ga_config_id,
        fecha_hora_comienzo, 
        cant_individuos, 
        cant_coordenadas, 
        prob_mut, 
        prob_cruce, 
        prob_mut_alelo, 
        tipo_seleccion, 
        torneo_tam, 
        mu_x, 
        mu_y, 
        sigma_x, 
        sigma_y, 
        nro_gen_parada, 
        alpha
    )
    VALUES (DEFAULT, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s , %s) 
    RETURNING ga_config_id"""

    cursor = conn.cursor()

    cursor.execute(sql, (hora_comienzo, cant_ind, cant_coord, prob_mut, prob_cruce, prob_mut_alelo, tipo_seleccion,
                   torneo_tam, mu_x, mu_y, sigma_x, sigma_y, nro_gen_parada, alpha))

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


def update_config(conn, config_id):
    hora_fin = datetime.now()

    sql = """UPDATE ga_config SET fecha_hora_fin = %s WHERE ga_config_id =%s"""
    cursor = conn.cursor()
    cursor.execute(sql, (hora_fin, config_id))

    conn.commit()
    cursor.close()


def close_connection(conn):
    conn.close()


def get_fitness(conn):
    sql = """
    with experimentos as
    (
        select rank() over(order by ga_config_id) as expe_id,
        tipo_seleccion ,
        ga_config_id 
        from ga_config gc 
        where flg_valido =1 and ga_config_id >=61
    )
    select e.expe_id, e.tipo_seleccion, iv.fitness 
    from experimentos as e
    inner join individuos_validos as iv
    on e.ga_config_id = iv.ga_config_id 
    """
    columns =['expe_id', 'tipo_seleccion', 'fitness']

    cursor = conn.cursor()
    cursor.execute(sql)
    test = cursor.fetchall()
    cursor.close()

    return test, columns


def get_duracion(conn):
    sql = """
    with avg_fit as
    (
        select ga_config_id , avg(fitness)
        from individuos_validos iv 
        where ga_config_id >=61
        group by ga_config_id 
    )
    select 
        rank() OVER(order by gc.ga_config_id),
        tipo_seleccion  , 
        fecha_hora_comienzo , 
        fecha_hora_fin ,
        alpha,
        extract(epoch from age(fecha_hora_fin, fecha_hora_comienzo))/3600, 
        b.avg
    from ga_config as gc
    inner join avg_fit as b
        on gc.ga_config_id = b.ga_config_id
    where flg_valido = 1 and gc.ga_config_id >=61;
    """

    columns = ['expe_id','tipo_seleccion','fecha_hora_comienzo', 'fecha_hora_fin', 'alpha','duracion', 'fitness_avg']

    cursor = conn.cursor()
    cursor.execute(sql)
    test = cursor.fetchall()
    cursor.close()

    return test, columns


def evolutivo_fitness(conn):
    sql = """
     with expe_id as 
    (
        select rank() over(order by ga_config_id) as expe_id, 
        ga_config_id 
        from ga_config
        where flg_valido =1 and ga_config_id >= 61
    )
    select eid.expe_id, 
    generacion_nro ,
    fitness
    
    from individuos_validos iv 
    inner join expe_id as eid
    on iv.ga_config_id = eid.ga_config_id
    order by expe_id desc
    """
    columns = ['expe_id','generacion_nro', 'fitness']

    cursor = conn.cursor()
    cursor.execute(sql)
    test = cursor.fetchall()
    cursor.close()

    return test, columns