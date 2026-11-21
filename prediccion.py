"""Modelo de Poisson para predecir partidos de la Copa Libertadores.

Mejoras respecto a la versión inicial:
- Usa el histórico completo 2000-2024.
- Pondera cada partido por antigüedad (vida media de 4 ediciones): pesa más lo reciente.
- Suavizado bayesiano: las fuerzas de ataque/defensa se mezclan con el promedio del
  torneo según cuántos partidos tenga el equipo. Así un equipo con 1-2 partidos no
  genera probabilidades absurdas (antes podía dar 0 % de victoria).
- Normaliza la matriz de marcadores (las probabilidades suman 100 %).
- Devuelve goles esperados, marcadores más probables, +2.5 goles, ambos marcan e historial.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

DATA_DIR = Path(__file__).resolve().parent / 'data'
ARCHIVO = DATA_DIR / 'libertadores-results-2000-2024.csv'
VIDA_MEDIA_EDICIONES = 4.0   # un partido de hace 4 ediciones pesa la mitad
PARTIDOS_PREVIOS = 4.0       # "partidos virtuales" con el promedio del torneo (suavizado)
MAX_GOLES = 10


def _cargar():
    archivo = ARCHIVO if ARCHIVO.exists() else DATA_DIR / 'libertadores-results-ds.csv'
    df = pd.read_csv(archivo, encoding='utf-8')
    df['Home Club'] = df['Home Club'].str.strip()
    df['Away Club'] = df['Away Club'].str.strip()
    df = df.dropna(subset=['Home Score', 'AwayScore'])
    ultima = df['Edition'].max()
    df['peso'] = 0.5 ** ((ultima - df['Edition']) / VIDA_MEDIA_EDICIONES)
    return df


df_partidos = _cargar()
EDICIONES = (int(df_partidos['Edition'].min()), int(df_partidos['Edition'].max()))

_w = df_partidos['peso']
mu_local = float(np.average(df_partidos['Home Score'], weights=_w))       # goles promedio del local
mu_visitante = float(np.average(df_partidos['AwayScore'], weights=_w))   # goles promedio del visitante


def _tasa(grupo, columna, media):
    """Promedio ponderado con suavizado hacia la media del torneo."""
    suma = (grupo[columna] * grupo['peso']).sum()
    n = grupo['peso'].sum()
    return (suma + PARTIDOS_PREVIOS * media) / (n + PARTIDOS_PREVIOS)


def _fuerzas():
    filas = {}
    equipos = sorted(set(df_partidos['Home Club']) | set(df_partidos['Away Club']))
    for eq in equipos:
        casa = df_partidos[df_partidos['Home Club'] == eq]
        fuera = df_partidos[df_partidos['Away Club'] == eq]
        filas[eq] = {
            'ataque_local': _tasa(casa, 'Home Score', mu_local) / mu_local,
            'defensa_local': _tasa(casa, 'AwayScore', mu_visitante) / mu_visitante,
            'ataque_visita': _tasa(fuera, 'AwayScore', mu_visitante) / mu_visitante,
            'defensa_visita': _tasa(fuera, 'Home Score', mu_local) / mu_local,
            'partidos': int(len(casa) + len(fuera)),
        }
    return pd.DataFrame.from_dict(filas, orient='index')


df_fuerzas = _fuerzas()


def obtener_lista_equipos():
    """Lista de equipos ordenada alfabéticamente."""
    return df_fuerzas.index.tolist()


def _historial(local, visitante, limite=5):
    h = df_partidos[((df_partidos['Home Club'] == local) & (df_partidos['Away Club'] == visitante))
                    | ((df_partidos['Home Club'] == visitante) & (df_partidos['Away Club'] == local))]
    h = h.sort_values('Edition', ascending=False).head(limite)
    return [
        {'edicion': int(r['Edition']), 'ronda': str(r['Round']), 'local': r['Home Club'],
         'visitante': r['Away Club'], 'marcador': f"{int(r['Home Score'])}-{int(r['AwayScore'])}"}
        for _, r in h.iterrows()
    ]


def obtener_prediccion(equipo_local, equipo_visitante):
    """Calcula las probabilidades del partido y devuelve un dict listo para el frontend."""
    if equipo_local not in df_fuerzas.index or equipo_visitante not in df_fuerzas.index:
        raise ValueError('Uno de los equipos no se encuentra en la base de datos.')
    if equipo_local == equipo_visitante:
        raise ValueError('Selecciona dos equipos diferentes.')

    L, V = df_fuerzas.loc[equipo_local], df_fuerzas.loc[equipo_visitante]
    lambda_local = L['ataque_local'] * V['defensa_visita'] * mu_local
    lambda_visitante = V['ataque_visita'] * L['defensa_local'] * mu_visitante

    goles = np.arange(MAX_GOLES + 1)
    matriz = np.outer(poisson.pmf(goles, lambda_local), poisson.pmf(goles, lambda_visitante))
    matriz /= matriz.sum()  # filas = goles del local, columnas = goles del visitante

    p_local = float(np.tril(matriz, -1).sum())
    p_empate = float(np.trace(matriz))
    p_visitante = float(np.triu(matriz, 1).sum())

    # Redondeo que siempre suma 100 %
    pct = np.array([p_local, p_empate, p_visitante]) * 100
    redondeado = np.round(pct, 2)
    redondeado[int(np.argmax(pct))] += round(100 - redondeado.sum(), 2)

    top = np.dstack(np.unravel_index(np.argsort(matriz.ravel())[::-1][:5], matriz.shape))[0]
    marcadores = [{'marcador': f'{i}-{j}', 'prob': round(float(matriz[i, j]) * 100, 2)} for i, j in top]

    total = np.add.outer(goles, goles)
    return {
        'prob_local': float(redondeado[0]),
        'prob_empate': float(redondeado[1]),
        'prob_visitante': float(redondeado[2]),
        'goles_esperados_local': round(float(lambda_local), 2),
        'goles_esperados_visitante': round(float(lambda_visitante), 2),
        'marcadores_probables': marcadores,
        'prob_mas_2_5': round(float(matriz[total > 2.5].sum()) * 100, 1),
        'prob_ambos_marcan': round(float(matriz[1:, 1:].sum()) * 100, 1),
        'partidos_local': int(L['partidos']),
        'partidos_visitante': int(V['partidos']),
        'historial': _historial(equipo_local, equipo_visitante),
        'ediciones': list(EDICIONES),
    }
