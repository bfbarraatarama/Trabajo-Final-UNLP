# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# =======================================================================================
# Proyecto: Trabajo Final - Método de Paneles Bidimensional Multielemento No estacionario
# Archivo: src/Importacion.py
# Autor: Bruno Francisco Barra Atarama
# Institución: 
#   Departamento de Ingeniería Aeroespacial
#   Facultad de Ingeniería
#   Universidad Nacional de La Plata
# Año: 2025
# Licencia: PolyForm Noncommercial License 1.0.0
# =======================================================================================

"""
Módulo para la importación de coordenadas a partir de archivos y la 
discretización de perfiles.
"""

import numpy as np
import copy
from typing import Literal

def importarPerfil(
        ruta : str, 
        formato : Literal['selig', 'lednicer'] = 'selig', 
        cuerda : float | None = None,
        bordeDeFugaCerrado : bool = True
):
    """
    Función que importa las coordenadas x e y de un archivo de texto, ya sea en formato
    selig o lednicer.

    Parameters
    ----------
    ruta : str
        Ruta del archivo de texto, incluyendo extensión.
    formato : {'selig', 'lednicer'}
        Literal que indica en qué formato se encuentran organizadas las coordenadas en el archivo.
    cuerda : float
        Cuerda del perfil de salida.
    bordeDeFugaCerrado : bool
        Si `True`, verifica si el borde de fuga está cerrado. Si `False`, omite esta verificación.

        Por defecto, `True`.
    
    Returns
    -------
    x : List[List[float]]
        Coordenadas x cargadas. 
        - Primera lista: BF -> BA, a través del intradós.
        - Segunda lista: BA -> BF, a través del extradós.
    y : List[List[float]]
        Coordenadas y cargadas. 
        - Primera lista: BF -> BA, a través del intradós.
        - Segunda lista: BA -> BF, a través del extradós.

    Raises
    ------
    FileNotFoundError
        Si no existe el archivo en `ruta`.
    ValueError
        Si `bordeDeFugaCerrado=True` y el perfil cargado no está cerrado.
    
    Notes
    -----
    - El tipo de dato de la salida no es el utilizado en el resto de la implementación fuera de este módulo.
    - Se probó funcionamiento con diversos formatos de header y separación de coordenadas de los archivos.

        Sin embargo, en el caso de fallar se recomienda revisar el formato del archivo y 
        modificarlo si es necesario.
    """
    with open(ruta, 'r') as file:  
        perfil = file.read()
    perfil = perfil.splitlines()
    for i in range(len(perfil)):
            perfili = perfil[i].split(' ')
            perfili = list(filter(None,perfili))
            if len(perfili) == 2:
                try:
                    float(perfili[0])
                    float(perfili[1])
                except:
                    pass
                else:
                    break
    if formato == 'selig':
        x = []
        y = []
        
        for i in range(i, len(perfil)):
            perfili = perfil[i].split(' ')
            perfili = list(filter(None,perfili))
            if len(perfili) == 2:
                try:
                    x.append(float(perfili[0]))
                    y.append(float(perfili[1]))
                except:
                    pass
            else:
                break
        x.reverse()
        y.reverse()
        
        iBA = x.index(min(x))
        x1 = x[0 : iBA + 1]
        x2 = x[iBA:]

        y1 = y[0 : iBA + 1]
        y2 = y[iBA:]

        x = [x1, x2]
        y = [y1, y2]
        
    elif formato == 'lednicer':
        x1 = []
        x2 = []
        y1 = []
        y2 = []
        while list(filter(None, perfil[i+1].split(' '))) == []:
            i += 1
        i += 1

        while list(filter(None, perfil[i].split(' '))) != []:
            perfili = perfil[i].split(' ')
            perfili = list(filter(None,perfili))
            try:
                x1.append(float(perfili[0]))
                y1.append(float(perfili[1]))
            except: 
                pass
            i += 1

        while list(filter(None, perfil[i+1].split(' '))) == []:
            i += 1
        i += 1
        while list(filter(None, perfil[i].split(' '))) != [] and i < len(perfil)-1:
            perfili = perfil[i].split(' ')
            perfili = list(filter(None,perfili))
            try:
                x2.append(float(perfili[0]))
                y2.append(float(perfili[1]))
            except: 
                pass
            i += 1
        if list(filter(None, perfil[i].split(' '))) != []:
            perfili = perfil[i].split(' ')
            perfili = list(filter(None,perfili))
            try:
                x2.append(float(perfili[0]))
                y2.append(float(perfili[1]))
            except: 
                pass
        x2.reverse()
        y2.reverse()
        x = [x2, x1]
        y = [y2, y1]

    if cuerda is not None:
        cuerda0 = x[0][0] - x[0][-1]
        factor = cuerda/cuerda0
        x = [[xyi * factor for xyi in xy] for xy in x]
        y = [[xyi * factor for xyi in xy] for xy in y]

    if bordeDeFugaCerrado:
        if x[0][0] != x[1][-1] or y[0][0] != y[1][-1]:
            raise ValueError('El borde de fuga del perfil cargado no está cerrado.')
    return x, y   

def discretizarPerfil(
        ruta : str, 
        nIntrados : int, 
        nExtrados : int, 
        cuerda : float | None = None, 
        formato : Literal['selig', 'lednicer'] = 'selig', 
        espaciamiento : Literal['cos', 'tanh'] = 'cos',
        bordeDeFugaCerrado : bool = True,
) -> np.ndarray:
    """
    Función que importa las coordenadas x e y de un archivo de texto, ya sea en formato
    selig o lednicer, y las utiliza para obtener una discretización particular.

    Parameters
    ----------
    ruta : str
        Ruta del archivo de texto, incluyendo extensión.
    nIntrados : int
        Número de paneles en los que discretizar intradós del perfil.
    nIntrados : int
        Número de paneles en los que discretizar extradós del perfil.
    formato : {'selig', 'lednicer'}
        Literal que indica en qué formato se encuentran organizadas las coordenadas en el archivo.
    cuerda : float
        Cuerda del perfil de salida.
    espaciamiento : {'cos', 'tanh'}
        Tipo de espaciamiento de los nodos utilizado en las discretizaciones.
        - `'cos'`: o de Chebyshev, es la más común y aumenta la discretización en el BA y el BF.
        - `'tanh'`: incrementa la discretización más que `'cos'`. Puede resultar excesiva la discretización y
        por eso es menos común.
    bordeDeFugaCerrado : bool
        Si `True`, verifica si el borde de fuga está cerrado. Si `False`, omite esta verificación.

        Por defecto, `True`.
    
    Returns
    -------
    r_xy : np.ndarray shape (2, N + 1)
        Coordenadas del perfil en el tipo de dato utilizado en el resto de la implementación fuera de este módulo.
    
    Raises
    ------
    FileNotFoundError
        Si no existe el archivo en `ruta`.
    ValueError
        - Si `bordeDeFugaCerrado=True` y el perfil cargado no está cerrado.
        - Si `espaciamiento` no es `'cos'` ni `'tanh'`.

    Notes
    -----
    Se probó funcionamiento con diversos formatos de header y separación de coordenadas de los archivos.

    Sin embargo, en el caso de fallar se recomienda revisar el formato del archivo y 
    modificarlo si es necesario.
    """

    x, y = importarPerfil(ruta, formato, cuerda, bordeDeFugaCerrado)

    c = x[0][0] - x[0][-1]
    n = [nIntrados+1, nExtrados+1]

    xInt = []
    yInt = []
    for i in range(2):
        if espaciamiento == 'cos':
            theta = np.linspace(0, np.pi, n[i])
            xx = x[0][-1] + c / 2 * (1 - np.cos(theta))
        elif espaciamiento == 'tanh':
            k = np.arange(n[i] + 1)
            t = k / n[i]
            alfa = 4
            xx = x[0][-1] + c * 0.5*(np.tanh(alfa*(2*t-1)) / np.tanh(alfa) + 1)
        else:
            raise ValueError(f"Espaciamiento inválido: {espaciamiento!r}\nOpciones válidas: 'cos' o 'tanh'.")

        xi = copy.deepcopy(x[i])
        yi = copy.deepcopy(y[i])
        if i == 0:
            xx = np.flip(xx)
            xx = xx[0:-1]
            xi.reverse()
            yi.reverse()
        yy = np.interp(xx, xi, yi)

        xx = xx.tolist()
        yy = yy.tolist()


        xInt += xx
        yInt += yy
        
    return np.stack([np.array(xInt), np.array(yInt)], axis=0) 


def polaresXFoil(ruta: str):
    with open(ruta, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # localizar header (línea que contiene 'alpha')
    header_idx = next((i for i, ln in enumerate(lines) if 'alpha' in ln.lower()), None)
    if header_idx is None:
        raise ValueError("No se encontró cabecera con 'alpha' en el archivo XFOIL.")

    header = lines[header_idx].lower().split()

    def idx(*names):
        for n in names:
            if n in header:
                return header.index(n)
        return None

    i_alpha = idx('alpha', 'alfa')
    i_cl    = idx('cl')
    i_cm    = idx('cm')
    i_cd    = idx('cd')

    alpha, Cl, Cm, Cd = [], [], [], []
    for ln in lines[header_idx+1:]:
        if not ln.strip():
            continue
        parts = ln.split()
        # línea con separadores o insuficiente longitud
        max_i = max([i for i in [i_alpha, i_cl, i_cm, i_cd] if i is not None])
        if len(parts) <= max_i:
            continue
        try:
            a = float(parts[i_alpha]) if i_alpha is not None else None
            if a is None:
                continue
            alpha.append(a)
            Cl.append(float(parts[i_cl]) if i_cl is not None else np.nan)
            Cm.append(float(parts[i_cm]) if i_cm is not None else np.nan)
            Cd.append(float(parts[i_cd]) if i_cd is not None else np.nan)
        except ValueError:
            # saltear líneas no numéricas
            continue

    return {
        'alfa': np.asarray(alpha, dtype=float),
        'Cl':        np.asarray(Cl,    dtype=float),
        'Cm':        np.asarray(Cm,    dtype=float),
        'Cd':        np.asarray(Cd,    dtype=float),
    }