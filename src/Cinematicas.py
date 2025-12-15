# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# =======================================================================================
# Proyecto: Trabajo Final - Método de Paneles Bidimensional Multielemento No estacionario
# Archivo: src/Cinematicas.py
# Autor: Bruno Francisco Barra Atarama
# Institución: 
#   Departamento de Ingeniería Aeroespacial
#   Facultad de Ingeniería
#   Universidad Nacional de La Plata
# Año: 2025
# Licencia: PolyForm Noncommercial License 1.0.0
# =======================================================================================

"""
Módulo para definir las cinemáticas de sólidos, del vector respecto al cual
se toman momentos o los nodos de un flap.
"""

import numpy as np
from numpy.typing import ArrayLike
from typing import overload, Literal, Tuple, Generator
import csv

from ._TernasMoviles2D import TernasMoviles2D
from .MP2D import MP2D

from .Tipos import CinematicaSalida, CinematicaCompletaSalida
from .Tipos import MRUParams, AOAParams, CustomParams, RotacionArmonicaParams, TraslacionArmonicaParams, NPZParams, AeroelasticidadParams, CuerpoRigidoCSVParams
from .Tipos import RMParams, RMDesdeCinematicaROParams, RMAeroelasticidadParams, RMCSVParams
from .Tipos import FlapParams
from .Tipos import GeneradorCinematica, GeneradorCinematicaAeroelasticidad, GeneradorRM, GeneradorRMAeroelasticidad
from .Tipos import ActualizadorCinematica, ActualizadorRM

# =============================================================================
# Cinemática de un sólido
# =============================================================================

@overload
def cinematica(
        tipo: Literal['aoa'], 
        params: AOAParams,
        rutaBase: str | None = None,
        formatoGuardado: Literal['npz', 'cuerpoRigidoCSV'] | None = 'npz',
) -> GeneradorCinematica: ...
@overload
def cinematica(
        tipo: Literal['mru'], 
        params: MRUParams,
        rutaBase: str | None = None,
        formatoGuardado: Literal['npz', 'cuerpoRigidoCSV'] | None = 'npz',
) -> GeneradorCinematica: ...
@overload
def cinematica(
        tipo: Literal['custom'], 
        params: CustomParams,
        rutaBase: str | None = None,
        formatoGuardado: Literal['npz', 'cuerpoRigidoCSV'] | None = 'npz',
) -> GeneradorCinematica: ...
@overload
def cinematica(
        tipo: Literal['rotacionArmonica'],
        params: RotacionArmonicaParams,
        rutaBase: str | None = None,
        formatoGuardado: Literal['npz', 'cuerpoRigidoCSV'] | None = 'npz',
) -> GeneradorCinematica: ...
@overload
def cinematica(
        tipo: Literal['traslacionArmonica'],
        params: TraslacionArmonicaParams,
        rutaBase: str | None = None,
        formatoGuardado: Literal['npz', 'cuerpoRigidoCSV'] | None = 'npz',
) -> GeneradorCinematica: ...
@overload
def cinematica(
        tipo: Literal['cuerpoRigidoCSV'],
        params: CuerpoRigidoCSVParams,
        rutaBase: str | None = None,
        formatoGuardado: Literal['npz', 'cuerpoRigidoCSV'] | None = 'npz',
) -> GeneradorCinematica: ...
@overload
def cinematica(
        tipo: Literal['npz'],
        params: NPZParams,
        rutaBase: str | None = None,
        formatoGuardado: Literal['npz', 'cuerpoRigidoCSV'] | None = 'npz',
) -> GeneradorCinematica: ...

@overload
def cinematica(
        tipo: Literal['aeroelasticidad'],
        params: AeroelasticidadParams,
) -> GeneradorCinematicaAeroelasticidad: ...

def cinematica(
        tipo: Literal['aoa', 'mru', 'custom', 'rotacionArmonica', 'traslacionArmonica', 'cuerpoRigidoCSV', 'npz', 'aeroelasticidad'],
        params: AOAParams | MRUParams | CustomParams| RotacionArmonicaParams | TraslacionArmonicaParams | CuerpoRigidoCSVParams | NPZParams| AeroelasticidadParams,
        rutaBase: str | None = None,
        formatoGuardado: Literal['npz', 'cuerpoRigidoCSV'] | None = 'npz',
) -> GeneradorCinematica | GeneradorCinematicaAeroelasticidad:
    """
    Genera un iterador con la cinemática solicitada, paso a paso.

    Parameters
    ----------
    tipo : {'mru', 'rotacionArmonica', 'traslacionArmonica', 'npz', 'aeroelasticidad'}
        Especifica el tipo de cinemática:
        - `'aoa'`: MRU, pero sin desplazarse y para varios ángulo de ataque (usa `Tipos.AOAParams`).
        - `'mru'`: movimiento rectilíneo uniforme (usa `Tipos.MRUParams`).
        - `'custom'`: movimiento arbitrario a partir de arreglos (usa `Tipos.CustomParams`).
        - `'rotacionArmonica'`: rotación armónica (usa `Tipos.RotacionArmonicaParams`).
        - `'traslacionArmonica'`: traslación armónica (usa `Tipos.TraslacionArmonicaParams`).
        - `'cuerpoRigidoCSV'`: movimiento de cuerpo rígido con cinemática de terna solidaria al cuerpo en un .csv (usa `Tipos.CuerpoRigidoCSVParams`).
        - `'npz'`: lee datos desde un .npz de Numpy (usa `Tipos.NPZParams`).
        - `'aeroelasticidad'`: cinemática con efectos aeroelásticos (usa `Tipos.AeroelasticidadParams`).
    params : Tipos.MRUParams | Tipos.RotacionArmonicaParams | Tipos.TraslacionArmonicaParams | Tipos.NPZParams| Tipos.AeroelasticidadParams
        Diccionario tipado de configuración según tipo. Se recomienda instanciarlo para 
        ver los parámetros y sus significados.
    rutaBase : str or None, optional
        Ruta (sin extensión) donde guardar el .npz o .csv generado. Solo se aplica si `tipo != 'aeroelasticidad'`.
        Por defecto, `None` (no guarda archivos).

    formatoGuardado {'npz', 'cuerpoRigidoCSV'}
        Especifica el formato en el que se guardará la cinemática generada.

    Returns
    -------
    gen : Tipos.GeneradorCinematica | Tipos.GeneradorCinematicaAeroelasticidad
        Generador que produce en cada paso:
        - Un objeto `Tipos.CinematicaSalida`. 
        - Para `'aeroelasticidad'`, puede aceptar mediante `.send()` una tupla
            `(mp2d, iS, it)` que se usa internamente en la siguiente iteración. 

    Raises
    ------
    ValueError
        Si `tipo` no coincide con ninguno de los soportados.

    Notes
    -----
    - La estructura y campos de `Tipos.CinematicaSalida` están definidos en `Tipos`.

    See Also
    -------
    Para mayor comprensión, se recomienda leer:
    - Ejemplos/ejemploAlaFlap.ipynb
    - Ejemplos/ejemploCinematicaAeroelastica.ipynb
    - Ejemplos/ejemploDesdeNPZ.ipynb
    """

    if tipo != 'aeroelasticidad':
        if tipo == 'aoa':
            t, RO, theta, VO, w, r_xy, VRelPC_xy = _AOA(**params)
        elif tipo == 'mru':
            t, RO, theta, VO, w, r_xy, VRelPC_xy = _MRU(**params)
        elif tipo == 'custom':
            t, RO, theta, VO, w, r_xy, VRelPC_xy = _custom(**params)
        elif tipo == 'rotacionArmonica':
            t, RO, theta, VO, w, r_xy, VRelPC_xy = _rotacionArmonica(**params)
        elif tipo == 'traslacionArmonica':
            t, RO, theta, VO, w, r_xy, VRelPC_xy = _traslacionArmonica(**params)
        elif tipo == 'cuerpoRigidoCSV':
            t, RO, theta, VO, w, r_xy, VRelPC_xy = _cuerpoRigidoCSV(**params)
        elif tipo =='npz':
            t, RO, theta, VO, w, r_xy, VRelPC_xy = _desdeNPZ(**params)
        
        if isinstance(rutaBase, str):
            if formatoGuardado == 'npz':
                _guardarCinematicaNPZ((t, RO, theta, VO, w, r_xy, VRelPC_xy), rutaBase)
            elif formatoGuardado == 'cuerpoRigidoCSV':
                _guardarCinematicaCSV((t, RO, theta, VO, w, r_xy, VRelPC_xy), rutaBase)
        
        return _cinematicaGen(t, RO, theta, VO, w, r_xy, VRelPC_xy)

    elif tipo == 'aeroelasticidad':
        return _cinematicaAeroelasticidadGen(**params)
    else:
        raise ValueError(f"Tipo de cinemática no reconocido: {tipo}")
    
def _cinematicaGen(
        t: np.ndarray,      # (T,)
        RO: np.ndarray,     # (T, 2, 1)
        theta: np.ndarray,  # (T,) 
        VO: np.ndarray,     # (T, 2, 1)
        w: np.ndarray,      # (T,)
        r_xy: np.ndarray,   # (T, 2, N+1)
        VRelPC_xy: np.ndarray,  # (T, 2, N)
) -> GeneradorCinematica:
    for i, tt in enumerate(t):
        yield tt, RO[i,:,:], theta[i], VO[i,:,:], w[i], r_xy[i,:,:], VRelPC_xy[i,:,:]

def _cinematicaAeroelasticidadGen(
        actualizadorCinematica: ActualizadorCinematica
) -> GeneradorCinematicaAeroelasticidad:
    info = None
    while True:
        if info is not None:
            mp, iS, it = info
            info = yield actualizadorCinematica(mp, iS, it)
        else:
            info = yield(actualizadorCinematica(None, None, None))

def _MRU(
        t: np.ndarray,      # (T,)
        r_xy: np.ndarray,   # (2, N+1) | # # (T, 2, N+1)
        alfa: float, 
        V: float,
        VRelPC_xy: np.ndarray | None = None, # (T, 2, N+1)
) -> CinematicaCompletaSalida: 
    V = - np.array([[V], [0]])
    theta = - np.deg2rad(alfa) 

    return _custom(t=t, r_xy=r_xy, VO=V, theta=theta, VRelPC_xy=VRelPC_xy)

def _AOA(
        r_xy: np.ndarray,   # (2, N + 1)
        alfa: ArrayLike,   # (N_a,)
        V: float,
) -> CinematicaCompletaSalida:
    alfa = np.array(alfa)

    t = alfa

    theta = - np.deg2rad(alfa)
    V = - np.array([[V], [0]])
    RO = np.array([[0], [0]])
    
    return _custom(t=t, r_xy=r_xy, VO=V, RO=RO, theta=theta)

def _custom(
    t: np.ndarray,                      # (T,)
    r_xy: np.ndarray,                   # (2, N + 1) | (T, 2, N + 1)
    VO: np.ndarray,                     # (2, 1) | (T, 2, 1)
    w: np.ndarray = None,               # (T,)
    RO: np.ndarray = None,              # (2, 1) | (T, 2, 1)
    theta:np.ndarray = None,            # (T,)
     VRelPC_xy: np.ndarray = None,      # (T, 2, N)
) -> CinematicaCompletaSalida:
    
    if r_xy.ndim == 2:
        r_xy = np.repeat(r_xy[None, :, :], len(t), axis=0)            # (T, 2, N + 1)
    
    if VO.ndim == 2:
        VO = np.repeat(VO[None, :, :], len(t), axis=0)
    
    if w is None:
        w = np.zeros_like(t, dtype=float)
    
    if RO is None:
        dt = np.diff(t, prepend=t[0])
        RO = np.empty((len(t), 2, 1), dtype=float)
        RO[0] = 0.
        RO[1:] = np.cumsum(VO[:-1] * dt[1:, None, None], axis=0)
    elif RO.ndim == 2:
        RO = np.repeat(RO[None, :, :], len(t), axis=0)
        
    if theta is None:
        dt = np.diff(t, prepend=t[0])
        theta = np.empty_like(t, dtype=float)
        theta[0] = 0.
        theta[1:] = np.cumsum(w[:-1] * dt[1:])
    elif np.isscalar(theta):
        theta = np.repeat(theta, len(t))    
    
    if VRelPC_xy is None:
        VRelPC_xy = np.zeros((len(t), 2, r_xy.shape[2] - 1), dtype=float)    # (T, 2, N)

    return t, RO, theta, VO, w, r_xy, VRelPC_xy

def _rotacionArmonica(
        t: np.ndarray,      # (T,)
        r_xy: np.ndarray,   # (2, N+1)
        a0: float,
        V: float,
        w: float,
        VRelPC_xy: np.ndarray = None,
) -> CinematicaCompletaSalida:
    
    V = - np.array([[V], [0]]) 

    thetaMax = -np.deg2rad(a0)

    theta = thetaMax * np.sin(w * t)
    w = thetaMax * w * np.cos(w * t)

    return _custom(t=t, r_xy=r_xy, VO=V, w=w, theta=theta)

def _traslacionArmonica(
        t: np.ndarray,      # (T,)
        r_xy: np.ndarray,   # (2, N + 1)
        h0: float,
        V: float,
        w: float,
        VRelPC_xy: np.ndarray | None = None,
) -> CinematicaCompletaSalida:
    
    V = - np.array([[V], [0]]) 
    RO = V[None,:,:] * t[:,None,None]           # (T, 2, 1)

    VO = np.repeat(V[None,:,:], len(t), axis=0) # (T, 2, 1)

    h0 = - h0     # Signo menos por definición del NACA TN 2465
    h = h0 * np.sin(w * t)
    hdot = h0 * w * np.cos(w * t)

    RO = RO + np.stack([np.zeros_like(h), h], axis=1)[:,:,None]
    VO = VO + np.stack([np.zeros_like(hdot), hdot], axis=1)[:,:,None]

    return _custom(t=t, r_xy=r_xy, VO=VO, RO=RO)

def _cuerpoRigidoCSV(
        r_xy: np.ndarray,   # (2, N + 1)
        rutaBase: str,
) -> CinematicaCompletaSalida:
    ruta = rutaBase + '.csv'
    data = np.genfromtxt(ruta, delimiter=',', names=True, dtype=float, encoding='utf-8')
    
    t = np.asarray(data['t'], dtype=float) 
    theta = np.asarray(data['theta'], dtype=float) 
    w = np.asarray(data['w'], dtype=float) 
    RO_X = np.asarray(data['RO_X'], dtype=float) 
    RO_Y = np.asarray(data['RO_Y'], dtype=float) 
    VO_X = np.asarray(data['VO_X'], dtype=float) 
    VO_Y = np.asarray(data['VO_Y'], dtype=float) 

    RO = np.column_stack((RO_X, RO_Y))[:, :, None]  # (T, 2, 1)
    VO = np.column_stack((VO_X, VO_Y))[:, :, None]  # (T, 2, 1)

    return _custom(t, r_xy, VO, w, RO, theta)

def _desdeNPZ(
        rutaBase: str,
) -> CinematicaCompletaSalida:
    data = np.load(rutaBase + '.npz', allow_pickle=False)

    t = data["t"]                       # (T,)
    RO = data["RO"]                     # (T, 2, 1)
    theta = data["theta"]               # (T,)
    VO = data["VO"]                     # (T, 2, 1)
    w = data["w"]                       # (T,)
    r_xy = data["r_xy"]                 # (T, 2, N+1)
    VRelPC_xy = data["VRelPC_xy"]       # (T, 2, N)
    return t, RO, theta, VO, w, r_xy, VRelPC_xy

def _guardarCinematicaCSV(
        cinematica: CinematicaCompletaSalida,
        rutaBase: str = 'cinematica'
):
    
    t, RO, theta, VO, w, r_xy, VRelPC_xy = cinematica

    RO = RO.reshape(-1, 2)
    VO = VO.reshape(-1, 2)

    ruta = rutaBase + '.csv'
    
    with open(ruta, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        writer.writerow(['t', 'RO_X', 'RO_Y', 'theta', 'VO_X', 'VO_Y', 'w'])
        
        for i in range(len(t)):
            writer.writerow([t[i], RO[i, 0], RO[i, 1], theta[i], VO[i, 0], VO[i, 1], w[i]])

def _guardarCinematicaNPZ(
    cinematica: CinematicaCompletaSalida,
    rutaBase: str = "cinematica"
):
    t, RO, theta, VO, w, r_xy, VRelPC_xy = cinematica

    np.savez(
        rutaBase + '.npz',
        t=np.asarray(t, dtype=float),
        RO=np.asarray(RO, dtype=float),
        theta=np.asarray(theta, dtype=float),
        VO=np.asarray(VO, dtype=float),
        w=np.asarray(w, dtype=float),
        r_xy=np.asarray(r_xy, dtype=float),
        VRelPC_xy=np.asarray(VRelPC_xy, dtype=float),
    )

# =============================================================================
# Cinemática para definir el punto de toma de momentos
# =============================================================================
@overload
def RM(
    tipo: Literal['RM'],
    params: RMParams,
) -> GeneradorRM: ...
@overload
def RM(
        tipo: Literal['desdeCinematicaRO'],
        params: RMDesdeCinematicaROParams,
) -> GeneradorRM: ...
@overload
def RM(
        tipo: Literal['csv'],
        params: RMCSVParams,
) -> GeneradorRM: ...
@overload
def RM(
        tipo: Literal['aeroelasticidad'],
        params: RMAeroelasticidadParams,
) -> GeneradorRMAeroelasticidad: ...

def RM(
        tipo: Literal['RM', 'desdeCinematicaRO', 'csv', 'aeroelasticidad'],
        params: RMParams | RMDesdeCinematicaROParams | RMCSVParams| RMAeroelasticidadParams
) -> GeneradorRM | GeneradorRMAeroelasticidad:
    """
    Genera un iterador de vectores RM para el cálculo del punto de toma de momentos
    para el cálculo de momentos aerodinámicos.

    Parameters
    ----------
    tipo : {'RM', 'desdeCinematicaRO', 'csv', 'aeroelasticidad'}
        Modo de obtención de RM:
        - 'RM': a partir de la cinemática definida en un arreglo o el origen.
        - 'desdeCinematicaRO': extrae `RM` del vector de origen `RO` de un generador 
        obtenido con `_Cinematicas.cinematica`.
        - 'csv': a partir de un .csv.
        - 'aeroelasticidad': calcula RM mediante la función de actualización
          de aeroelasticidad brindada.
    params : Tipos.RMParams | Tipos.RMDesdeCinematicaROParams | Tipos.RMCSVParams | Tipos.RMAeroelasticidadParams
        Diccionario tipado de configuración según tipo. Se recomienda instanciarlo para 
        ver los parámetros y sus significados.
        
        Puede ser:
        - `Tipos.RMParams` para `'RM'`.
        - `Tipos.RMDesdeCinematicaROParams` para `'desdeCinematicaRO'`.
        - `Tipos.RMCSVParams` para `'csv'`.
        - `Tipos.RMAeroelasticidadParams` para `'aeroelasticidad'`.

    Returns
    -------
    gen : Tipos.GeneradorRM | Tipos.GeneradorRMAeroelasticidad
        Generador que produce en cada paso:
        - Un array `RM` de forma (2, 1) con la posición de origen `RO` de la cinemática
        brindada.
        - Para `'aeroelasticidad'`, puede aceptar mediante `.send()` una tupla
            `(mp2d, it)` que se usa internamente en la siguiente iteración. 

    Raises
    ------
    ValueError
        Si `tipo` no es uno de los valores soportados.

    See Also
    -------
    Para mayor comprensión, se recomienda leer:
    - Ejemplos/ejemploAlaFlap.py
    - Ejemplos/ejemploCinematicaAeroelastica.py
    - Ejemplos/ejemploDesdeNPZ.py
    """

    if tipo != 'aeroelasticidad':
        if tipo == 'RM':
            RM = _RM(**params)
        elif tipo == 'desdeCinematicaRO':
            RM = _RMDesdeRO(**params)
        elif tipo =='csv':
            RM = _RMCSV(**params)

        return _RMGen(RM)
    elif tipo == 'aeroelasticidad':
        return _RMAeroelasticidadGen(**params)
    
    else:
        raise ValueError(f"Tipo de cinemática no reconocido: {tipo}")
    
def _RMGen(
        RM: np.ndarray,     # (T, 2, 1)
) -> GeneradorRM:
    for i in range(RM.shape[0]):
        yield RM[i,:,:]

def _RM(
        RM: np.ndarray | None = None,     # (T, 2, 1) o (2, 1)
        rep: int = 1,
) -> np.ndarray:
    if RM is None:
        RM = np.array([[0], [0]])
    if RM.ndim == 2:
        RM = np.repeat(RM[None, :, :], repeats=rep, axis=0)
    return RM

def _RMDesdeRO(
        cinematica: Generator[Tuple[float, np.ndarray, float, np.ndarray, float, np.ndarray, np.ndarray], None, None]
) -> np.ndarray:
    RO = np.stack([RO for _, RO, *_ in cinematica], axis=0)
    return RO

def _RMCSV(
        rutaBase: str,     
) -> np.ndarray:

    ruta = rutaBase + '.csv'
    data = np.genfromtxt(ruta, delimiter=',', names=True, dtype=float, encoding='utf-8')
    
    RO_X = np.asarray(data['RO_X'], dtype=float) 
    RO_Y = np.asarray(data['RO_Y'], dtype=float) 

    RO = np.column_stack((RO_X, RO_Y))[:, :, None]  # (T, 2, 1)

    return RO

def _RMAeroelasticidadGen(
        actualizadorRM: ActualizadorRM
) -> GeneradorRMAeroelasticidad:
    info = None
    while True:
        if info is not None:
            mp, it = info
            info = yield actualizadorRM(mp, it)
        else:
            info = yield actualizadorRM(None, None)

# =============================================================================
# Operación de coordenadas
# =============================================================================

def moverCoordenadas(
        r_xy: np.ndarray | None = np.array([[0], [0]]), 
        Dx: float | None = 0., 
        Dy: float | None = 0., 
        Dth: float | None = 0.,
) -> np.ndarray:
    '''
    Función para mover un perfil con una traslación horizontal y un vertcial, así como con una rotación.

    Parameters
    ----------
    r_xy : np.ndarray shape (2, N + 1)
        Coordenadas originales del perfil. Por defecto, es el origen.
    Dx : float
        Cantidad a desplazar a lo largo del versor e_x. Por defecto, `0.0`
    Dy : float
        Cantidad a desplazar a lo largo del versor e_y. Por defecto, `0.0`
    Dth : float
        Cantidad a rotar [°], positivo antihorario. Por defecto, `0.0`

        .. note::
            Se rota luego de desplazar.
    
    Returns
    -------
    r_xy_mod : np.ndarray shape (2, N + 1)
        Coordenadas modificadas.
    '''
    Dth = np.deg2rad(Dth)
    TM = TernasMoviles2D(np.array([[Dx], [Dy]]), np.array([Dth]))
    return TM.r2R_1TM(r_xy, 0)


def flap(FlapParams: FlapParams) -> np.ndarray: 
    ''' 
    Función para mover un perfil como un flap según las definiciones de NACA Report No. 614 (19930091603) 
    y obtener sus nuevas coordenadas xy.

    Notes
    -----
    - Se asume que el BA del perfil principal está en (0,0).
    - Se asume que el perfil principal tiene cuerda media paralela al eje x. Si luego se quiere rotar al conjunto 
    se utiliza una terna externa.

    Parameters
    ----------
    FlapParams : Tipos.FlapParams
        Diccionario tipado de configuración. Se recomienda instanciarlo para 
        ver los parámetros y sus significados.
    
    Returns
    -------
    r_xy : np.ndarray, shape (2, N)
        Coordenadas del perfil del flap.
    r_ROT_xy : np.ndarray, shape (2, 1)
        Coordenadas del eje de rotación del flap.
    '''
    r_xy, df, cw, cf, h_TEw_ROTf, v_TEw_ROTf, h_ROTf_BAf, v_ROTf_MCf = FlapParams.values()

    r_xy       = FlapParams['r_xy']
    df         = FlapParams['df']
    cw         = FlapParams['cw']
    cf         = FlapParams['cf']
    h_TEw_ROTf = FlapParams['h_TEw_ROTf']
    v_TEw_ROTf = FlapParams['v_TEw_ROTf']
    h_ROTf_BAf = FlapParams['h_ROTf_BAf']
    v_ROTf_MCf = FlapParams['v_ROTf_MCf']
    
    # Se desplaza el perfil para ubicar el punto de rotación en el origen.
    r_xy = moverCoordenadas(
        r_xy,
        Dx = - h_ROTf_BAf * cf,
        Dy = v_ROTf_MCf * cf,
    )

    # Se desplaza el centro de rotación a la posición final y se rota el perfil según la deflexión.
    r_ROT_xy = np.array([[(1 + h_TEw_ROTf) * cw], [- v_TEw_ROTf * cw]])
    r_xy = moverCoordenadas(
        r_xy,
        Dx = r_ROT_xy[0, 0],
        Dy = r_ROT_xy[1, 0],
        Dth = - df
    )
    return r_xy, r_ROT_xy
    