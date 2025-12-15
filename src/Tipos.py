# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# =======================================================================================
# Proyecto: Trabajo Final - Método de Paneles Bidimensional Multielemento No estacionario
# Archivo: src/Tipos.py
# Autor: Bruno Francisco Barra Atarama
# Institución: 
#   Departamento de Ingeniería Aeroespacial
#   Facultad de Ingeniería
#   Universidad Nacional de La Plata
# Año: 2025
# Licencia: PolyForm Noncommercial License 1.0.0
# =======================================================================================
"""
Módulo donde se compendian diferentes objetos tipados utilizados en otros módulos.

Su fin es organizador.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Tuple, Protocol, TypedDict, NotRequired, Generator, Literal, TypeAlias, List, Optional, Callable

if TYPE_CHECKING:
    from .MP2D import MP2D

# =============================================================================
# Tipos
# =============================================================================

CinematicaSalida: TypeAlias = Tuple[
    float, np.ndarray, float, np.ndarray, float, np.ndarray, np.ndarray
]
"""
Salida de cinemática en un instante:
- t : float. Tiempo de simulación.
- RO : np.ndarray shape (2, 1). Origen de la terna móvil respecto del de la de referencia.
- theta : float. Ángulo de la terna móvil respecto de la de referencia.
- VO : np.ndarray shape (2, 1). Velocidad del origen de la terna móvil.
- w : float. Velocidad angular de la terna móvil.
- r_xy : np.ndarray shape (2, N + 1). Coordenadas de los nodos referidas a la terna móvil.
- VRelPC_xy : np.ndarray shape (2, N). Velocidades relativas en los puntos de colocación referidas a la terna móvil.
"""

CinematicaCompletaSalida: TypeAlias = Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]
"""
Cinemática de todos los instantes:
- t : np.ndarray shape (T,). Tiempos de simulación.
- RO : np.ndarray shape (T, 2, 1). Orígenes de la terna móvil respecto del de la de referencia.
- theta : np.ndarray shape (T,). Ángulos de la terna móvil respecto de la de referencia.
- VO : np.ndarray shape (T, 2, 1). Velocidades del origen de la terna móvil.
- w : np.ndarray shape (T,). Velocidades angulares de la terna móvil.
- r_xy : np.ndarray shape (T, 2, N + 1). Coordenadas de los nodos referidas a la terna móvil.
- VRelPC_xy : np.ndarray shape (2, N). Velocidades relativas en los puntos de colocación referidas a la terna móvil.
"""

GeneradorCinematica : TypeAlias = Generator[CinematicaSalida, None, None]
"""
Generador que produce en cada paso un objeto `Tipos.CinematicaSalida`. 
"""

GeneradorCinematicaAeroelasticidad : TypeAlias = Generator[CinematicaSalida, Optional[Tuple['MP2D', int, int]], None]
"""
Generador que produce en cada paso un objeto `Tipos.CinematicaSalida` y que puede
aceptar mediante `.send()` una tupla `(mp2d, iS, it)` que se usa internamente en la siguiente iteración.

Se usa conjuntamente con el protocolo `Tipos.ActualizadorCinematica`.
"""

GeneradorRM : TypeAlias = Generator[np.ndarray, None, None]
"""
Generador que produce en cada paso un vector RM para la toma de momentos. 
"""

GeneradorRMAeroelasticidad : TypeAlias = Generator[Optional[np.ndarray], Optional[Tuple['MP2D',int]], None]
"""
Generador que produce en cada paso un vector RM para la toma de momentos y que puede
aceptar mediante `.send()` una tupla `(mp2d, it)` que se usa internamente en la siguiente iteración.

Se usa conjuntamente con el protocolo `Tipos.ActualizadorRM`.
"""
# =============================================================================
# Protocolos
# =============================================================================
class ActualizadorCinematica(Protocol):
    """
    Protocolo para funciones que generan y actualizan la cinemática aeroelástica
    en cada paso de la simulación.

    Flujo de invocación
    -------------------
    1. Inicialización del generador(purga en `__post_init__` de `MP2D.MP2D`):
        - `mp = None`  
        - `iS = None`  
        - `it = None`  
        Se configura el estado interno y devuelve `None`.

    2. Pasos de simulación:
        - `mp = instancia de MP2D.MP2D`  
        - `iS = índice del sólido (0 <= iS <= N_S)`  
        - `it = índice del instante de simulación (int >= 0)`  
        Se devuelve la cinemática correspondiente a ese instante.

    Parameters
    ----------
    mp : MP2D.MP2D | None
        Instancia de la simulación. Será `None` solo en la invocación de purga.
    iS : int | None
        Índice del sólido. Será `None` solo en la invocación de purga.
    it : int | None
        Índice del instante de simulación. Será `None` solo en la invocación de purga.

    Returns
    -------
    cin : Tipos.CinematicaSalida
        Cinemática en ese instante

    Notes
    -----
    - En `MP2D.MP2D.__post_init__`, se purga cada generador con `next(...)` para
      ejecutar la llamada inicial con `(None, None, None)`.  
    - En cada paso de simulación, `MP2D.MP2D.avanzar` hace `.send((mp, iS, it))`.  
    - Para diferenciar la purga de los pasos posteriores, dentro de la función
      se puede usar `if mp is None`.  
    """
    def __call__(self, mp: None, iS: None, it: None) -> CinematicaSalida: ...
    def __call__(self, mp: MP2D, iS: int, it: int) -> CinematicaSalida: ...

class ActualizadorRM(Protocol):
    """
    Protocolo para funciones que generan y actualizan el vector RM
    (punto de toma de momentos) en cada paso de la simulación.

    Flujo de invocación
    -----------------------
    1. Inicialización del generador (purga en `MP2D.MP2D.__post_init__`):
        - `mp = None`
        - `it = None`  
        Se configura el estado interno y devuelve None.
    
    2. Paso de simulación (incluido `it = 0` en modo aeroelástico):
        - `mp = instancia de MP2D.MP2D`   
        - `it = índice del instante de simulación (int >= 0)`  
        Se devuelve el vector `RM` en ese instante.

    Parameters
    ----------
    mp : MP2D.MP2D | None
        Instancia de la simulación. Será `None` solo en la invocación de purga.

        El actualizador es invocado luego de resolver las intensidades y antes de los
        cálculos aerodinámicos, por lo que la instancia `mp` se encuentra en un estadío intermedio,
        donde toda la cinemática de sólidos y estelas están definidas pero aun no las presiones, fuerzas y momentos.

        Por eso, debe tenerse precaución a la hora de indexar con `-1` de listas temporales.
    it : int | None
        Índice del instante de simulación. Será `None` solo en la invocación de purga.

    Returns
    -------
    np.ndarray | None
        - `None` en la llamada inicial (`mp=None`, `it=None`).  
        - `np.ndarray` de forma (2, 1) con las coordenadas RM en cada paso subsecuente.

    Notes
    -----
    - En cada paso de simulación, `MP2D.MP2Davanzar` hace `.send((mp, iS, it))`.
    - En `MP2D.MP2D.__post_init__`, se purga cada generador con `next(...)` para
    ejecutar la llamada inicial con `(None, None, None)`.  
    - Para diferenciar la purga de los pasos posteriores, dentro de la función
    se puede usar `if mp is None`.   
    """
    def __call__(self, mp: None, it: None) -> None: ...
    def __call__(self, mp: MP2D, it: int) -> np.ndarray: ...

# =============================================================================
# Diccionarios tipados
# =============================================================================

# Cinemática.py

class MRUParams(TypedDict):
    """
    Diccionario tipado de configuración de cinemática.

    Cinemática:

    Movimiento rectilíneo uniforme con velocidad de traslación (-`V`, 0) y que parte desde el origen
    del sistema coordenado fijo XY.

    El perfil está fijo a una terna móvil xy cuyos ejes coinciden con la terna fija XY cuando no hay rotación ni traslación. 

    Parameters
    ----------
    t : np.ndarray shape (T,)
        Instantes de simulación.
    r_xy : np.ndarray shape (2, N + 1) | (T, 2, N + 1)
        Coordenadas de (x, y) del perfil en el sistema coordenado solidario a la terna móvil.
        - En el caso de que sea un arreglo de dos dimensiones, estas serán las coordenadas para 
        todo instante (movimiento rígido).
        - Si fuera uno de tres, cada `r_xy[it, :, :]` serán las coordeandas para el instante `t[it]`
    alfa : float
        Ángulo de ataque (°), positivo horario. El eje de rotación es el origen de la terna móvil
    V : float
        Velocidad escalar de avance. Con esta se construye la velocidad vectorial como (-`V`, 0).
    VRelPC_xy : np.ndarray shape (T, 2, N), optional
        Velocidades relativas y en el sistema coordenado solidario a la terna mólvil, de los puntos de colocación
        del sólido.
        - Si shape (T, 2, N), es un arreglo tal que cada `VRelPC_xy[it, :, :]` sean la velocidades en cuestión
        en el instante t[it].
        - Por defecto, se consideran nulas estas velocidades (movimiento rígido) e internamente se trata este caso.
    """
    t: np.ndarray           # (T,)
    r_xy: np.ndarray        # (2, N + 1) o (T, 2, N + 1)
    alfa: float      
    V: float
    VRelPC_xy: np.ndarray   # (T, 2, N)

class AOAParams(TypedDict):
    """
    Diccionario tipado de configuración de cinemática.

    Cinemática:

    Sólido con diferentes ángulos de ataques pero sin desplazarse, como si en cada posición estuviera animado con un movimiento 
    rectilíneo uniforme con velocidad de traslación (-`V`, 0). Está pensado para experimentos estacionarios.
    
    El perfil está fijo a una terna móvil xy cuyos ejes coinciden con la terna fija XY cuando no hay rotación ni traslación. 

    Parameters
    ----------
    r_xy : np.ndarray shape (2, N + 1)
        Coordenadas de (x, y) del perfil en el sistema coordenado solidario a la terna móvil.
        - En el caso de que sea un arreglo de dos dimensiones, estas serán las coordenadas para 
        todo instante (movimiento rígido).
        - Si fuera uno de tres, cada `r_xy[it, :, :]` serán las coordeandas para el instante `t[it]`
    alfa : ArrayLike, shape (N_a,)
        Ángulo de ataque (°), positivo horario. El eje de rotación es el origen de la terna móvil
    V : float
        Velocidad escalar de avance. Con esta se construye la velocidad vectorial como (-`V`, 0).
    """
    r_xy: np.ndarray    # (2, N + 1) o (T, 2, N + 1)
    alfa: float         # (N_a,)
    V: float

class CustomParams(TypedDict):
    """
    Diccionario tipado de configuración de cinemática.

    Cinemática:

    Configurable a partir de arreglos y comportamientos por defecto.

    Parameters
    ----------
    t : np.ndarray shape (T,)
        Instantes de simulación.
    r_xy : np.ndarray shape (2, N + 1) | (T, 2, N + 1)
        Coordenadas de (x, y) del perfil en el sistema coordenado solidario a la terna móvil.
        - En el caso de que sea un arreglo de dos dimensiones, estas serán las coordenadas para 
        todo instante (movimiento rígido).
        - Si fuera uno de tres, cada `r_xy[it, :, :]` serán las coordeandas para el instante `t[it]`
    VO : np.ndarray shape (2, 1) | (T, 2, 1)
        Velocidad (U, V) del origen de la terna móvil.
        - Si shape (2, 1), se repite esta velocidad en los `T` instantes.
        - Si shape (T, 2, 1), `VO[it,:]` es la velocidad para el instante `t[it]`.
    w : np.ndarray shape (T,), optional
        Velocidades angulares de la terna móvil, positivas antihorario, en cada instante de simulación. El eje de rotación es el origen de la terna móvil
        - Si shape (T,), `w[it]` es la velocidad angular en el instante `t[it]`.
        - Por defecto, es nula en todo instante.
    RO : np.ndarray shape (2, 1) | (T, 2, 1), optional
        Posiciones (X, Y) del origen de la terna móvil.
        - Si shape (T, 2, 1), `RO[it, :, :]` es la posición en el instante `t[it]`.
        - Si shape (2, 1), se repite esta posición en los `T` instantes. Usar esto solo en el modo estacionario.
        - Por defecto, se obtiene a por integración directa de Euler de la velocidad, desde `RO[0, :, :] = (0, 0)`.
    theta : float or np.ndarray shape (T,), optional
        Actitud de la terna móvil solidaria al sólido, positiva antihorario. El eje de rotación es el origen de la terna móvil
        - Si es float, se repite para los `T` instantes.
        - Si shape (T,), `theta[it]` es la actitud en el instante `t[it]`.
        - Por defecto, se obtiene por integración directa de Euler de la velocidad angular, desde `theta[0] = 0`.
        
        .. note::
            Si la velocidad angular ni la actitud se definen, la actitud resulta nula para los `T` instantes.
    VRelPC_xy : np.ndarray shape (T, 2, N), optional
        Velocidades relativas y en el sistema coordenado solidario a la terna mólvil, de los puntos de colocación
        del sólido.
        - Si shape (T, 2, N), es un arreglo tal que cada `VRelPC_xy[it, :, :]` sean la velocidades en cuestión
        en el instante t[it].
        - Por defecto, se consideran nulas estas velocidades (movimiento rígido) e internamente se trata este caso.
    """
    t: np.ndarray               # (T,)
    r_xy: np.ndarray            # (2, N + 1) | (T, 2, N + 1)
    VO: np.ndarray              # (2, 1) | (T, 2, 1)
    w: np.ndarray               # (T,)
    RO: np.ndarray              # (2, 1) | (T, 2, 1)
    theta:np.ndarray            # (T,)
    VRelPC_xy: np.ndarray       # (T, 2, N)

class RotacionArmonicaParams(TypedDict):
    """
    Diccionario tipado de configuración de cinemática.

    Cinemática:

    Movimiento rectilíneo armónico (sinusoidal) de rotación con velocidad de traslación (-`V`, 0) y que parte desde el origen
    del sistema coordenado fijo XY.

    El perfil está fijo a una terna móvil xy cuyos ejes coinciden con la terna fija XY cuando no hay rotación ni traslación. 

    Parameters
    ----------
    t : np.ndarray shape (T,)
        Instantes de simulación.
    r_xy : np.ndarray shape (2, N + 1) | (T, 2, N + 1)
        Coordenadas de (x, y) del perfil en el sistema coordenado solidario a la terna móvil.
        - En el caso de que sea un arreglo de dos dimensiones, estas serán las coordenadas para 
        todo instante (movimiento rígido).
        - Si fuera uno de tres, cada `r_xy[it, :, :]` serán las coordeandas para el instante `t[it]`
    V : float
        Velocidad escalar de avance. Con esta se construye la velocidad vectorial como (-`V`, 0).
    VRelPC_xy : np.ndarray shape (T, 2, N), optional
        Velocidades relativas y en el sistema coordenado solidario a la terna mólvil, de los puntos de colocación
        del sólido.
        - Por defecto, se consideran nulas estas velocidades (movimiento rígido) e internamente se trata este caso.
        - De lo contrario, debe ser un arreglo tal que cada `VRelPC_xy[it, :, :]` sean la velocidades en cuestión
        en el instante t[it].
    a0 : float
        Amplitud del ángulo de ataque (°), positivo horario. El eje de rotación es el origen de la terna móvil.
    w : float
        Velocidad angular del movimiento armónico.

    See Also
    --------
    En Ejemplos/testArmonico.py se utiliza este diccionario para configurar experimentos que simulen los realizados
    en el NACA Technical Note 2465 (19930092144).
    """
    t: np.ndarray           # (T,)
    r_xy: np.ndarray        # (2, N + 1) o (T, 2, N + 1)
    V: float
    VRelPC_xy: np.ndarray   # (T, 2, N)
    a0: float
    w: float
    
    
class TraslacionArmonicaParams(TypedDict):
    """
    Diccionario tipado de configuración de cinemática.

    Cinemática:

    Movimiento rectilíneo armónico (sinusoidal) de traslación vertical con velocidad de 
    traslación horizontal base (-`V`, 0) y que parte desde el origen del sistema coordenado fijo XY.

    El perfil está fijo a una terna móvil xy cuyos ejes coinciden con la terna fija XY cuando no hay rotación ni traslación. 

    Parameters
    ----------
    t : np.ndarray shape (T,)
        Instantes de simulación.
    r_xy : np.ndarray shape (2, N + 1) | (T, 2, N + 1)
        Coordenadas de (x, y) del perfil en el sistema coordenado solidario a la terna móvil.
        - En el caso de que sea un arreglo de dos dimensiones, estas serán las coordenadas para 
        todo instante (movimiento rígido).
        - Si fuera uno de tres, cada `r_xy[it, :, :]` serán las coordeandas para el instante `t[it]`
    V : float
        Velocidad escalar de avance. Con esta se construye la velocidad vectorial como (-`V`, 0).
    VRelPC_xy : np.ndarray shape (T, 2, N), optional
        Velocidades relativas y en el sistema coordenado solidario a la terna mólvil, de los puntos de colocación
        del sólido.
        - Por defecto, se consideran nulas estas velocidades (movimiento rígido) e internamente se trata este caso.
        - De lo contrario, debe ser un arreglo tal que cada `VRelPC_xy[it, :, :]` sean la velocidades en cuestión
        en el instante t[it].
    h0 : float
        Amplitud del desplazamiento vertical, positivo hacia +y.
    w : float
        Velocidad angular del movimiento armónico.

    See Also
    --------
    En Ejemplos/testArmonico.py se utiliza este diccionario para configurar experimentos que simulen los realizados
    en el NACA Technical Note 2465 (19930092144).
    """
    t: np.ndarray           # (T,)
    r_xy: np.ndarray        # (2, N + 1) o (T, 2, N + 1)
    V: float
    VRelPC_xy: np.ndarray   # (T, 2, N)
    h0: float
    w: float
class CuerpoRigidoCSVParams(TypedDict):
    """
    Diccionario tipado de configuración de cinemática.

    Cinemática:

    Se considera un sólido indeformable con un terna móvil solidaría a él y con origen coincidente al origen del marco fijo en el instante `t = 0`,
    y cuya descripción cinemática define el movimiento del cuerpo.

    Parameters
    ----------

    r_xy : np.ndarray shape (2, N + 1)
        Coordenadas (x, y) del perfil en el sistema coordenado solidario a la terna móvil.
    rutaBase : str
        Ruta al archivo .csv, sin incluir la extensión.
    
    Notes
    -----
    - El .csv debe tener **todas** las claves: t, RO_X, RO_Y, theta, VO_X, VO_Y y w.
    - No se valida la ausencia o el mal formato de ninguna columna.
    """
    r_xy: np.ndarray        #(2, N + 1)
    rutaBase: str

class NPZParams(TypedDict):
    """
    Diccionario tipado de configuración de cinemática.

    Cinemática:

    Aquella almacenada en un .npz con las claves t, RO, theta, VO, w, r_xy, VRelPC_xy.

    Parameters
    ----------
    rutaBase : str
        Ruta al archivo .npz, sin incluir la extensión.

    Notes
    -----
    - El .npz debe tener **todas** las claves: t, RO, theta, VO, w, r_xy y VRelPC_xy.
    - No se valida la ausencia o el mal formato de ninguna columna.
    """
    ruta: str

class AeroelasticidadParams(TypedDict):
    """
    Diccionario tipado de configuración de cinemática.

    Cinemática:

    Aquella que se defina en la función de cinemática particular utilizada,
    que siga con el protocolo de `Tipos.ActualizadorCinematica`

    Parameters
    ----------
    actualizadorCinematica : Tipos.ActualizadorCinematica
        Función que genera y actualiza la cinemática aeroelástica en cada paso de la simulación.
    """
    actualizadorCinematica: ActualizadorCinematica

class FlapParams(TypedDict):
    """
    Diccionario tipado para el posicionamiento de un flap.

    Los parámetros utilizados siguen las deficioniones utilizadas en el NACA Report No. 614 (19930091690).

    Parameters
    ----------
    r_xy : np.ndarray shape (2, N + 1)
        Coordenadas de (x, y) del perfil del flap. 
    df : float
        Deflexión del flap (°), positivo horario.
    cw : float
        Cuerda del perfil del ala.
    cf : float
        Cuerda del flap.

        .. note:: Las coordenadas en `r_xy` deben ser consistentes con esta cuerda.
    h_TEw_ROTf : float
        Distancia horizontal (paralela a la cuerda media del ala) entre el borde de fuga del ala y 
        el eje de rotación del flap. Positivo atrasa (+x) el flap.

        .. note::
            Adimensionalizada con `cw`.
    v_TEw_ROTf: float
        Distancia vertical (perpendicular a la cuerda media del ala) entre el borde de fuga del ala y
        el eje de rotación del flap. Positivo desciende (-y) el flap.
        
        .. note::
            Adimensionalizada con `cw`.
    h_ROTf_BAf: float
        Distanica paralela a la cuerda media del flap, entre el eje de rotación y el borde de ataque del flap. 
        Positivo adelanta (-x) el flap.
        
        .. note::
            Adimensionalizada con `cf`.
    v_ROTf_MCf: float
        Distancia perpendicular a la cuerda media del flap, entre el eje de rotación y la cueerda media del flap. 
        Positivo asciende (+y) el flap.
        
        .. note::
            Adimensionalizada con `cf`.
    """
    r_xy: np.ndarray
    df: float
    cw: float
    cf: float
    h_TEw_ROTf: float
    v_TEw_ROTf: float
    h_ROTf_BAf: float
    v_ROTf_MCf: float

class RMParams(TypedDict):
    """
    Diccionario tipado de configuración de la cinemática del vector de toma de momentos.

    Cinemática:

    Aquella que se defina en un arreglo de puntos de toma de momentos o el origen.

    Parameters
    ----------
    RM : np.ndarray shape (T, 2, 1) or (2, 1), optional
        Puede ser: 
        - Un arreglo con los puntos de toma de momento en cada uno de los `T` instantes.
        - Un vector, en cuyo caso se repite `rep` veces en el generador.
        - Por defecto, se repite el origen `rep` veces en el generador.

        Por defecto se utiliza la lógica de `None`.
    rep : int, optional
        Puede ser:
        - Número de veces que se repite un vector en el generador.
        - Por defecto, es `1` y se aplica solo en el caso de `RM` vector.
    """
    RM: NotRequired[np.ndarray]
    rep: NotRequired[int]

class RMDesdeCinematicaROParams(TypedDict):
    """
    Diccionario tipado de configuración de la cinemática del vector de toma de momentos.

    Cinemática:

    Aquella que se defina en la variable `RO` de un salida del tipo `Tipos.CinematicaSalida`, que puede ser
    generada con `Cinematicas.cinematica`.

    Parameters
    ----------
    cinematica : Tipos.GeneradorCinematica
        Generador de tuplas del tipo `Tipos.CinematicaSalida, que puede ser
        generado con `Cinematicas.cinematica`.
    """
    cinematica: GeneradorCinematica
class RMCSVParams(TypedDict):
    """
    Diccionario tipado de configuración de la cinemática del vector de toma de momentos.

    Cinemática:

    Aquella que se defina en un .csv en las columnas `RO_X` y `RO_Y`.

    Parameters
    ----------
    rutaBase : str
        Ruta al archivo .csv, sin incluir la extensión.
    
    Notes
    -----
    - El .csv debe tener **todas** las claves: RO_X y RO_Y.
    - No se valida la ausencia o el mal formato de ninguna columna.
    """
    rutaBase: str
class RMAeroelasticidadParams(TypedDict):
    """
    Diccionario tipado de configuración de la cinemática del vector de toma de momentos.

    Cinemática:

    Aquella que se defina en la función de cinemática particular utilizada,
    que siga con el protocolo de `Tipos.ActualizadorCinematica`

    Parameters
    ----------
    actualizadorRM : Tipos.ActualizadorCinematica
        Función que genera y actualiza el vector RM (punto de toma de momentos) en cada paso de la simulación.
    """
    actualizadorRM: ActualizadorRM

# MP2D.py

class SELInfo(TypedDict):
    """
    Diccionario tipado para almacenar información de los SEL resueltos por el simulador.

    Parameters
    ----------
    SELSol: Literal['svd', 'qr', 'lu']
        Método de resolución del sistema de ecuaciones lineales.
    nCond: List[float]
        Lista con el número de condición de la matriz de coeficientes en cada instante.
    normaRes: List[float]
        Norma de los residuos de la solución en cada instante. `normaRes = ||Ax - b||`
    normaInd: List[float]
        Norma del vector independiente. `normaInd = ||b||`
    """
    SELSol: Literal['svd', 'qr', 'lu']
    nCond: List[float]
    normaRes: List[float]
    normaInd: List[float]

class MPConfig(TypedDict):
    """
    Diccionario tipado de configuración del simulador MP2D.MP2D.

    Configura los ajustes de simulación generales.

    Notes
    -----
    Si no se completa alguna clave, se completa con los valores por defecto.

    Parameters
    ----------
    nombres : List[str]
        Nombres de los sólidos modelados. Por defecto, se completan con `'Sólido i_S'` con `0<= i_S <= N_S`.
    estacionarios : bool
        - Si `True`, se simula un problema estacionario. 
        - Si `False`, uno no estacionario.

        Por defecto, `False`.
    vorticeArranque:
        - Si `True`, en el primer instante se considera un vórtice de arranque que permite cumplir con la condición de Kutta.
        En este caso, el primer instante es equivalente a comenzar en régimen estacionario.
        - Si `False`, en el primer instante se anula la intensidad del vórtce de arranque, por lo que la condición de Kutta no
        necesarimente se satisface.

        Por defecto, `True`.
    aeroelasticidad : bool
        - Si `True`, en cada paso se alimenta la cinemática con información de la simulación. 
        - Si `False`, se utiliza cinemática predefinida.

        .. note::

            Si se tienen múltiples sólidos, la configuración se aplica a todos.

        Por defecto, `False`.
    RMAeroelasticidad : bool
        - Si `True`, en cada paso se alimenta la cinemática de RM con información de la simulación. 
        - Si `False`, se utiliza cinemática de RM predefinida.

        Por defecto, `False`.
    rigidez : float
        Escalar que se espera en el rango 0 <= rigidez <= 1 y que regula la posición del último nodo de estela creado.
        - Si `rigidez == 1.0`, el nuevo nodo se ubica por integración directa de Euler de la velocidad 
        en el borde de fuga en el paso anterior.
        - Si `rigidez == 0`, el nuevo nodo se ubica en la posición anterior del borde de fuga.

        Por defecto `1.0`.
    mostrarProgreso : bool
        - Si `True`, se muestra el progreso con `tqdm`. 
        - Si `False`, no se muestra.

        Por defecto, `True`.
        .. note:: 
            En el caso no aeroelástico y cuando se va a mostrar el progreso, si no se especifica
            un número de instantes a simular, antes de la simulación el `Tipos.GeneradorCinematica` en la primera posición de MP2D.cinematicas 
            se recorre para la contabilización de los instantes disponibles, copiando sus valores para construir otro generador 
            para usar en la simulación.

            Esto se hace notar ya que este comportamiento podría comprometer la funcionalidad implementada en
            generadores aeroelásticos particulares.
    rotulo_t : str
        Rotulo asociado al vector `MP2D.t`, ya que este puede representar otra magnitud en experimentos no estacionarios.
        
        Por defecto, `'t'`.
    invertirCm: bool
        - Si `True`, los momentos calculados tienen el sentido definido por el eje z y la regla de la mano derecha.
        - Si `False`, se invierte el signo.

        Por defecto, `True`.
    SELInfo: 
        - Si `True`, se calcula y guarda información de los SEL resueltos.
        - Si `False`, se omite este calculo.

        Por defecto, `False`.

    SELSol: Literal['svd', 'qr', 'lu']
        Método de resolución del sistema de ecuaciones lineales.

        Por defecto, `qr`.
    """
    nombres: List[str] | None
    estacionario: bool
    vorticeArranque: bool
    aeroelasticidad: bool
    RMAeroelasticidad: bool
    rigidez: float
    mostrarProgreso: bool
    rotulo_t: str
    invertirCm: bool
    SELInfo: True
    SELSol: Literal['svd', 'qr', 'lu']

class RefConfig(TypedDict):
    """
    Diccionario tipado de configuración del simulador MP2D.MP2D.

    Configura las magnitudes de referencia.

    Notes
    -----
    Si no se completa alguna clave, se completa con los valores por defecto.

    Parameters
    ----------
    rho : float
        Densidad. Por defecto, `1.225`.
    P : float
        Presión. Por defecto, `101325.0`.
    V : float
        Velocidad. Por defecto, `1.0`.
    l : float
        Longitud. Por defecto, `1.0`.
    nu: float
        Viscosidad cinemática. Por defecto `1.5e-5`.
    a : float
        Velocidad del sonido. Por defecto `340.3`
    g : float
        Aceleración de la gravedad. Por defecto, `9.81`.
    Q : float
        Presión dinámica. 

        .. note::
            Se calcula automáticamente con las otras magnitudes de referencia.
    
        
        .. note::
            Se calcula automáticamente con las otras magnitudes de referencia.
    """
    rho: float
    P: float
    V: float
    l: float
    a: float
    nu: float
    g : float
    
    Q : NotRequired[float]


class PanConfig(TypedDict):
    """
    Diccionario tipado de configuración del simulador MP2D.MP2D.

    Configura ajustes de los paneles.

    Notes
    -----
    Si no se completa alguna clave, se completa con los valores por defecto.

    Parameters
    ----------
    singularidad : {`dobleteConstante`}
        Indica el tipo de singularidad utilizada. Por defecto, 'dobleteConstante'.

        .. note:: Actualmente solo está implementada `dobleteConstante`.
    condicion : {neumann}
        Indica la condición de borde utilizada. Por defecto, 'dobleteConstante'.
        
        .. note:: Actualmente solo está implementada `neumann`.
    xPC_x2_xyp : float
        Razón coordenada x_PC / coordenada x_2, en el sistema de coordenado de los paneles,
        que se utilizará.

        Debe ser `0 < xPC_x2_xyp < 1` y se recomienda no utilizar valores cercanos a los extremos.

        Por defecto, `0.5` (a la mitad del panel).
    """
    singularidad: Literal['dobleteConstante']
    condicion: Literal['neumann']
    xPC_x2_xyp: float

class ControlConfig(TypedDict):
    """
    Diccionario tipado de configuración del simulador MP2D.MP2D.

    Configura los controles que se realizan en tiempo de ejecución.

    Parameters
    ----------
    instantesDeSimulacion : bool
        Si `True`, en el caso no aeroelástico y si se hubiera especificado un número de instantes simulados,
        antes de la simulación el `Tipos.GeneradorCinematica` en la primera posición de MP2D.cinematicas se recorre 
        para la contabilización de los instantes disponibles, copiando sus valores para construir otro generador 
        para usar en la simulación.

        Esto se hace notar ya que este comportamiento podría comprometer la funcionalidad implementada en los 
        generadores aeroelásticos.

        Si 'False', se desactiva esta verificación. Esto puede evita comprometer la funcionalidad implementada 
        en generadores aeroelásticos particulares.
    """
    instantesDeSimulacion: bool

class AdimInfo(TypedDict):
    """"
    Diccionario tipado en el que se guardan las relaciones adimensionales calculadas a partir de las magnitudes de referencia.

    Parameters
    ----------
    Re : float
        Número de Reynolds.
    Ma : float
        Número de Mach.
    Fr : float
        Número de Froude.
    _Re : float
        1 / Re.
    _Fr2 : float 
        1 / Fr ** 2
    Ma2 : float
        Ma ** 2.
    Ma2_Re : float
        Ma ** 2 / Re.
    Ma2_Fr2 : float
        Ma ** 2 / Fr ** 2.
    """
    Re: float
    Ma: float
    Fr: float

    _Re: float
    _Fr2: float

    Ma2: float
    Ma2_Re: float
    Ma2_Fr2: float

# =============================================================================
# Configuraciones por defecto
# =============================================================================

def _MP2D_refConfig_defecto() -> RefConfig:
    return {
        'rho': 1.225,
        'P' : 101325.,
        'V' : 1.,
        'l' : 1.,
        'nu' : 1.5e-5,
        'a' : 340.3,
        'g' : 9.81, 
    }

def _MP2D_mpConfig_defecto() -> MPConfig:
    return {
        'nombres': None,
        'estacionario': False,
        'vorticeArranque' : True,
        'rigidez': 1.,
        'aeroelasticidad' : False,
        'RMAeroelasticidad' : False,
        'mostrarProgreso' : True,
        'rotulo_t' : 't, [s]',
        'invertirCm' : True,
        'SELInfo' : False,
        'SELSol': 'qr',
    }

def _MP2D_panConfig_defecto() -> PanConfig:
    return {
        'singularidad': 'dobleteConstante',
        'condicion'   : 'neumann',
        'xPC_x2_xyp': 0.5,
    }

def _MP2D_controlConfig_defecto() -> ControlConfig:
    return {
        'instantesDeSimulacion' : True
    }

def _completar_dic(dic: dict | None, dic_defecto: Callable[[], dict]) -> dict:
    dic_defecto: dict = dic_defecto().copy()
    if dic is not None:
        dic_defecto.update(dic)
    return dic_defecto

def _MP2D_SELInfo_defecto(mpConfig: MPConfig) ->SELInfo:
    if mpConfig['SELInfo']:
        res = {
            'nCond' : [],
            'normaRes' : [],
            'normaInd' : [],
        }
    else:
        res = {}

    res['SELSol'] = mpConfig['SELSol']
    return res

