# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# =======================================================================================
# Proyecto: Trabajo Final - Método de Paneles Bidimensional Multielemento No estacionario
# Archivo: src/_ConjuntosSolidos2D.py
# Autor: Bruno Francisco Barra Atarama
# Institución: 
#   Departamento de Ingeniería Aeroespacial
#   Facultad de Ingeniería
#   Universidad Nacional de La Plata
# Año: 2025
# Licencia: PolyForm Noncommercial License 1.0.0
# =======================================================================================

"""
Módulo para definir la clase `ConjuntoSolidos2D` que representa un conjunto
de sólidos en el plano conformados por paneles y sus estela conformadas
por paneles y vórtices.
"""

from dataclasses import dataclass, field, InitVar
from typing import List, Tuple
import numpy as np

from .Tipos import PanConfig

from ._TernasMoviles2D import TernasMoviles2D
from ._Paneles2D import Paneles2D
from ._Vortices2D import Vortices2D

@dataclass
class ConjuntoSolidos2D:
    """
    Conjunto de `N_S` sólidos en el plano conformados por paneles y sus estelas
    conformadas por paneles y vórtices.

    Los objetos de esta clase solo son contenedores de información geométrica y cinemática.
    No tienen métodos especializados definidos.

    Notes
    -----
    - XY denota sistema coordenado global.
    - xy denota sistema coordando solidario a cada sólido.
    - N_S: número de sólidos.
    - N_s: número de paneles del sólido s.
    - N_w_s: número de singularidades de estela del sólido s. Considera 1 vórtice y N_w_s - 1 paneles.

    Parameters
    ----------
    R_XY : tuple of np.ndarray, length N_S
        Coordenadas de los nodos de cada sólido, cada array con forma (2, N_s+1).
    VRelPC_xy : tuple of np.ndarray or tuple of None, length N_S
        Velocidades relativas en los puntos de colocación, cada array con forma (2, N_s)
        Por defecto, los array se completan con ceros.
    TM : \_TernasMoviles2D.TernasMoviles
        Cinemática de las ternas móviles solidarias a cada sólido,  
        capaz de transformar entre marcos global y solidarios.
    R_XY_estelas : tuple of np.ndarray, length N_S
        Coordenadas de los nodos de la estela de cada sólido, cada array con forma (2, N_w_s - 1). 

        .. notes::

            - El orden de los nodos es recien creado -> más viejo.

            Esto se debe al funcionamiento de _Paneles2D.Paneles2D. Si se pasara en el orden
            convencional, una estela horizontal que comience en el BF en -x y termine en el fluido en x,
            tendría normal hacia -y, lo cual no estaría acorde a lo definido en el método de los paneles.

            - Solo se consideran N_w_s - 1 coordenadas, porque la coordenada del vórtice de arranque
            coincide con el nodo en la última posición, con lo que no se lo incorpora en `R_XY_estelas`.
        
    intensidades_estelas : tuple of np.ndarray, length N_S
        Intensidades de las singularidades de las estelas (paneles + vórtice), cada array con forma (N_w_s + 1).
    panConfig : list of Tipos.PanConfig
        Lista de configuraciones de paneles, una por sólido.

    Attributes
    ----------
    solidos : list of _Paneles2D.Paneles2D
        Objetos `Paneles2D` que representan cada sólido.
    TM : _TernasMoviles2D.TernasMoviles2D
        Ternas móviles de cada sólido.
    estelas : list of Paneles2D
        Objetos `Paneles2D` que representan las estelas (si existen).
    vortices : list of Vortices2D
        Objetos `Vortices2D` con los vórtices de arranque de las estelas.
    PC_XY : list of np.ndarray
        Puntos de colocación, cada array con forma (2, N_s).
    normales_XY : list of np.ndarray
        Vectores normales a los paneles, cada array con forma (2, N_s).
    VPC_XY : list of np.ndarray
        Velocidades en los puntos de colocación, cada array con forma (2, N_s).
    VInd : list of np.ndarray
        Velocidades inducidas (totales) en cada punto de colocación, cada array con forma (2, N_s).
    """

    R_XY: InitVar[Tuple[np.ndarray, ...]]                           # (N_S,2,N_s+1)
    VRelPC_xy: InitVar[Tuple[np.ndarray, ...] | Tuple[None,...]]    # (N_S,2,N_s)

    TM: TernasMoviles2D                                     # (N_S,)

    R_XY_estelas: InitVar[Tuple[np.ndarray,...]]            # (N_S,2,N_w_s)
    intensidades_estelas: InitVar[Tuple[np.ndarray,...]]    # (N_S, N_w_s)

    panConfig: InitVar[List[PanConfig]]

    solidos: List[Paneles2D] = field(init=False)
    estelas: List[Paneles2D] = field(init=False)
    vortices: List[Vortices2D] = field(init=False)

    PC_XY: List[np.ndarray] = field(init=False)         # (N_S,2,N_s)
    normales_XY: List[np.ndarray] = field(init=False)   # (N_S,2,N_s)
    VPC_XY: List[np.ndarray] = field(init=False)        #(N_S,2,N_s)

    VInd: List[np.ndarray] = field(init=False)         #(N_S,2,N_s)

    def __post_init__(
            self,
            R_XY: Tuple[np.ndarray, ...],
            VRelPC_xy: Tuple[np.ndarray, ...],
            R_XY_estelas: Tuple[np.ndarray,...],
            intensidades_estelas: Tuple[np.ndarray],
            panConfig: List[dict],
            ):
        
        self.solidos = []
        self.estelas = []
        self.vortices = []

        self.PC_XY = []
        self.normales_XY = []
        self.VPC_XY = []

        
        if R_XY_estelas[0].shape[1] < 1:
            raise ValueError('R_XY_estelas debe ser una lista de N_S arreglos de la forma (2, N_s) con N_s >= 1, s = 1,...,N_S')
        
        for s in range(len(R_XY)):
            # Se generan los paneles de los solidos.
            self.solidos.append(Paneles2D(R_XY[s], panConfig[s], 1.))
            # Se generan los vortices de los solidos.
            self.vortices.append(Vortices2D(R_XY_estelas[s][:,-1:], intensidades_estelas[s][-1]))

            # Si ya se avanzó un instante, se generan los paneles de estela.
            if R_XY_estelas[0].shape[1] >= 2:
                self.estelas.append(Paneles2D(R_XY_estelas[s], panConfig[s], intensidades_estelas[s][:-1]))

            # Se guardan coordenadas, normales y velocidades de los puntos de colocación.
            self.PC_XY.append(self.solidos[s].PC_xy())
            self.normales_XY.append(self.solidos[s].normales_xy())
            
            if VRelPC_xy[s] is None:
                VRelPC_xy[s] = np.zeros_like(self.PC_XY[s])
            self.VPC_XY.append(self.TM.v2V_1TM(self.PC_XY[s], VRelPC_xy[s], s))

        

        

    
        
        
            

