# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# =======================================================================================
# Proyecto: Trabajo Final - Método de Paneles Bidimensional Multielemento No estacionario
# Archivo: src/_Paneles2D.py
# Autor: Bruno Francisco Barra Atarama
# Institución: 
#   Departamento de Ingeniería Aeroespacial
#   Facultad de Ingeniería
#   Universidad Nacional de La Plata
# Año: 2025
# Licencia: PolyForm Noncommercial License 1.0.0
# =======================================================================================

"""
Módulo en el que define la clase Paneles2D que permite representar un conjunto 
de paneles con singularidad distribuida, obtener su influencia y ángulos 
característicos en puntos del plano.

Actualmente solo se encuentra implementada una distribución de dobletes
constante.
"""

import numpy as np
from dataclasses import dataclass, field

from .Tipos import PanConfig
from ._TernasMoviles2D import TernasMoviles2D
from ._Vortices2D import Vortices2D

@dataclass
class Paneles2D:
    """
    Conjunto de `N` paneles con singularidad distribuida.

    .. note::

        Actualmente solo se encuentra implementada una distribución de dobletes
        constante.

    Attributes
    ----------
    r_xy : np.ndarray shape (2, N + 1)
        Posiciones `(x, y)` de los nodos de N paneles conformados por los segmentos
        rectos unen los nodos consecutivamente.
    panConfig : Tipos.Panconfig
        Diccionario de configuración.
    intensidades : float | np.ndarray, optional
        Intensidad de cada singularidad. Tras la inicialización siempre consiste
        en un array de forma `(N,)`. Si se proporciona un escalar, se repite para
        todos los vórtices. Por defecto, `1.0`.
    TP : \_TernasMoviles2D.TernasMoviles2D
        Objeto que guarda la posición, actitud y matrices de transformación de 
        coordenadas de los paneles
    x2_xyp : np.ndarray, shape (N,)
        Arreglo de coordenadas `x2` en el sistema coordenado relativo a cada panel.
        Estas coordenadas equivalen a las longitudes de cada panel.

    .. note::

        Las coordenadas `(x, y)` pueden ser relativas a un sólido o globales.
    """
    r_xy: np.ndarray    # (2, N + 1)
    panConfig: PanConfig
    intensidades: float | np.ndarray | None = 1.  # Escalar | (N,) | 1.
    
    TP: TernasMoviles2D = field(init=False)         # (N, 2, 2)
    x2_xyp: np.ndarray = field(init=False)          # (N,)

    def __post_init__(self):
        if self.intensidades is None:
            self.intensidades = 1.
        if np.isscalar(self.intensidades):
            self.intensidades = np.full((self.r_xy.shape[1] - 1,), self.intensidades)

        theta = np.atan2(self.r_xy[1, 1:] - self.r_xy[1, :-1], self.r_xy[0, 1:] - self.r_xy[0, :-1])  # (N,)

        self.TP = TernasMoviles2D(self.r_xy[:, :-1], theta)

        self.x2_xyp = np.linalg.norm(self.r_xy[:, 1:] - self.r_xy[:, :-1], axis=0)   # (N,) Longitudes de cada columna bidimensional (se realiza la norma variando la fila, eje 0)

    def PC_xy(self) -> np.ndarray:    # (2, N)
        """
        Método que calcula los vectores posición de los puntos de colocación
        en coordenadas `(x, y)`, ya sean estas relativas a un sólido o globales
        """
        return self.TP.r2R(
            np.stack([
                self.panConfig['xPC_x2_xyp'] * self.x2_xyp,
                np.zeros_like(self.x2_xyp)
            ], axis=-2)
        )    
    
    def normales_xy(self) -> np.ndarray:  # (2, N)
        """
        Método que calcula los vectores normales a los paneles 
        en coordenadas `(x, y)`, ya sean estas relativas a un sólido o globales
        """
        return self.TP.n2N_full(np.array([[0], [1]]))[:, :, 0]   # (2,N) Normales a los paneles.
        
    def VInd_xy(
            self,
            r_xy: np.ndarray,           # (2, M)
            ubicaciones: None | np.ndarray,    # (N, M)
            afuera: bool = True,
            intensidadesUnitarias: bool = True
    ) -> np.ndarray:    #(2, N, M)
        """
        Método que calcula individualmente las velocidades inducidas por todas las 
        singularidades en todos los puntos de evaluación en `r_xy`.

        Parameters
        ----------
            r_xy : np.ndarray, shape (2, M)
                Coordenadas de evaluación.
            ubicaciones : np.ndarray | None, shape (N, M)
                Matriz indicadora de influencias.

                Si un elemento es `0`, para esa singularidad y para ese punto de
                evaluación, se considera al punto como si estuviera fuera de la 
                singularidad.

                Si un elemento es `-1`, punto de evaluación en la singularidad.

                Si un elemento es `1`, punto de evaluación en el extremo r1 del panel.

                Si un elemento es `2`, punto de evaluación en el extremo r2 del panel.

                Por defecto, se consideran todos los puntos fuera de las 
                singularidades.
            afuera : bool, optional
                Argumento en desuso. Implica que en los casos de puntos de evaluación
                en las singularidades, se considere el límite por afuera del sólido
                si `True`, y por dentro en el caso contrario.
                Por defecto, `True`.
            intensidadesUnitarias : bool, optional
                Ignora intensidades almacenadas en la instancia si `True`.
        Returns
        -------
            VInd : np.ndarray, shape (2, N, M)
                Velocidades inducidas individualmente.
        """
        
        N = self.r_xy.shape[1] - 1
        M = r_xy.shape[1]

        coef = 1/2/np.pi
        r_xyp = self.TP.R2r_full(r_xy)  # (2, N, M)

        if ubicaciones is None:
            ## Evaluación fuera de las singularidades en todos lados.
            x2 = self.x2_xyp[:,None]   
            x = r_xyp[0, :, :]  
            y = r_xyp[1, :, :]

            R2 = x ** 2 + y ** 2
            D = (x - x2) ** 2 + y ** 2

            Vx = - coef * (y / R2 - y / D)
            Vy = coef * (x / R2 - (x - x2) / D)

            VInd = np.stack([Vx, Vy], axis=0)

            VInd = self.TP.n2N_2NM(VInd)

        else:
            VInd = np.zeros((2,N,M))
            VInd_local = np.zeros((2,N,M))

            ## 0: evaluación fuera del panel.
            n, m = np.nonzero(ubicaciones == 0)
            if n.size:
            
                x2 = self.x2_xyp[n]     # (K,)
                x = r_xyp[0, n, m]         # (K,)
                y = r_xyp[1, n, m]         # (K,)

                R2 = x ** 2 + y ** 2
                D = (x - x2) ** 2 + y ** 2

                Vx = - coef * (y / R2 - y / D)
                Vy = coef * (x / R2 - (x - x2) / D)

                VInd_local[0, n, m] = Vx
                VInd_local[1, n, m] = Vy

            ## -1 evaluación en el panel pero no en los extremos.
            n, m = np.nonzero(ubicaciones == -1)
            if n.size:
            
                x2 = self.x2_xyp[n]         # (K,)
                x = r_xyp[0, n, m]          # (K,)
                y = r_xyp[1, n, m]          # (K,)

                Vy = coef * (1 / x - 1 / (x - x2))

                VInd_local[1, n, m] = Vy

            ## 1 evaluación en el extremo 1 del panel.
            # Índice del panel se corresponde con el índice del extremo 1 del panel.
            n, m = np.nonzero(ubicaciones == 1)
            if n.size:

                r1_xy = self.r_xy[:, n]     # (2, K)
                r2_xy = self.r_xy[:, n + 1] # (2, K)
                
                vortices = Vortices2D(r2_xy, intensidades=-1)
                VInd[:, n, m] = vortices.VInd_xy_diag(r1_xy, intensidadesUnitarias=False)

            ## 2 evaluación en el extremo 2 del panel.
            # Índice del panel se corresponde con el índice del extremo 1 del panel.
            n, m = np.nonzero(ubicaciones == 2)
            if n.size:

                r1_xy = self.r_xy[:, n]     # (2, K)
                r2_xy = self.r_xy[:, n + 1] # (2, K)
                
                vortices = Vortices2D(r1_xy, intensidades=1)
                VInd[:, n, m] = vortices.VInd_xy_diag(r2_xy, intensidadesUnitarias=False)

            VInd_local = self.TP.n2N_2NM(VInd_local)    # Solo las contribuciones en ubicaciones -1 y 0 deben transformarse. 
            # Las de los vórtices en 1 y 2 ya se calcularon en la terna solidaria al sólido.
            VInd += VInd_local

        if not intensidadesUnitarias:
            if np.isscalar(self.intensidades):
                VInd *= self.intensidades
            else:
                VInd *= self.intensidades[:, None]
        return VInd
    
    def th12(
            self,
            r_xy: np.ndarray,   # (2, M)
    ) -> np.ndarray:            # (M, N)
        """
        Calcula los ángulos polares desde cada extremo de los paneles
        hacia los puntos de evaluación en `r_xy`.

        Parameters
        ----------
        r_xy : np.ndarray, shape (2, M)
            Coordenadas de los puntos de evaluación.

        Returns
        -------
        theta1, theta2 : tuple of np.ndarray, each shape (M, N)
            - `theta1`: ángulos polares de los vectores que van
              desde el primer nodo de cada panel hacia cada punto.
            - `theta2`: ángulos polares de los vectores que van
              desde el segundo nodo de cada panel hacia cada punto.
        """
        r1 = r_xy[:, None, :] - self.r_xy[:,:-1, None]
        r2 = r_xy[:, None, :] - self.r_xy[:,1:, None]
        return np.arctan2(r1[1,:,:], r1[0,:,:]), np.arctan2(r2[1,:,:], r2[0,:,:])
