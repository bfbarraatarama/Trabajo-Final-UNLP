# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# =======================================================================================
# Proyecto: Trabajo Final - Método de Paneles Bidimensional Multielemento No estacionario
# Archivo: src/_Vortices2D.py
# Autor: Bruno Francisco Barra Atarama
# Institución: 
#   Departamento de Ingeniería Aeroespacial
#   Facultad de Ingeniería
#   Universidad Nacional de La Plata
# Año: 2025
# Licencia: PolyForm Noncommercial License 1.0.0
# =======================================================================================

"""
Módulo en el que se define la clase `Vortices2D` que permite representar un conjunto 
de vórtices puntuales bidimensionales, obtener su influencia y ángulos 
característicos en puntos del plano.

"""

import numpy as np
from dataclasses import dataclass

@dataclass
class Vortices2D:
    """
    Conjunto de `N` vórtices puntuales bidimensionales.

    Attributes
    ----------
    r0_xy : np.ndarray, shape (2, N)
        Posiciones `(x, y)` de los vórtices.
    intensidades : float | np.ndarray, optional
        Intensidad de cada singularidad. Tras la inicialización siempre consiste
        en un array de forma (N,). Si se proporciona un escalar, se repite para
        todos los vórtices. Por defecto, `1.0`.

    .. note::

        Las coordenadas `(x, y)` pueden ser relativas a un sólido o globales.
    """

    r0_xy: np.ndarray   #(2, N)
    intensidades: float | np.ndarray | None = 1.  # Escalar | (N,) | 1.

    def __post_init__(self):
        if self.intensidades is None:
            self.intensidades = 1.
        if np.isscalar(self.intensidades):
            self.intensidades = np.full((self.r0_xy.shape[1],), self.intensidades)
    
    def VInd_xy(
            self,
            r_xy: np.ndarray,           # (2, M)
            ubicaciones: None | np.ndarray,    # (N, M)
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

                .. note::
                    Se adopta el `-1` por convención, pero en realidad con ser
                    distinto de `0` basta.

                Por defecto, se consideran todos los puntos fuera de las 
                singularidades.
            intensidadesUnitarias : bool, optional
                Ignora intensidades almacenadas en la instancia si `True`.
        Returns
        -------
            VInd : np.ndarray, shape (2, N, M)
                Velocidades inducidas individualmente.
        """

        N = self.r0_xy.shape[1]
        M = r_xy.shape[1]

        VInd = np.zeros((2, N, M))

        if ubicaciones is None:
            ## Evaluación fuera del vórtice en todos lados.
            x0 = self.r0_xy[0, :]   # (K,)
            y0 = self.r0_xy[1, :]   # (K,)
            x = r_xy[0, :]          # (K,)
            y = r_xy[1, :]          # (K,)

            dx = x - x0
            dy = y - y0
            r2 = dx ** 2 + dy ** 2

            coef = - 1/2/np.pi  # Agrego un menos repescto al Katz y Plotkin, por convención del sentido de circulación.

            Vx = coef * dy / r2
            Vy = - coef * dx / r2

            VInd[0, :, :] = Vx
            VInd[1, :, :] = Vy
        else:
            n, m = np.nonzero(ubicaciones == 0) ## 0 evaluación fuera del vórtice.
            if n.size:
                x0 = self.r0_xy[0, n]   # (K,)
                y0 = self.r0_xy[1, n]   # (K,)
                x = r_xy[0, m]          # (K,)
                y = r_xy[1, m]          # (K,)

                dx = x - x0
                dy = y - y0
                r2 = dx ** 2 + dy ** 2

                coef = - 1/2/np.pi  # Agrego un menos repescto al Katz y Plotkin, por convención del sentido de circulación.

                Vx = coef * dy / r2
                Vy = - coef * dx / r2

                VInd[0, n, m] = Vx
                VInd[1, n, m] = Vy

                ## -1 evaluación en el vórtice: no se hace nada porque la autoinfluencia es nula.

        if not intensidadesUnitarias:
            if np.isscalar(self.intensidades):
                VInd *= self.intensidades
            else:
                VInd *= self.intensidades[:, None]
        return VInd
    

    def VInd_xy_diag(
            self,
            r_xy: np.ndarray,           # (2, N)
            intensidadesUnitarias: bool = True
    ) -> np.ndarray:    #(2, N)
        """
        Método que calcula individualmente las velocidades inducidas por cada una de las singularidades 
        en cada uno de los puntos `r_xy` en la misma posición.

        Es decir, se devuelve la velocidad inducida por la singularidad en la posición 0 sobre el punto 
        de evaluación en la posición 0, la inducida por la singularidad en la posición 1 sobre el punto
        en la posición 1, y así sucesivamente.

        .. note::

        Se asumen los puntos de evaluación fuera de las singularidades.

        Parameters
        ----------
            r_xy : np.ndarray, shape (2, N)
                Coordenadas de evaluación. Mísma longitud que `Vortices2D.r0_xy`
            intensidadesUnitarias : bool, optional
                Ignora intensidades almacenadas en la instancia si `True`.
        Returns
        -------
            VInd : np.ndarray, shape (2, N)
                Velocidades inducidas.
        """
        # Solo se calcula fuera de los paneles. 
        # Si se introduce r_xy inconsistente con eso, habrá problemas numéricos por indeterminaciones matemáticas.

        x0 = self.r0_xy[0,:] 
        y0 = self.r0_xy[1,:]
        x = r_xy[0,:]
        y = r_xy[1,:]

        dx = x - x0
        dy = y - y0
        r2 = dx ** 2 + dy ** 2

        coef = - 1/2/np.pi  # Agrego un menos repescto al Katz y Plotkin, por convención del sentido de circulación.

        Vx = coef * dy / r2
        Vy = - coef * dx / r2

        VInd = np.stack((Vx, Vy), axis=0)   # (2, N)

        if not intensidadesUnitarias:
            if np.isscalar(self.intensidades):
                VInd *= self.intensidades
            else:
                VInd *= self.intensidades[None,:]
        return VInd

    def th(
            self,
            r_xy: np.ndarray,           # (2, M)
    ) -> np.ndarray:    #(N, M)
        """
        Método que calcula el ángulo polar (atan2) de cada vector desde `Vortices2D.r0_xy` hasta `r_xy`.

        Parameters
        ----------
        r_xy : np.ndarray, shape (2, M)
            Puntos de evaluación.

        Returns
        -------
        th : np.ndarray, shape (N, M).
            Matriz de ángulos.
        """
        r = r_xy[:, None, :] - self.r0_xy[:,:, None]
        return np.arctan2(r[1,:,:], r[0,:,:])
    