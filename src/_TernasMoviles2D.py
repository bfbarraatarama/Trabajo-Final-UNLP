# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# =======================================================================================
# Proyecto: Trabajo Final - Método de Paneles Bidimensional Multielemento No estacionario
# Archivo: src/_TernasMoviles2D.py
# Autor: Bruno Francisco Barra Atarama
# Institución: 
#   Departamento de Ingeniería Aeroespacial
#   Facultad de Ingeniería
#   Universidad Nacional de La Plata
# Año: 2025
# Licencia: PolyForm Noncommercial License 1.0.0
# =======================================================================================

"""
Módulo en el que se define la clase contenedora de ternas móviles con métodos
dedicados para las transformaciones de coordenadas.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union, Sequence

@dataclass
class TernasMoviles2D:
    '''
    Conjunto de `N` ternas móviles con movimientos restringidos al plano.

    Notes
    -----
    - Cada terna móvil con origen en `O` tiene asociada un sistema coordenado solidario `xy`.
    - Además de las ternas móviles, se considera una terna fija con origen en `Ω` y con un }
    sistema coordenado `XY` solidario a ella.
    - Se denota con `R` a los vectores posición definidos en el sistema coordenado `XY` y 
    con `r` a las posiciones definidas en coordenadas `xy`.
    - Se denota con `N` a un vector en coordenadas `XY` y con `n` al mismo vector en el 
    sistema `xy`. Notar que en este caso la norma euclidiana es invariante entre `N` y `n`,
    pero no necesariamente lo es entre `R` y `r`.
    - Se denota con `V` a una velocidad definida en `XY` y con `v` a su equivalente en el 
    sistema `xy`.

    Attributes
    ----------
    RO : np.ndarray shape (2, N)
        Matriz cuyas columnas son los vectores posición de los orígenes `O` de las ternas móviles. 
    theta : np.ndarray shape (N,)
        Vector con las actitudes de las ternas móviles respecto de la fija. Se mide desde el eje X
        hasta el x, positivo antihorario.
    VO : np.ndarray shape (2, N)
        Matriz cuyas columnas son los vectores de velocidad de traslación de los orígenes `O` de
        las ternas móviles.
    w : np.ndarray (N,)
        Vector con las velocidades ángulares de las ternas móviles.
    T : np.ndarray shape (N, 2, 2)
        Arreglo que guarda las `N` matrices de rotación N -> n.

        .. note:: T y TInv tienen como primer índice al de las ternas para agilizar indexación.
    TInv : np.ndarray shape (N, 2, 2)
        Arreglo que guarda las `N` matrices de rotación N -> n.
    '''
    RO: np.ndarray      # (2, N)
    theta: np.ndarray   # (N,)
    VO: np.ndarray | None = None    # (2, N)
    w: np.ndarray | None = None     # (N,)
    
    T: np.ndarray = field(init=False)       # (N, 2, 2)
    TInv: np.ndarray = field(init=False)    # (N, 2, 2)

    def __post_init__(self):
        # Se calculan cosenos y senos de las matrices de transformación
        c = np.cos(self.theta)  # (N,)
        s = np.sin(self.theta)  # (N,)
        
        # Filas de T
        fila1 = np.stack([c, s], axis=-1)   # (N, 2)
        fila2 = np.stack([-s, c], axis=-1)  # (N, 2)

        self.T = np.stack([fila1, fila2], axis=-2)  # (N,2,2)
        # Con axis=-2 en el stack, se introduce el nuevo eje (filas) entre el índice de terna y el de columnas.

        self.TInv = np.transpose(self.T, (0, 2, 1))  # (N,2,2)   Se contstruye TInv transponiendo T.

# -----------------------------------------------------------------------------
# Métodos para transformar N vectores de entrada, cada una con la terna en la
# misma posición.
# -----------------------------------------------------------------------------
    def R2r(
            self, 
            R: np.ndarray,  # (2, N)
    ) -> np.ndarray:    # (2, N)
        """
        Transformar cada uno de los N vectores de entrada con la terna en la 
        misma posición.

        Parameters
        ----------
        R : np.ndarray shape (2, N)
            Vectores posición en `XY`.
        
        Returns
        -------
        r : np.ndarray shape (2, N)
            Vectores posición en `xy`.
        """
        R = R - self.RO
        return np.einsum('nij,jn->in', self.T, R)   # (2, N)

    def r2R(
            self, 
            r: np.ndarray,  # (2, N) 
    ) -> np.ndarray:    # (2, N) 
        """
        Transformar cada uno de los N vectores de entrada con la terna en la 
        misma posición.

        Parameters
        ----------
        r : np.ndarray shape (2, N)
            Vectores posición en `xy`.
        
        Returns
        -------
        R : np.ndarray shape (2, N)
            Vectores posición en `XY`.
        """
        return np.einsum('nij,jn->in', self.TInv, r) + self.RO  # (2, N)
    
    def N2n(
            self, 
            N: np.ndarray,   # (2, N) 
    ) -> np.ndarray:    # (2, N) 
        """
        Transformar cada uno de los N vectores de entrada con la terna en la 
        misma posición.

        Parameters
        ----------
        N : np.ndarray shape (2, N)
            Vectores en `XY`.
        
        Returns
        -------
        n : np.ndarray shape (2, N)
            Vectores en `xy`.
        """
        return np.einsum('nij,jn->in', self.T, N)  # (2, N)
    
    def n2N(
            self, 
            n: np.ndarray,   # (2, N) 
    ) -> np.ndarray:    # (2, N)
        """
        Transformar cada uno de los N vectores de entrada con la terna en la 
        misma posición.

        Parameters
        ----------
        n : np.ndarray shape (2, N)
            Vectores en `xy`.
        
        Returns
        -------
        N : np.ndarray shape (2, N)
            Vectores en `XY`.
        """
        return np.einsum('nij,jn->in', self.TInv, n)  # (2, N)
    
    def v2V(
            self,
            R: np.ndarray,   # (2, N)
            vRel: np.ndarray,    # (2, N)
    ) -> np.ndarray:
        """
        Transformar cada uno de los N vectores de entrada con la terna en la 
        misma posición.

        Parameters
        ----------
        R : np.ndarray shape (2, N)
            Vectores posición de los puntos de evaluación en `XY`.
        vRel : np.ndarray shape (2, N)
            Vectores velocidad relativa de los puntos de evaluación en `xy`.

        Returns
        -------
        V : np.ndarray shape (2, N)
            Vectores velocidad en `XY`.
        """
        R = R - self.RO 
        return self.n2N(vRel) + self.VO + np.stack([- self.w * R[1], self.w * R[0]], axis = 0)  # (2, N) VRel + VO + w x (R - RO)
    
    
# -----------------------------------------------------------------------------
# Métodos para transformar M vectores de entrada, con una sola terna móvil.
# -----------------------------------------------------------------------------

    def R2r_1TM(
            self, 
            R: np.ndarray,  # (2, M)
            idc: int,
    ) -> np.ndarray:    # (2, M)
        """
        Transformar cada uno de los M vectores de entrada con la terna en la 
        posición indicada.

        Parameters
        ----------
        R : np.ndarray shape (2, M)
            Vectores posición en `XY`.
        idc : int
            Índice de la terna móvil a utilizar.
        
        Returns
        -------
        r : np.ndarray shape (2, M)
            Vectores posición en `xy`.
        """
        idc = int(idc)
        
        R = R - self.RO[:,idc][:,None]
        return np.einsum('ij,jn->in', self.T[idc], R)   # (2, M)

    def r2R_1TM(
            self, 
            r: np.ndarray,  # (2, M) 
            idc: int,
    ) -> np.ndarray:    # (2, M)
        """
        Transformar cada uno de los M vectores de entrada con la terna en la 
        posición indicada.

        Parameters
        ----------
        r : np.ndarray shape (2, M)
            Vectores posición en `xy`.
        idc : int
            Índice de la terna móvil a utilizar.
        
        Returns
        -------
        R : np.ndarray shape (2, M)
            Vectores posición en `XY`.
        """
        idc = int(idc) 
        
        return np.einsum('ij,jn->in', self.TInv[idc], r) + self.RO[:,idc][:,None]  # (2, M)

    def N2n_1TM(
            self, 
            N: np.ndarray,   # (2, M) 
            idc: int,
    ) -> np.ndarray:    # (2, M)
        """
        Transformar cada uno de los M vectores de entrada con la terna en la 
        posición indicada.

        Parameters
        ----------
        N : np.ndarray shape (2, M)
            Vectores en `XY`.
        idc : int
            Índice de la terna móvil a utilizar.
        
        Returns
        -------
        n : np.ndarray shape (2, M)
            Vectores en `xy`.
        """
        idc = int(idc)

        return np.einsum('ij,jm->im', self.T[idc], N)  # (2, M)

    def n2N_1TM(
            self, 
            n: np.ndarray,   # (2, M) 
            idc: int,
    ) -> np.ndarray:    # (2, M)
        """
        Transformar cada uno de los M vectores de entrada con la terna en la 
        posición indicada.

        Parameters
        ----------
        n : np.ndarray shape (2, M)
            Vectores en `xy`.
        idc : int
            Índice de la terna móvil a utilizar.
        
        Returns
        -------
        N : np.ndarray shape (2, M)
            Vectores en `XY`.
        """
        idc = int(idc)

        return np.einsum('ij,jm->im', self.TInv[idc], n)  # (2, M)
    
    def v2V_1TM(
            self,
            R: np.ndarray,   # (2, M)
            vRel: np.ndarray,    # (2, M)
            idc: int,
    ) -> np.ndarray:    # (2, M)
        """
        Transformar cada uno de los M vectores de entrada con la terna en la 
        posición indicada.

        Parameters
        ----------
        R : np.ndarray shape (2, M)
            Vectores posición de los puntos de evaluación en `XY`.
        vRel : np.ndarray shape (2, M)
            Vectores velocidad relativa de los puntos de evaluación en `xy`.
        idc : int
            Índice de la terna móvil a utilizar.

        Returns
        -------
        V : np.ndarray shape (2, M)
            Vectores velocidad en `XY`.
        """
        idc = int(idc)

        R = R - self.RO[:, idc][:,None]

        # V = VRel + VO + w x (R - RO)
        return self.n2N_1TM(vRel, idc) + self.VO[:,idc][:,None] + np.stack([- self.w[idc] * R[1], self.w[idc] * R[0]], axis = 0)  # (2, M) 
    
# -----------------------------------------------------------------------------
# Métodos para transformar M vectores de entrada, cada uno con las N ternas móviles.
# -----------------------------------------------------------------------------
    @staticmethod
    def _idc(
            idc: Optional[Union[int, slice, Sequence[int]]]
    ) -> Union[slice, np.ndarray]:
        if idc is None:
            return slice(None)   
        elif isinstance(idc, int):
            return np.array([idc])
        elif isinstance(idc, slice):
            return idc
        return np.asarray(idc)

    def R2r_full(
            self, 
            R: np.ndarray,  # (2, M) 
            idc: Optional[Union[int, slice, Sequence[int]]] = None,   # La longitud de este idc es el nuevo N.
    ) -> np.ndarray:    # (2, N, M)
        """
        Transformar cada uno de los M vectores de entrada con cada una de las ternas o un subconjunto de ellas.

        N denota al número de ternas efectivamente utilizadas.

        Parameters
        ----------
        R : np.ndarray shape (2, M)
            Vectores posición en `XY`.
        idc : int | slice | Sequence[int] | None
            Índice/s de las ternas móviles a utilizar. Por defecto se usan todas.
        
        Returns
        -------
        r : np.ndarray shape (2, N, M)
            Vectores posición en `xy`.
        """
        idc = self._idc(idc)

        R = R[:, None, :] - self.RO[:, idc, None]
        return np.einsum('nij,jnm->inm', self.T[idc], R)   # (2, N, M)

    def r2R_full(
            self, 
            r: np.ndarray,   # (2, M)
            idc: Optional[Union[int, slice, Sequence[int]]] = None,   # La longitud de este idc es el nuevo N.
    ) -> np.ndarray:    # (2, N, M) 
        """
        Transformar cada uno de los M vectores de entrada con cada una de las N ternas o un subconjunto de ellas.

        N denota al número de ternas efectivamente utilizadas.

        Parameters
        ----------
        r : np.ndarray shape (2, M)
            Vectores posición en `xy`.
        idc : int | slice | Sequence[int] | None
            Índice/s de las ternas móviles a utilizar. Por defecto se usan todas.
        
        Returns
        -------
        R : np.ndarray shape (2, N, M)
            Vectores posición en `XY`.
        """       
        idc = self._idc(idc)
        return np.einsum('nij,jnm->inm', self.TInv[idc], r[:, None, :]) + self.RO[:, idc, None]  # (2, N, M)
    
    def N2n_full(
            self, 
            N: np.ndarray,   # (2, M) 
            idc: Optional[Union[int, slice, Sequence[int]]] = None,   # La longitud de este idc es el nuevo N.
    ) -> np.ndarray:    # (2, N, M) 
        """
        Transformar cada uno de los M vectores de entrada con cada una de las N ternas o un subconjunto de ellas.

        N denota al número de ternas efectivamente utilizadas.
        
        Parameters
        ----------
        N : np.ndarray shape (2, M)
            Vectores en `XY`.
        idc : int | slice | Sequence[int] | None
            Índice/s de las ternas móviles a utilizar. Por defecto se usan todas.
        
        Returns
        -------
        n : np.ndarray shape (2, N, M)
            Vectores en `xy`.
        """
        idc = self._idc(idc)
        return np.einsum('nij,jnm->inm', self.T[idc], N[:, None, :])  # (2, N, M)
    
    def n2N_full(
            self, 
            n: np.ndarray,   # (2, M) 
            idc: Optional[Union[int, slice, Sequence[int]]] = None,   # La longitud de este idc es el nuevo N.
    ) -> np.ndarray:    # (2, N, M)
        """
        Transformar cada uno de los M vectores de entrada con cada una de las N ternas o un subconjunto de ellas.

        N denota al número de ternas efectivamente utilizadas.
        
        Parameters
        ----------
        n : np.ndarray shape (2, M)
            Vectores en `xy`.
        idc : int | slice | Sequence[int] | None
            Índice/s de las ternas móviles a utilizar. Por defecto se usan todas.
        
        Returns
        -------
        N : np.ndarray shape (2, N, M)
            Vectores en `XY`.
        """
        idc = self._idc(idc)
        return np.einsum('nij,jnm->inm', self.TInv[idc], n[:, None, :])  # (2, N, M)
    
    def v2V_full(
            self,
            R: np.ndarray,   # (2, M)
            vRel: np.ndarray,    # (2, M)
            idc: Optional[Union[int, slice, Sequence[int]]] = None,   # La longitud de este idc es el nuevo N.
    ) -> np.ndarray:
        """
        Transformar cada uno de los M vectores de entrada con cada una de las N ternas o un subconjunto de ellas.

        N denota al número de ternas efectivamente utilizadas.
        
        Parameters
        ----------
        R : np.ndarray shape (2, M)
            Vectores posición de los puntos de evaluación en `XY`.
        vRel : np.ndarray shape (2, M)
            Vectores velocidad relativa de los puntos de evaluación en `xy`.
        idc : int | slice | Sequence[int] | None
            Índice/s de las ternas móviles a utilizar. Por defecto se usan todas.
        
        Returns
        -------
        V : np.ndarray shape (2, N, M)
            Vectores velocidad en `XY`.
        """
        idc = self._idc(idc)

        # V = VRel + VO + w x (R - RO)
        V = self.n2N_full(vRel, idc) + self.VO[:, idc, None] # VRel + VO, (2, N, M)

        R = R[:, None, :] - self.RO[:, idc, None]  # (2, N, M)

        # + w x (R - RO)
        w_col = self.w[idc, None]               # (N, 1)
        V[0] += -w_col * R[1]                   # componente x
        V[1] +=  w_col * R[0]                   # componente y
        return V


# -----------------------------------------------------------------------------
# Métodos para transformar NxM vectores de entrada guardados en un arreglo (2, N, M), 
# con las N ternas móviles, haciendo corresponder los índices de las ternas con
# el segundo índice de la entrada.
# -----------------------------------------------------------------------------

    def n2N_2NM(
            self,
            n: np.ndarray,    # (2, N, M)
            idc: Optional[Union[int, slice, Sequence[int]]] = None,   # La longitud de este idc es el nuevo N. 
    ) -> np.ndarray:    # (2, N, M)
        """
        Transformar cada uno de los NxM vectores de entrada guardados en un arreglo (2, N, M), 
        con las N ternas móviles, haciendo corresponder los índices de las ternas con
        el segundo índice de la entrada.

        N denota al número de ternas efectivamente utilizadas.
        
        Parameters
        ----------
        n : np.ndarray shape (2, N, M)
            Vectores en `xy`.
        idc : int | slice | Sequence[int] | None
            Índice/s de los vectores a transformar y las ternas móviles a utilizar. 
            
            Por defecto se transforma y usa todo.
        
        Returns
        -------
        N : np.ndarray shape (2, N, M)
            Vectores en `XY`.
        """

        idc = self._idc(idc)

        return np.einsum('nij,jnm->inm', self.TInv[idc], n[:, idc, :])  # (2, N, M)
