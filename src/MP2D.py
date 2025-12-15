# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# =======================================================================================
# Proyecto: Trabajo Final - Método de Paneles Bidimensional Multielemento No estacionario
# Archivo: src/MP2D.py
# Autor: Bruno Francisco Barra Atarama
# Institución: 
#   Departamento de Ingeniería Aeroespacial
#   Facultad de Ingeniería
#   Universidad Nacional de La Plata
# Año: 2025
# Licencia: PolyForm Noncommercial License 1.0.0
# =======================================================================================

"""
Módulo que contiene el núcleo de la simulación por el método de los paneles
bidimensional no estacionario.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Literal, Sequence
import warnings
import pickle

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from scipy.io import savemat
from scipy.linalg import lstsq, solve
from scipy.sparse.linalg import gmres
from scipy.linalg._misc import LinAlgError

from tqdm import tqdm
from time import time

from ._TernasMoviles2D import TernasMoviles2D
from ._ConjuntoSolidos2D import ConjuntoSolidos2D

from .Tipos import _MP2D_controlConfig_defecto, _MP2D_mpConfig_defecto, _MP2D_panConfig_defecto, _MP2D_refConfig_defecto, _completar_dic, _MP2D_SELInfo_defecto 
from .Tipos import ControlConfig, MPConfig, PanConfig, RefConfig, AdimInfo
from .Tipos import GeneradorCinematica, GeneradorCinematicaAeroelasticidad, GeneradorRM, GeneradorRMAeroelasticidad

@dataclass
class MP2D:
    """
    Clase con la cual se ensambla un experimento de flujo potencial bidimensional estacionario 
    o no estacionario que puede simularse utilizando el método MP2D.simular(...).

    Notes
    --------
    -   Aquí se presentan de forma sintética los atributos de la clase para no sobrecargar el docstring. 
    
        Por eso, se recomienda fuertemente prestar atención al tipado y construir los primeros experimentos
        usando los tipos creados y aquí utilizados, ya que sus propios docstrings brindan más detalles.

        Estos tipos se pueden importar del módulo `Tipos`.

    -   La simulación asume que no hay ninguna interferencia entre sólidos y estelas, más alla de la concurrencia
        entre el BF de un sólido y el nodo del último panel de su estal generado.

    Attributes
    ----------
    cinematicas : List[Tipos.GeneradorCinematica] | List[Tipos.GeneradorCinematicaAeroelasticidad]
        Conjunto de generadores obtenidos con `_Cinematicas.cinematica` con los que se construyan los 
        sólidos de la simulación.
    RM : Tipos.GeneradorRM | Tipos.GeneradorRMAeroelasticidad | None
        Conjunto de generadores obtenidos con `_Cinematicas.RM` con los que se obtienen los vectores de
        toma de momento para el cálculo de momentos aerodinámicos.

        Por defecto, no se utiliza y no se calculan momentos aerodinámicos.

        También puede ser None (por defecto) y en este caso no se computan momentos aerodinámicos.
    conjuntos : List[\_ConjuntoSolidos2D.ConjuntoSolidos2D]
        Lista de los conjuntos de sólidos en cada instante.
    mpConfig : Tipos.MPConfig | None
        Configuración del simulador. Por defecto se completa con el diccionario tipado por defecto.
    refConfig : Tipos.RefConfig | None
        Configuración de las magnitudes de referencia. Estas magnitudes son las que se utilizan
        para calcular los coeficientes aerodinámicos. Por defecto se completa con el diccionario tipado por defecto.
    panConfig : PanConfig | List[PanConfig] | None
        Configuración del panelado. Se utiliza una configuración por sólido. Si se pasa uno solo, este replica para todos.
        Si no se pasa, se completan con instancias del diccionario tipado por defecto.
    controlConfig : Tipos.ControlConfig | None
        Configuración de los controles en tiempo de ejecución. Por defecto se completa con el diccionario tipado por defecto.
    adimInfo : Tipos.admiInfo
        Diccionario con relaciones adimensionales calculadas a partir de las magnitudes de referencia.
    SELInfo : Tipos.SELInfo
        Información de los SEL resueltos por el simulador.
    t : List[float]
        Lista con los tiempos simulados.
    it : int
        Número de la última iteración de la simulación completa. Es `-1` antes de completar algún paso.

        .. note::
            El primer instante se corresponde con `it = 1`.
    N_S : int
        Cantidad de sólidos. Debe permanecer constante durante la simulación.
    N_s : List[int]
        Lista con la cantidad de paneles por solido. Debe permanecer constante durante la simulación.
    RMRec : List[np.ndarray] each shape (2, 1)
        Lista con los vectores para la toma de momentos en el cálculo de momentos aerodinámicos.
    CpRec : List[List[np.ndarray]] each shape (N_s[s],)
        Lista que por cada instante simulado guarda listas de arreglos con los Cp calculados en los puntos de colocación. 
    dFRec : List[List[np.ndarray]] each shape (2, N_s[s])
        Lista que por cada instante simulado guarda la fuerza aerodinámica actuante sobre cada panel 
        de cada sólido.
    CxyTotalRec : List[np.ndarray] each shape (2,)
        Lista que por cada instante simulado guarda los coeficientes de fuerzas aerodinámicos en arreglos
        donde la primera posición se corresponde con Cx y la segunda con Cy.
    CmTotalRec : List[float]
        Lista que por cada instante simualdo guarda el coeficiente de momento aerodinámico Cm.
    CxyRec : List[np.ndarray] each shape (2, N_S)
        Lista que por cada instante simulado guarda los coeficientes de fuerzas aerodinámicos de cada
        sólico en arreglos donde si se indexa con:
        - `[0, s]`, con `0 <= s <= N_S`, se obtiene el coeficiente Cx del sólido en la posición `s`.
        - `[1, s]`, con `0 <= s <= N_S`, se obtiene con el coeficiente Cy del sólido en la posición `s`.
    CmRec : List[np.ndarray] each shape (N_S,)
        Lista que por cada instante simulado guarda los coeficientes de momentos aerodinámicos de cada
        sólido en arreglos donde si se indexa con `[s]`, con `0 <= s <= N_S`, se obtiene
        el coeficiente Cm del sólido en la posición `s`.
    tRes : List[float]
        Lista con los tiempos de resolución de cada iteración.
    """

    # Cinemáticas
    cinematicas: GeneradorCinematica | List[GeneradorCinematica] | GeneradorCinematicaAeroelasticidad | List[GeneradorCinematicaAeroelasticidad]
    RM: GeneradorRM | GeneradorRMAeroelasticidad | None = None

    # Sólidos
    conjuntos: List[ConjuntoSolidos2D] = field(init=False)

    # Configuraciones
    mpConfig: MPConfig | None = field(default_factory=_MP2D_mpConfig_defecto)
    refConfig: RefConfig | None = field(default_factory=_MP2D_refConfig_defecto)
    panConfig: PanConfig | List[PanConfig] | None = None
    controlConfig: ControlConfig | None = field(default_factory=_MP2D_controlConfig_defecto)
    
    # Información
    t: List[float] = field(init=False)
    it: int = field(init=False)

    N_S: int = field(init=False)
    N_s: List[int] = field(init=False)

    adimInfo: AdimInfo = field(init=False)

    # Registros
    RMRec: List[np.ndarray] = field(init=False)

    CpRec: List[List[np.ndarray]] = field(init=False)
    dFRec: List[np.ndarray] = field(init=False)

    CxyTotalRec: List[np.ndarray] = field(init=False)
    CmTotalRec: List[float] = field(init=False)

    CxyRec: List[np.ndarray] = field(init=False)
    CmRec: List[np.ndarray] = field(init=False)

    tRes: List[float] = field(init=False)

    def __post_init__(self):
        # En el caso de un único sólido, se convierte a lista el generador, en el caso de que se haya pasado un generador.
        if not isinstance(self.cinematicas, list):
            self.cinematicas = [self.cinematicas]

        self.N_S = len(self.cinematicas)    
        self.it = -1

        # Inicialización de listas temporales.
        self.t = []
        self.conjuntos = []

        self.RMRec = []
        
        self.CpRec = []
        self.dFRec = []

        self.CxyTotalRec = []
        self.CmTotalRec = []

        self.CxyRec = []
        self.CmRec = []

        self.tRes = []

        # Gestión de las configuraciones. Se completan valores ausentes con los valores por defecto.
        self.refConfig = _completar_dic(self.refConfig, _MP2D_refConfig_defecto)
        self.mpConfig = _completar_dic(self.mpConfig, _MP2D_mpConfig_defecto)
        self.controlConfig = _completar_dic(self.controlConfig, _MP2D_controlConfig_defecto)

        if self.panConfig is None:
            cfg = _MP2D_panConfig_defecto()
            self.panConfig = [cfg.copy() for _ in range(self.N_S)]

        if isinstance(self.panConfig, dict):
            cfg = _completar_dic(self.panConfig, _MP2D_panConfig_defecto)
            self.panConfig = [cfg.copy() for _ in range(self.N_S)]
            
        elif isinstance(self.panConfig, list):
            if len(self.panConfig) != self.N_S:
                raise ValueError(f'Longitud de panConfig no válida. Cuando panConfig es una lista, debe tener {self.N_S} elementos.')
            self.panConfig = [
                _completar_dic(cfg, _MP2D_panConfig_defecto)
                for cfg in self.panConfig
            ]
        else:
            raise TypeError('panConfig debe ser dict, list o None.')
        
        if self.mpConfig['nombres'] is None:
            self.mpConfig['nombres'] = [f'Sólido {i}' for i in range(self.N_S)]

        self.SELInfo = _MP2D_SELInfo_defecto(self.mpConfig)
        
        # Se calcula la presión dinámica de referencia.
        self.refConfig['Q'] = 0.5 * self.refConfig['rho'] * self.refConfig['V'] ** 2    

        # Se calculan relaciones adimensionales de referencia.
        Re = self.refConfig['V'] * self.refConfig['l'] / self.refConfig['nu']
        Ma = self.refConfig['V'] / self.refConfig['a']
        Fr = self.refConfig['V'] / np.sqrt(self.refConfig['g'] * self.refConfig['l'])

        self.adimInfo = {}

        self.adimInfo['Re'] = float(Re)
        self.adimInfo['Ma'] = float(Ma)
        self.adimInfo['Fr'] = float(Fr)

        self.adimInfo['_Re'] = float(1 / Re)
        self.adimInfo['_Fr2'] = float(1 / Fr ** 2)

        self.adimInfo['Ma2'] = float(Ma ** 2)
        self.adimInfo['Ma2_Re'] = float(Ma ** 2 / Re)
        self.adimInfo['Ma2_Fr2'] = float(Ma ** 2 / Fr ** 2)


        # Se purgan los generadores en los casos aeroelásticos.
        if self.mpConfig['aeroelasticidad']:
            [next(cinematica) for cinematica in self.cinematicas]
        if self.mpConfig['RMAeroelasticidad']:
            next(self.RM)


# =============================================================================
# Métodos que articulan el time-stepping.
# =============================================================================
    def simular(
            self, 
            n_iter: int | None = None,
    ):
        """
        Metodo para simular el experimento una vez inicializado y configurado correctamente.

        En el caso no estacionario, el experimento siempre comienza con los sólidos con un único vórtice (cuyas intensidades
        son nulas o tal que satisfagan la condición de Kutta, segun MP2D.mpConfig['vorticeArranque']) y los siguiente pasos 
        de simulación se corresponden a instantes posteriores, resolviendo el time-stepping.

        En el caso estacionario, cada paso de simulación resuelve un nuevo instante inicial, con las estelas de los sólidos
        componiéndose únicamente de un vórtice de arranque, con sus intensidades siguiendo el mismo comportamiento. 
        Parameters
        ----------
        n_iter : int | None
            Número de instantes a simular.

            - Si `not MP2D.mpConfig['aeroelasticidad'] and (MP2D.controlConfig['instantesDeSimulacion'] or MP2D.mpConfig['mostrarProgreso'])':
                antes de la simulación el `Tipos.GeneradorCinematica` en la primera posición de MP2D.cinematicas se recorre 
                para la contabilización de los instantes disponibles, copiando sus valores para construir otro generador 
                para usar en la simulación.

                Esto se hace notar ya que este comportamiento podría comprometer la funcionalidad implementada en
                generadores aeroelásticos particulares.

            - Si `MP2D.mpConfig['aeroelasticidad']: debe ser un entero positivo.
            - En los casos restantes, se itera hasta que se extinga por primera vez algún generador, ya sea
            un `Tipos.GeneradorCinematica` o un `Tipos.GeneradorRM`.
        """

        if not self.mpConfig['aeroelasticidad']:
            if self.controlConfig['instantesDeSimulacion'] or self.mpConfig['mostrarProgreso']:
                # Se itera sobre el primer generador para conocer el número de instantes disponibles, 
                # asumiendo consistencia con los otros generadores (en ejecución se comprueba)
                
                gen0 = self.cinematicas[0]
                cinematicas0 = []   # Almacén de la cinemática en los diferentes instantes.
                n_disp = 0  # Número de pasos de tiempo disponibles.

                for cin0 in gen0:
                    cinematicas0.append(cin0)
                    n_disp += 1

                self.cinematicas[0] = iter(cinematicas0)    # Se reemplaza el generador extinguido por uno nuevo

                if n_iter is None:
                    n_iter = n_disp
                else: 
                    if n_iter > n_disp:
                        raise ValueError(f'n_iter={n_iter} excede el número de pasos disponibles en la cinemática ({n_disp}).')
        else:
            if n_iter is None:
                raise ValueError(f'n_iter debe ser un entero positivo cuando MP2D.mpConfig["aeroelasticidad"] = True')
            
        # Se simula
        if n_iter is not None:
            try:
                for _ in tqdm(range(n_iter), total=n_iter, desc='Simulando', unit=' pasos', disable=not self.mpConfig['mostrarProgreso']):
                    self._avanzar()
            except StopIteration:
                raise StopIteration(f'La simulación terminó prematuramente porque algún generador se extinguió antes de la iteración n_iter={n_iter}')
        else:
            while True:
                try:
                    self._avanzar()
                except StopIteration:
                    print('Simulación terminó.')
                    if len(self.t) != len(self.RM):
                        warnings.warn('La simulación terminó porque algún Tipos.GeneradorRM se extinguió antes' \
                        'que cualquier Tipos.GeneradorCinematica, por lo tanto existen listas temporales del último' \
                        'paso de tiempo pueden estar incompletas.' \
                        'No se implementó ningún tipo de limpieza de elementos excedentes.' \
                        'Se puede consultar el atributo MP2D.it para conocer el último paso completo de simulación.')
                    break
    
    def _avanzar(self):
        """
        Método privado que resuelve un paso de tiempo.

        Retoma el número de iteración almacenado en `MP2D.it`.
        """
        inicio = time()
        it = self.it + 1    # Número de la iteración que se va a resolver.
        
        # -----------------------------------------------------------------------------
        # Cinemática de los sólidos.
        # -----------------------------------------------------------------------------
        # Se obtiene la cinemática de los sólidos.

        if not self.mpConfig['aeroelasticidad']:
            cinematicas = [next(cinematica) for cinematica in self.cinematicas]
        else:
            cinematicas = [cinematica.send((self, iS, it)) for iS, cinematica in enumerate(self.cinematicas)]

        t, RO, theta, VO, wTM, r_xy, VRelPC_xy = zip(*cinematicas)     # Tuplas de longitud (N_S,).
        
        # Construcción de las ternas móviles.
        RO = np.concatenate(RO, axis=1)     # (2, N_S)
        theta = np.array(theta)             # (N_S,)
        VO = np.concatenate(VO, axis=1)     # (2, N_S)
        wTM = np.array(wTM)                 # (N_S,)

        TM = TernasMoviles2D(RO, theta, VO, wTM)
        
        # Transformación de las coordenadas de los nodos de los sólidos, al sistema fijo.
        R_XY = [TM.r2R_1TM(r, s) for s, r in enumerate(r_xy)]

        # -----------------------------------------------------------------------------
        # Algunas verificaciones.
        # -----------------------------------------------------------------------------

        # Verificación de la consistencia de los tiempos almacenados en cada cinemática.
        if not self.mpConfig['estacionario']:
            t0 = t[0]
            if any(abs(t0 - ti) > 1e-12 for ti in t):
                raise ValueError(f'Desacuerdo de tiempos entre al menos 2 sólidos en el paso {it}, tiempo {t0}.')
            # Verificación de que el tiempo sea creciente.
            if it > 0 and t0 <= self.t[-1]:
                raise ValueError(f'El tiempo decreció en el paso {it}, pasó de {self.t[-1]} a {t0}.')

        t = t[0]
        self.t.append(t)    # Se agrega el tiempo si es válido.

        if it == 0:
            # Se determina el atributo N_s. Nunca más se vuelve a asignar.
            self.N_s = []
            for i in range(self.N_S):
                self.N_s.append(r_xy[i].shape[1] - 1)
        else:
            # Verificación de que se conservaron las cantidades de paneles por sólidos.
            for s, N_s_0 in enumerate(self.N_s):
                N_s_t = r_xy[s].shape[1] - 1
                if N_s_t != N_s_0:
                    raise ValueError(f'El sólido en la posición {s} pasó de tener {N_s_0} paneles a tener {N_s_t} paneles, en el instante {t}. Los sólidos deben tener un numéro constante de paneles a lo largo del tiempo')
        
        # Verificación de bordes de fuga cerrados.
        for s in range(self.N_S):
            if not np.array_equal(R_XY[s][:, 0], R_XY[s][:, -1]):
                raise ValueError(f'Borde de fuga no cerrado en el sólido en la posición {s}.')

        # -----------------------------------------------------------------------------
        # Enrrollamientos de las estelas
        # -----------------------------------------------------------------------------
        
        # Se obtienen las nuevas posiciones de las estelas y sus intensidades unitarias.
        # Se utiliza la información de las posiciones de los nuevos sólidos (particularmente, su BF)
        # además de información del conjunto de sólidos y sus estelas en el paso anterior, si hubiera.
        # En el caso estacionario, las estelas solo se componen de vórtices de arranque.
        R_XY_estelas, intensidades_estelas = self._enrrollamiento(R_XY)

        # Las singularidades de las estelas (y sus nodos) se ordenan de más nuevos -> más viejos. 
        #
        # Esto se debe a cómo se define _Paneles2D.Paneles2D y sus normales, y la convención utilizada
        # en el desarrollo teórico.

        # -----------------------------------------------------------------------------
        # Construcción y resolución del SEL
        # -----------------------------------------------------------------------------

        # Construcción de la instancia almacén del nuevo conjunto de sólidos y sus estelas.
        conjuntoActual = ConjuntoSolidos2D(R_XY, VRelPC_xy, TM, R_XY_estelas, intensidades_estelas, self.panConfig)

        self.conjuntos.append(conjuntoActual)

        # Velocidades inducidas (valores principales) en los puntos de colocación, tanto por los paneles de los sólidos como los de sus estelas
        # suponiendo intensidades unitarias
        VIndPorSolidos, VIndPorEstelas = self._VInd_intensidadUnitaria(conjuntoActual)

        # Construcción del SEL a partir de las velocidades inducidas y el conjunto actual (contiene las normales a los PC y las velocidades cinemáticas en los PC).
        A, b = self._SELConstructor(conjuntoActual, VIndPorSolidos, VIndPorEstelas) 
    
        # Resolución del SEL.  
        x = self._SELSol(A, b)

        # Asignación de las intensidades resueltas a los paneles.
        i = 0
        for s, solido in enumerate(conjuntoActual.solidos):
            N_s = conjuntoActual.PC_XY[s].shape[1]
            solido.intensidades = x[i:i+N_s]    # Asignación en paneles de sólidos.

            i += N_s
            
            # Asignación en el último panel de estela creado.
            if not self.mpConfig['estacionario'] and it > 0: # Instantes posteriores al inicial.
                conjuntoActual.estelas[s].intensidades[0] = solido.intensidades[-1] - solido.intensidades[0]

            # Instante inicial.
            else:       
                if self.mpConfig['vorticeArranque']:
                    conjuntoActual.vortices[s].intensidades[0] = solido.intensidades[-1] - solido.intensidades[0]
                else:    
                    conjuntoActual.vortices[s].intensidades[0] = 0

        # -----------------------------------------------------------------------------
        # Cálculos posteriores (aerodinámicos)
        # -----------------------------------------------------------------------------

        # Se actualizan las velocidades inducidas con las intensidades obtenidas (antes se habian considerado unitarias).
        VIndPorSolidos, VIndPorEstelas = self._actualizarVIndUnitaria(conjuntoActual, VIndPorSolidos, VIndPorEstelas)

        # Se obtiene la velocidad inducida resultante (componente principal) en cada punto de colocación.
        conjuntoActual.VInd = self._VInd_resultante(VIndPorSolidos, VIndPorEstelas)

        # Se obtiene la componente regular de las velocidades inducidas en cada punto de colocación.
        VInd_regular = self._VInd_regular(conjuntoActual)

        # Se obtiene la velocidad inducida total (tanto en términos matemáticos, como cinemáticos en relación al sistema fijo)
        for i in range(self.N_S):
            conjuntoActual.VInd[i] += VInd_regular[i]


        # Se obtiene el punto de toma de momentos.
        if self.RM is not None:
            if not self.mpConfig['RMAeroelasticidad']:
                self.RMRec.append(next(self.RM))
            else:
                self.RMRec.append(self.RM.send((self, it)))

        # Se realizan los cálculos aerodinámicos y se guardan los resultados.
        dF, Cp, CxyTotal, CmTotal, Cxy, Cm = self.calculosAerodinamicos(it)

        self.CpRec.append(Cp)
        self.dFRec.append(dF)
        self.CxyTotalRec.append(CxyTotal)
        self.CxyRec.append(Cxy)
        if self.RM is not None:
            self.CmTotalRec.append(CmTotal)
            self.CmRec.append(Cm)
        
        # Se actualiza el valor de la última iteración resuelta.
        self.it += 1
        self.tRes.append(time() - inicio)

# =============================================================================
# Métodos que asisten a MP2D._avanzar(...)
# =============================================================================
# Algunos métodos presentados aquí, podrían ser funciones y estar definidas en
# un módulo distinto, pues no dependen directamente de una instancia de MP2D.
# Incluso, pequeñas dependencias que existieron fueron mitigadas para que lo 
# anterior fuera estrictamente cierto.
#
# Sin embargo, se las deja en este módulo como métodos, pues guardan íntima
# relación con el bucle de simulación y en un futuro podrían requerir acceder
# a otra información contenida en la instancia de MP2D simulada, como por 
# ejemplo configuraciones de control, sin tener que pasar esto como parámetros
# adicionales.

# Velocidades inducidas
# =============================================================================
    def _VInd_intensidadUnitaria(
            self,
            conjunto: ConjuntoSolidos2D,
        ) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
        """
        Método que calcula las velocidades inducidas por cada panel en los puntos de colocación considerando intensidades unitarias.

        .. note::
            Actualmente no utiliza la instancia de MP2D.

        Parameters
        ----------
        conjunto : _ConjuntoSolidos2D.ConjuntoSolidos2D
            Conjunto de sólidos y sus estelas que inducen velocidades y cuyos puntos de colocación (de los sólidos)
            son los puntos de evaluación de las velocidades inducidas.
        
        Returns
        -------
        VIndPorSolidos : List[List[np.ndarray]]
            Velocidades inducidas por los sólidos del conjunto (**asumiendo intensidades unitarias**).
            
            - El primer índice indica el sólido influenciado.
            - El segundo índice indica el sólido que influencia.

            Luego, si el sólido influenciado `i` tiene M puntos de colocación y el `j` que
            influencia tiene N paneles, el arreglo `VIndPorSolidos[i][j]` tiene shape (2, N, M).
        VIndPorEstelas : List[List[np.ndarray]]
            Velocidades inducidas por las estelas del conjunto (**asumiendo intensidades unitarias**).
            - El primer índice indica el sólido influenciado.
            - El segundo índice indica la estela que influencia.

            Luego, si el sólido influenciado `i` tiene M puntos de colocación y la estela `j` que
            influencia tiene N paneles, el arreglo `VIndPorEstelas[i][j]` tiene shape (2, N, M).
        """
        N_S = len(conjunto.solidos)

        instanteInicial = False 
        if not conjunto.estelas:    # En el primer instante no hay conjunto de estelas (dobletes), aunque sí existe el vórtice de arranque.
            instanteInicial = True

        # Se incializa las listas de salida.
        VIndPorSolidos = []
        VIndPorEstelas = []
        for i in range(N_S):    # Bucle de sólidos influenciados
            R_XY = conjunto.PC_XY[i]    # Puntos de colocación.
            M = R_XY.shape[1]

            # Se recorren los sólidos (y sus estelas) que influencian.
            VInd_i = []
            VInd_i_estela = []            
            for j, solidoQueInfluencia in enumerate(conjunto.solidos):
                
                if not instanteInicial:
                    estelaQueInfluencia = conjunto.estelas[j]

                N = conjunto.PC_XY[j].shape[1]

                # En esta instancia, el único caso en el que el punto de evaluación puede encontrarse
                # en la singularidad que influencia es en el caso de un sólido autoinfluenciándose.
                # En cuyo caso, N = M y los elementos de la diagonal son las autoinfluencias.
                if j == i:
                    ubicaciones = - np.eye(N, M)
                else:
                    ubicaciones = None

                VInd_i.append(solidoQueInfluencia.VInd_xy(R_XY, ubicaciones, afuera=True, intensidadesUnitarias=True)) # Influencia sólidos.
                
                vorticeQueInfluencia = conjunto.vortices[j]     # Influencia del vórtice de arranque.
                VInd = vorticeQueInfluencia.VInd_xy(R_XY, ubicaciones=None, intensidadesUnitarias=True)
                
                if not instanteInicial:     # Influencia de los paneles de dobletes, si existen.
                    VEst = estelaQueInfluencia.VInd_xy(R_XY, ubicaciones=None, afuera=True, intensidadesUnitarias=True)
                    VInd = np.concatenate([VEst, VInd], axis=1)     # Se juntan las influencias de dobletes y vórtices,
                                                                    # respetando la convención de últimos -> primeros
                                                                    # de los elementos de la estela.
                VInd_i_estela.append(VInd)

            VIndPorSolidos.append(VInd_i)
            VIndPorEstelas.append(VInd_i_estela)
        return VIndPorSolidos, VIndPorEstelas

    def _actualizarVIndUnitaria(
            self,
            conjunto: ConjuntoSolidos2D,
            VIndPorSolidos: List[List[np.ndarray]], 
            VIndPorEstelas: List[List[np.ndarray]]
    ) -> List[np.ndarray]:
        """
        Método que a partir de las intensidades de las singularidades resueltas y las velocidades inducidas
        por cada panel en cada punto de colocación y considerando intensidades unitarias, computan las 
        velocidades inducidas reales.

        .. note::
            Actualmente no utiliza la instancia de MP2D.

        Parameters
        ----------
        conjunto : _ConjuntoSolidos2D.ConjuntoSolidos2D
            Conjunto de sólidos y sus estelas que inducen velocidades y cuyos puntos de colocación (de los sólidos)
            son los puntos de evaluación de las velocidades inducidas.
        VIndPorSolidos : List[List[np.ndarray]]
            Velocidades inducidas por los sólidos del conjunto (**asumiendo intensidades unitarias**).
            - El primer índice indica el sólido influenciado.
            - El segundo índice indica el sólido que influencia.

            Luego, si el sólido influenciado `i` tiene M puntos de colocación y el `j` que
            influencia tiene N paneles, el arreglo `VIndPorSolidos[i][j]` tiene shape (2, N, M).
        VIndPorEstelas : List[List[np.ndarray]]
            Velocidades inducidas por las estelas del conjunto (**asumiendo intensidades unitarias**).
            - El primer índice indica el sólido influenciado.
            - El segundo índice indica la estela que influencia.

            Luego, si el sólido influenciado `i` tiene M puntos de colocación y la estela `j` que
            influencia tiene N paneles, el arreglo `VIndPorEstelas[i][j]` tiene shape (2, N, M).

        Returns
        -------
        VIndPorSolidos : List[List[np.ndarray]]
            Velocidades inducidas por los sólidos del conjunto (**reales**).

            Mantiene la misma forma de la entrada.
        VIndPorEstelas : List[List[np.ndarray]]
            Velocidades inducidas por las estelas del conjunto (**reales**).

            Mantiene la misma forma de la entrada.
        """
        N_S = len(conjunto.solidos)

        instanteInicial = False
        if not conjunto.estelas:
            instanteInicial = True

        for i in range(N_S):
            for j in range(N_S):
                VIndPorSolidos[i][j] *= conjunto.solidos[j].intensidades[None,:,None]
                if instanteInicial:
                    VIndPorEstelas[i][j] *= conjunto.vortices[j].intensidades[None,:,None]
                else:
                    intensidades = np.concatenate([conjunto.estelas[j].intensidades, conjunto.vortices[j].intensidades], axis=0)
                    VIndPorEstelas[i][j] *= intensidades[None,:,None]
        return VIndPorSolidos, VIndPorEstelas

    def _VInd_resultante(
            self,
            VIndPorSolidos: List[List[np.ndarray]],
            VIndPorEstelas: List[List[np.ndarray]],
    ) -> List[np.ndarray]:
        """
        Método que calcula la resultante de las velocidades inducidas en cada punto de colocación.

        .. note::
            Actualmente no utiliza la instancia de MP2D.

        Parameters
        ----------
        VIndPorSolidos : List[List[np.ndarray]]
            Velocidades inducidas por los sólidos del conjunto (**reales**).
            - El primer índice indica el sólido influenciado.
            - El segundo índice indica el sólido que influencia.

            Luego, si el sólido influenciado `i` tiene M puntos de colocación y el `j` que
            influencia tiene N paneles, el arreglo `VIndPorSolidos[i][j]` tiene shape (2, N, M).
        VIndPorEstelas : List[List[np.ndarray]]
            Velocidades inducidas por las estelas del conjunto (**reales**).
            - El primer índice indica el sólido influenciado.
            - El segundo índice indica la estela que influencia.

            Luego, si el sólido influenciado `i` tiene M puntos de colocación y la estela `j` que
            influencia tiene N paneles, el arreglo `VIndPorEstelas[i][j]` tiene shape (2, N, M).

        Returns
        -------
        VInd : List[np.ndarray], each shape (2, N_s):
            Velocidades inducidas totales en los puntos de colocación de cada sólido.
        """
        VInd = []
        for i in range(len(VIndPorSolidos)):
            M = VIndPorSolidos[i][0].shape[2]
            VInd_i = np.zeros((2, M))
            for j in range(len(VIndPorSolidos[i])):
                VInd_i += np.sum(VIndPorSolidos[i][j], axis=1)
                VInd_i += np.sum(VIndPorEstelas[i][j], axis=1)
            VInd.append(VInd_i)
        return VInd
    
    def _SELConstructor(
            self,
            conjunto: ConjuntoSolidos2D,
            VIndPorSolidos: List[List[np.ndarray]],
            VIndPorEstelas: List[List[np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Método que construye el SEL del método de los paneles no estacionario de la forma:

        Ax = b = b' - Cw @ xw
        
        donde

        - A: es la matriz de coeficientes que contiene la influencia de los sólidos sobre ellos mismos, y la
        del último panel de estela creado (intensidad desconocida, Kutta) sobre ellos.

        - x: intensidades de los paneles de los sólidos.

        - xw: intensidades de los paneles de estela, excepto la de los últimos creados.

        - Cw: coeficientes de influencia de los paneles de estela (excepto los de los últimos creados)
        sobre los sólidos.

        - b': componentes normales de las velocidades cinemáticas de los puntos de colocación de los sólidos.

        .. note::
            Actualmente no utiliza la instancia de MP2D.

        Parameters
        ----------
        conjunto : _ConjuntoSolidos2D.ConjuntoSolidos2D
            Conjunto de sólidos y sus estelas que inducen velocidades y cuyos puntos de colocación (de los sólidos)
            son los puntos de evaluación de las velocidades inducidas.
        VIndPorSolidos : List[List[np.ndarray]]
            Velocidades inducidas por los sólidos del conjunto (**asumiendo intensidades unitarias**).
            - El primer índice indica el sólido influenciado.
            - El segundo índice indica el sólido que influencia.

            Luego, si el sólido influenciado `i` tiene M puntos de colocación y el `j` que
            influencia tiene N paneles, el arreglo `VIndPorSolidos[i][j]` tiene shape (2, N, M).
        VIndPorEstelas : List[List[np.ndarray]]
            Velocidades inducidas por las estelas del conjunto (**asumiendo intensidades unitarias**).
            - El primer índice indica el sólido influenciado.
            - El segundo índice indica la estela que influencia.

            Luego, si el sólido influenciado `i` tiene M puntos de colocación y la estela `j` que
            influencia tiene N paneles, el arreglo `VIndPorEstelas[i][j]` tiene shape (2, N, M).

        Returns
        -------
        A : np.ndarray
            Matriz de coeficientes del SEL.
        b : np.ndarray
            Vector independiente del SEL.
        """
        N_S = len(conjunto.solidos)

        instanteInicial = False 
        if not conjunto.estelas:    # En el primer instante no hay conjunto de estelas (dobletes), aunque sí existe el vórtice de arranque.
            instanteInicial = True

        # Inicialización de las matrices y vectores definidos por bloques como listas, luego se concatenan.
        A = []
        Cw = []
        bPrima = []
        xw = []

        for i in range(N_S):    # Bucle de cada bloque fila (sólido influenciado).
            # Normales de los paneles del sólido influenciado.
            normales_i = conjunto.normales_XY[i]     # (2, N_i)

            # Componentes normales de las velocidades cinemáticas
            VCinematica = conjunto.VPC_XY[i]    # (2, N_i)
            bPrima.append(np.einsum('ji,ji->i', VCinematica, normales_i))

            # Intensidades de los vórtices de arranque.
            if instanteInicial:
                xw_i = np.array([]) # En el primer instante se desconocen.
            else:
                xw_i = conjunto.vortices[i].intensidades    

                # Intensidades conocidas de los dobletes de la estela.
                if len(conjunto.estelas[0].intensidades) >= 2:
                    xw_i = np.concatenate([conjunto.estelas[i].intensidades[1:], xw_i], axis=0)

            xw.append(xw_i)

            # Se inicializan los contenedores de los bloques (columna) que conforman los bloques filas
            # de las matrices de coeficientes de influencia.
            A_i = []
            Cw_i = []
            for j in range(N_S):    # Bucle de cada bloque columna (sólidos y estelas que influencian).
                # Se recuerda que los coeficientes de influencia son las componenetes normales de
                # las velocidades inducidas de cada panel sobre cada punto de colocación, suponiendo
                # intensidadese unitarias.

                # Influencia de solidos
                Aij = VIndPorSolidos[i][j]     # (2, N_j, N_i)
                Aij = np.einsum('kji,ki->ij', Aij, normales_i)    # (N_i, N_j)
                
                # Influencia de estelas
                Cwij = VIndPorEstelas[i][j]     # (2, N_jEstela,N_i)
                Cwij = np.einsum('kji,ki->ij', Cwij, normales_i)    # (N_i, N_jEstela)
                
                # Kutta (se incorpora la influencia del último panel de estela a la matriz de coeficientes del SEL)
                Aij[:,0] -= Cwij[:, 0]
                Aij[:,-1] += Cwij[:, 0]

                # Se remueve la influencia del último panel de estela en la matriz del lado derecho del SEL.
                Cwij = Cwij[:,1:]

                A_i.append(Aij)
                Cw_i.append(Cwij)

            A.append(A_i)
            Cw.append(Cw_i)

        # Se concatenan los bloques.
        A = np.block(A)
        Cw = np.block(Cw)
        xw = np.concatenate(xw, axis=0)
        bPrima = np.concatenate(bPrima, axis=0)

        # Se obtiene el vector independiente total.
        if instanteInicial:
            b = bPrima
        else:
            b = bPrima - (Cw @ xw)
        return A, b

    def _SELSol(
            self, 
            A: np.ndarray, 
            b: np.ndarray, 
    ):
        """
        Método para resolver el sístema de ecuaciones lineales

        Parameters
        ----------
        A : np.ndarray
            Matriz de coeficientes.
        b : np.ndarray
            Vector independiente
        Returns
        -------
        x : np.ndarray
            Vector solución.
        """
        metodo = self.mpConfig['SELSol']
        
        if metodo == 'svd':
            x, *_ = lstsq(A, b, lapack_driver='gelsd')
        elif metodo == 'qr':
            x, *_ = lstsq(A, b, lapack_driver='gelsy')
        elif metodo =='lu':
            try:
                x = solve(A, b)
            except LinAlgError as e:
                raise RuntimeError(f"El método {metodo} no soportó la mala condición de la matriz de coeficientese. Probar con otro método.")
        else:
            raise ValueError(f"Metodo '{metodo}' no implementado.")
        
        if self.mpConfig['SELInfo']:
            self.SELInfo['nCond'].append(np.linalg.cond(A))
            self.SELInfo['normaRes'].append(np.linalg.norm(A @ x - b))
            self.SELInfo['normaInd'].append(np.linalg.norm(b))
        return x

# Enrrollamientos de las estelas
# =============================================================================

    def _enrrollamiento(
            self,
            R_XY: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Método que lleva a cabo el enrrollamiento de la estela, obteniendo la nueva posición de los nodos de la estela y 
        sus vectores de intensidades unitarias.

        Parameters
        ----------
        R_XY : List[np.ndarray] each shape (2, N_s + 1)
            Lista de las posiciones de los nodos de cada sólido.
        
        Returns
        -------
        R_XY_estelas : List[np.ndarray] each shape (2, N_w_s + 1)
            Listas de las posiciones de los nodos de la estela de cada sólido en el nuevo instante.
        intensidades_estelas : List[np.ndarray] each shape (N_w_s)
            Lista de las intensidades unitarias de la estela de cada sólido en el nuevo instante.

        Notes
        -----
        
        Las singularidades de las estelas (y sus nodos) se ordenan de más nuevos -> más viejos. 
        
        Esto se debe a cómo se define _Paneles2D.Paneles2D y sus normales, y la convención utilizada
        en el desarrollo teórico.
        """
        it = self.it + 1
        # Primer instante o estacionario.
        if it == 0 or self.mpConfig['estacionario']:
            R_XY_estelas = [R_XY_s[:, 0:1] for R_XY_s in R_XY]          # En el primer instante se ubican los vórices de arranque en los BF.
            intensidades_estelas = [[1.,] for _ in range(self.N_S)]     # Se inicializan las intensidades de forma unitaria.

        # Instantes posteriores
        else:
            # Obtención las nuevas posiciones de los paneles de estela presentes en el instante anterior.
            dT = self.t[-1] - self.t[-2]
            R_XY_estelas = self._integracionEstelas(self.conjuntos[-1], dT)
            
            # Se agregan los nuevos nodos de las estelas (los BF), anteponiéndolos en los arreglos, respetando la convención más nuevos -> más viejos.
            for s in range(len(R_XY_estelas)):
                R_XY_estelas[s] = np.concatenate([R_XY[s][:,0:1], R_XY_estelas[s]], axis=1) 
            
            # Se anteponen las nuevas intensidades untarias, en el segundo instante y en los posteriores.
            if it == 1:
                intensidades_estelas = [np.concatenate([[1.,], self.conjuntos[-1].vortices[s].intensidades], axis=0) for s in range(self.N_S)]  # Segundo instante.
            else:
                intensidades_estelas = [np.concatenate([[1.,], self.conjuntos[-1].estelas[s].intensidades, self.conjuntos[-1].vortices[s].intensidades], axis=0) for s in range(self.N_S)]  # Posteriores.
        
        return R_XY_estelas, intensidades_estelas

    def _integracionEstelas(
            self,
            conjunto: ConjuntoSolidos2D,
            dT: float,
    ) -> np.ndarray:
        """
        Método que integra las velocidades de los nodos de la estela para obtener las posiciones de estos en el 
        nuevo instante.

        También aplica la corrección por rigidez configurada. Ver Tipos.MPConfig.

        .. note::

        El método utilizado es la integración directa de Euler. Más métodos podrían ser evaluados en el futuro.

        Parameters
        ----------
        conjunto : _ConjuntosSolidos2D.ConjuntoSolidos2D
            Conjunto de sólidos y sus estelas en el último paso resuelto, anterior a aquel en el que
            se quiere conocer la posición de la estela.
        dT : float
            Diferencia de tiempo en el que se realizará la integración.

        Returns
        -------
        R_XY_estelas : List[np.ndarray]
            Las nuevas posiciones de los nodos de estela que existían en el paso anterior.
        """
        # Se obtienen las velocidades inducidas totales en los nodos de estelas.
        VIndEstelas = self._VIndEstelas(conjunto)

        instanteInicial = False     # Se verifica si el último instante era el inicial.
        if not conjunto.estelas:
            instanteInicial = True

        rigidez = self.mpConfig['rigidez']  # Parámetro de rigidez de la integración. Ver Tipos.MPConfig.

        # El último instante es el inicial (solo vórtices de arranque).
        if instanteInicial:
            R_XY_estelas = [vortice.r0_xy + dT * VIndEstelas[s] for s, vortice in enumerate(conjunto.vortices)] # Se integran las posiciones
            for i in range(self.N_S):
                R_XY_estelas[i][:, :] = rigidez * R_XY_estelas[i][:, :] + (1 - rigidez) * conjunto.vortices[i].r0_xy  # Se aplica la rigidez.

        # El último instante no es el inicial.
        else:
            R_XY_estelas = [estela.r_xy + dT * VIndEstelas[s] for s, estela in enumerate(conjunto.estelas)] # Se integran las posiciones.
            for i in range(self.N_S):
                R_XY_estelas[i][:, :1] = rigidez * R_XY_estelas[i][:, :1] + (1 - rigidez) * conjunto.estelas[i].r_xy[:,:1]    # Se aplica la rigidez.

        return R_XY_estelas

    def _VIndEstelas(
            self, 
            conjunto: ConjuntoSolidos2D,
        ) -> List[np.ndarray]:

        """
        Método para calcular las velocidades inducidas en los nodos de la estela del conjunto, por el mismo conjunto.

        .. note::
            Actualmente no utiliza la instancia de MP2D.

        Parameters
        ----------
        conjunto : _ConjuntoSolidos2D.ConjuntoSolidos2D
            Conjunto al cual pertenecen las estelas a las cuales pertenecen los nodos cuyas velocidades quieren conocerse.
        
        Returns
        -------
        VIndEstelas : List[np.ndarray] each shape (2, N_w_s + 1)
            Velocidades inducidas por el conjunto en los nodos de las estelas.
        """
        N_S = len(conjunto.solidos)

        instanteInicial = False
        if not conjunto.estelas:
            instanteInicial = True
        
        VIndPorSolidos = []
        VIndPorEstelas = []
        for i in range(N_S):    # Bucle de estelas influenciadas.
            # Se obtienen los puntos de evaluación.
            if instanteInicial:
                R_XY = conjunto.vortices[i].r0_xy
            else:
                R_XY = conjunto.estelas[i].r_xy
            M = R_XY.shape[1]

            # Se recorren los sólidos (y sus estelas) que influencian.
            VInd_i = []
            VInd_i_estela = []
            for j, solidoQueInfluencia in enumerate(conjunto.solidos):
                
                # Influencia de sólidos
                # ---------------------
                N = conjunto.PC_XY[j].shape[1]
                
                # Si el sólido es aquel al que pertenece la estela, se resuelve la indeterminación en los paneles adyacentes al BF.
                if j == i:  
                    ubicaciones = np.zeros((N, M))
                    ubicaciones[0, 0] = 1
                    ubicaciones[-1, 0] = 2
                # Si no, las influencias son todas externas
                else:
                    ubicaciones = None

                VInd_i.append(solidoQueInfluencia.VInd_xy(R_XY, ubicaciones, afuera=True, intensidadesUnitarias=False))
                
                # Influencia de estelas
                # ---------------------
                # En el instante anterior solo estaba el vórtice de arranque.
                if instanteInicial:
                    if j == i:
                        VInd = np.zeros((2, 1, M))  # La autoinfluencia del vórtice es nula.
                    else:
                        vorticeQueInfluencia = conjunto.vortices[j]
                        VInd = vorticeQueInfluencia.VInd_xy(R_XY, ubicaciones=None, intensidadesUnitarias=False)
                
                # En el instante anterior ya existen paneles de dobletes además del vórtice de arranque.
                else:
                    vorticeQueInfluencia = conjunto.vortices[j]
                    estelaQueInfluencia = conjunto.estelas[j]

                    N = len(estelaQueInfluencia.x2_xyp)
                    # Autoinfluencia de la estela.
                    if j == i:
                        # Vórtice de arranque.
                        ubicaciones = np.zeros((1, M))
                        ubicaciones[0,-1] = -1  # Autoinfluencia del vórtice nula.
                        VInd = vorticeQueInfluencia.VInd_xy(R_XY, ubicaciones, intensidadesUnitarias=False)

                        # Dobletes.
                        ubicaciones = np.zeros((N, M))  # N + 1 = M, i = j
                        for ii in range(N):
                            for jj in range(M): 
                                # Se resuelven las indeterminaciones de las influencias de los paneles adyacentes a los nodos 
                                if ii == jj:
                                    ubicaciones[ii, jj] = 1
                                if jj == ii + 1:
                                    ubicaciones[ii, jj] = 2
                    else:
                        # Vórtice de arranque.
                        VInd = vorticeQueInfluencia.VInd_xy(R_XY, ubicaciones=None, intensidadesUnitarias=False)    # (2,1,M)

                        # Dobletes.
                        ubicaciones = None
                    VEst = estelaQueInfluencia.VInd_xy(R_XY, ubicaciones, True, intensidadesUnitarias=False)

                    VInd = np.concatenate([VEst, VInd], axis=1) # Se concatenan las velocidades inducidas por vórtices y por dobletes.

                VInd_i_estela.append(VInd)

            VIndPorSolidos.append(VInd_i)
            VIndPorEstelas.append(VInd_i_estela)

            VIndEstelas = self._VInd_resultante(VIndPorSolidos, VIndPorEstelas) # Se calcula la resultante en cada nodo.

        return VIndEstelas


# Cálculos posteriores (aerodinámicos)
# =============================================================================
    def calculosAerodinamicos(
            self,
            it: int,
    ):
        """
        Método para obtener información aerodinámica luego de una simulación.

        Parameters
        ----------
        it : int
            Índice del instante bajo estudio.
        
        Returns
        -------
        dF : List[np.ndarray] each shape (2, N_s[s])
            Fuerza aerodinámica actuante sobre cada panel de cada sólido.
        Cp : List[np.ndarray] each shape (N_s[s],)
            Cp calculados en los puntos de colocación.
        CxyTotal : np.ndarray shape (2,)
            Coeficientes de fuerzas aerodinámicas del conjunto de sólidos, donde la primera posición se 
            corresponde con Cx y la segunda con Cy.
        CmTotal : float
            Coeficiente de momento aerodinámico Cm del conjunto de sólidos.
        Cxy : np.ndarray shape (2, N_S)
            Coeficientes de fuerzas aerodinámicas de cada sólido donde si se indexa con:
            - `[0, s]`, con `0 <= s <= N_S`, se obtiene el coeficiente Cx del sólido en la posición `s`.
            - `[1, s]`, con `0 <= s <= N_S`, se obtiene con el coeficiente Cy del sólido en la posición `s`.
        Cm : np.ndarray shape (N_S,)
            Coeficientes de momentos aerodinámicos de cada sólido, donde si se indexa con `[s]`, con `0 <= s <= N_S`, se obtiene
            el coeficiente Cm del sólido en la posición `s`.
        """
        it = int(it)
        conjunto = self.conjuntos[it]

        p_pRef = self._p_pRef(it, self.mpConfig['estacionario'])

        Cp = self._Cp(p_pRef)

        dF = self._deltaF(conjunto, p_pRef)

        F, FTotal = self._F(dF)
        CxyTotal = FTotal / self.refConfig['Q'] / self.refConfig['l']
        Cxy = F / self.refConfig['Q'] / self.refConfig['l']

        if len(self.RMRec) > 0:
            RM = self.RMRec[it][:, 0]
            M, MTotal = self._M(conjunto, dF, RM)
            CmTotal = MTotal / self.refConfig['Q'] / self.refConfig['l'] ** 2
            Cm = M / self.refConfig['Q'] / self.refConfig['l'] ** 2

            if self.mpConfig['invertirCm']:
                CmTotal = - CmTotal
                Cm = - Cm
        else:
            CmTotal, Cm = None, None

        return dF, Cp, CxyTotal, CmTotal, Cxy, Cm

    def _p_pRef(
            self,
            it: int,
            estacionario: bool = False,
    ) -> List[np.ndarray]:
        """
        Método que calcula la diferencia de presiones `presión estática - presión de referencia` en cada panel.

        Parameters
        ----------
        it : int
            Índice del instante bajo estudio.
        estacionario : bool
            Indica si se considera la ecuación de Bernoulli estacionaria o no estacionaria.

            Sobreescribe la configuración de la instancia MP2D.MP2D.
        
        Returns
        -------
        p_pRef : List[np.ndarray] each shape (N_s[s],)
            Diferencia de presiones `presión estática - presión de referencia` en cada panel.
        """
        conjunto = self.conjuntos[it]

        # Si el cálculo es no estacionario, se calcula la contribución de la variación temporal del potencial.
        if not estacionario:
            DPhi_Dt= self._DPhi_Dt(it)

        p_pRef = []
        rho = self.refConfig['rho']
        
        for i in range(self.N_S):
            VAbs = conjunto.VInd[i] 
            VPC = conjunto.VPC_XY[i]
            if estacionario or all([not estacionario, it == 0, self.mpConfig['vorticeArranque']]):
                VRel = conjunto.VInd[i] - conjunto.VPC_XY[i]    # Velocidad relativa, en cada punto de colocación.
                p_pRef.append(- rho * (np.einsum('ij,ij->j', VRel, VRel) - self.refConfig['V'] ** 2)/ 2)
            else:
                dPhi_dt = DPhi_Dt[i] - np.einsum('ij,ij->j', VPC, VAbs) # Derivada parcial temporal
                p_pRef.append(- rho * (np.einsum('ij,ij->j', VAbs, VAbs) / 2 + dPhi_dt))
        return p_pRef

    def _deltaF(
            self, 
            conjunto: ConjuntoSolidos2D,
            p_pRef: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Método que calcula la fuerza resultante aerodinámica actuante en cada panel.

        .. note::
            Actualmente no utiliza la instancia de MP2D.

        Parameters
        ----------
        conjunto : _ConjuntoSolidos2D.ConjuntoSolidos2D
            Conjunto bajo estudio.
        p_pRef : List[np.ndarray] each shape (N_s[s],)
            Diferencia de presiones `presión estática - presión de referencia` en cada panel,
            correspondiente al conjunto bajo estudio y calculado previamente.
        
        Returns
        -------
        dF : List[np.ndarray] each shape (2, N_s[s])
            Arreglos con la fuerza resultante aerodinámica actuante en cada panel.
        """

        N_S = len(conjunto.solidos)
        dF = []
        for i in range(N_S):
            dF.append(- p_pRef[i] * conjunto.normales_XY[i] * conjunto.solidos[i].x2_xyp)
        return dF
        
    def _F(
            self,
            dF: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Método que calcula la fuerza resultante aerodinámica de cada sólido y la total.
        
        .. note::
            Actualmente no utiliza la instancia de MP2D.

        Parameters
        ----------
        dF : List[np.ndarray] each shape (2, N_s[s])
            Resultantes aerodinámicas actuantes en cada panel.
        
        Returns
        -------
        F : np.ndarray shape (2, N_S)
            Arreglo con la fuerza resultante aerodinámica en cada sólido.
        FTotal : np.ndarray shape (2,)
            Arreglo con la fuerza resultante aerodinámica total.
        """
        N_S = len(dF)
        F = np.zeros((2, N_S))
        for i in range(self.N_S):
            F[:,i] = np.sum(dF[i], axis=1)
        return F, F.sum(axis=1)

    def _M(
            self,
            conjunto: ConjuntoSolidos2D,
            dF: List[np.ndarray],
            RM: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Método que calcula el momento resultante aerodinámico de cada sólido y el total.
        
        .. note::
            Actualmente no utiliza la instancia de MP2D.

        Parameters
        ----------
        conjunto : _ConjuntoSolidos2D.ConjuntoSolidos2D
            Conjunto bajo estudio.
        dF : List[np.ndarray] each shape (2, N_s[s])
            Resultantes aerodinámicas actuantes en cada panel.
        
        Returns
        -------
        M : np.ndarray shape (N_S)
            Arreglo con el momento resultante aerodinámica en cada sólido.
        MTotal : float
            Momento resultante aerodinámico total.
        """
        M = np.zeros((self.N_S,))

        PC = conjunto.PC_XY

        for i in range(self.N_S):
            R = PC[i] - RM[:,None]

            M[i] = (R[0, :] * dF[i][1, :] - R[1, :] * dF[i][0, :]).sum()
        return M, M.sum()

    def _Cp(
            self,
            p_pRef: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Método que calcula el coeficiente de presiones en cada panel de los sólidos.

        Parameters
        ----------
        p_pRef : List[np.ndarray] each shape (N_s[s],)
            Diferencia de presiones `presión estática - presión de referencia` en cada panel
        
        Returns
        -------
        Cp : List[np.ndarray] each shape (N_s[s],)
            Coeficiente de presiones en cada panel.
        """
        N_S = len(p_pRef)
        return [(p_pRef[i] / self.refConfig['Q']) for i in range(N_S)]

    def _VInd_regular(
            self,
            conjunto: ConjuntoSolidos2D
    ) -> List[np.ndarray]:
        """
        Método que calcula la componente regular de la velocidad inducida en los puntos de
        colocación de un conjunto, debido a la influencia del propio conjunto.

        .. note::
            Actualmente no utiliza la instancia de MP2D.

        Parameters
        ----------
        conjunto : _ConjuntoSolidos2D.ConjuntoSolidos2D
            Conjunto bajo estudio.

        Returns
        -------
        VInd_regular : List[np.ndarray] each shape (2, N_s[s])
            Componente regular de la velocidad inducida en los puntos de
            colocación de un conjunto, debido a la influencia del propio conjunto.
        """

        VInd_regular = []
        # Esta componente se debe únicamente a la autoinfluencia de los paneles sólidos,
        # por lo que solo se itera sobre ellos una vez.
        for i, solidoInfluenciado in enumerate(conjunto.solidos):
            beta = self.panConfig[i]['xPC_x2_xyp']

            # Se quiere construir dmu/dl
            # mu
            mu = solidoInfluenciado.intensidades

            # l
            dl = solidoInfluenciado.x2_xyp              # Longitudes de los paneles
            dl = beta * dl[:-1] + (1 - beta) * dl[1:]   # Longitudes sobre los paneles comprendidas entre los puntos de colocación.
            
            l = np.empty(mu.shape, dtype=float)
            l[0] = 0
            l[1:] = np.cumsum(dl)

            dmu_dl = np.gradient(mu, l, edge_order=1)   # Centrada en el interior y adelantada/atrasada en los bordes.
            
            # Vectores unitarios tangentes a los paneles.
            e_t = solidoInfluenciado.r_xy[:, 1:] - solidoInfluenciado.r_xy[:, :-1]  # (2,N)
            e_t = e_t / np.linalg.norm(e_t, axis=0, keepdims=True)

            VInd_regular.append(- dmu_dl * e_t / 2)   # Vector de velocidades regulares.
        return VInd_regular

    def _phiPC(
            self,
            it: int,
            th12_solidos_viejos: List[List[Tuple[np.ndarray]]] = None,
            th12_estelas_viejos: List[List[Tuple[np.ndarray]]] = None,
            th_vortices_viejos: List[List[np.ndarray]] = None,
    ) -> Tuple[List[np.ndarray], List[List[Tuple[np.ndarray]]], List[List[Tuple[np.ndarray]]], List[List[np.ndarray]]]:
        """
        Método que calcula el potencial inducido en los puntos de colocación en un instante dado.
        
        Además, como se utilizan las expresiones en coordenadas polares, de proveerse ángulos viejos (de referencia), 
        se resuelven los cortes que puedan interferir, modificando el nuevo ángulo para que la diferencia con el de 
        referencia no exceda 2pi en módulo.

        Notes
        -----
        - Convención de índices:
            - `i`: índice del sólido influenciado.
            - `j`: índice del sólido que influencia.
        - Los potenciales aquí calculados no son consistentes entre ellos, pues a lo sumo lo son con los
        ángulos de referencia pasado. Esto conlleva a utilizarlos para hacer una visualización de ello
        presentaría defectos debido a los cortes.

        Parameters
        ----------
        it : int
            Índice del instante bajo estudio.
        th12_solidos_viejos : List[List[Optional[Tuple[np.ndarray]]]] each shape (N_s[i], N_s[j]), optional
            Ángulos viejos (de refeferencia) para resolver los cortes que puedan existir. Relativo a los sólidos.
        th12_estelas_viejos : List[List[Optional[Tuple[np.ndarray]]]] each shape (N_s[i], N_s[j]), optional
            Ángulos viejos (de refeferencia) para resolver los cortes que puedan existir. Relativo a los dobletes de la estela.
        th_vortices_viejos : List[List[Optional[Tuple[np.ndarray]]]] each shape (N_s[i], N_s[j]), optional
            Ángulos viejos (de refeferencia) para resolver los cortes que puedan existir. Relativo a los vórtices de la estela.

        Returns
        -------
        phi: List[np.ndarray] each shape (N_s[i],)
            Potenciales inducidos en los puntos de colocación de cada sólido.
        th12_solidos_nuevos : List[List[Tuple[np.ndarray]]] each shape (N_s[i], N_s[j]), optional
            Ángulos nuevos. Relativo a los sólidos.
        th12_estelas_nuevos : List[List[Tuple[np.ndarray]]] each shape (N_s[i], N_s[j]), optional
            Ángulos nuevos. Relativo a los dobletes de la estela.
        th_vortices_nuevos : List[List[Tuple[np.ndarray]]] each shape (N_s[i], N_s[j]), optional
            Ángulos nuevos. Relativo a los vórtices de la estela.
        """
        it = int(it)
        conjunto = self.conjuntos[it]

        phi = []

        instanteInicial = False
        if not conjunto.estelas:
            instanteInicial = True

        th12_solidos_nuevos = []
        th12_estelas_nuevos = []
        th_vortices_nuevos = []

        # Bucle sobre los sólidos influenciados.
        for i in range(self.N_S):
            R_XY = conjunto.PC_XY[i]    # Puntos de evaluación.
            M = R_XY.shape[1]
            phi_i = np.zeros((M,))
            th12_solidos_nuevos_i = []
            th12_estelas_nuevos_i = []
            th_vortices_nuevos_i = []

            # Bucle sobre los sólidos que influencian.
            for j in range(self.N_S):
                # Influencia por sólido
                # ---------------------
                th1, th2 = conjunto.solidos[j].th12(R_XY)    # (M, N). Se obtienen los ángulos desde los nodos hasta los puntos de evaluación.

                # Se desenvuelven los ángulos utilizando los de referencia, si es que se pasaron.
                if th12_solidos_viejos:
                    th1 = self._desenvolverTheta(th1,th12_solidos_viejos[i][j][0])
                    th2 = self._desenvolverTheta(th2, th12_solidos_viejos[i][j][1])

                th12_solidos_nuevos_i.append((th1, th2))

                # Se calcula el potencial considerando a todos los puntos de evaluación como si fueran externos a los paneles.
                # Esto no es cierto en el caso i = j, pero esto se resuelve a continuación.
                intensidades = conjunto.solidos[j].intensidades
                phi_j = self._phi_ext_dobletes(
                    th1,
                    th2,
                    intensidades
                )

                # En el caso i = j, se reemplaza la diagonal por la expresión para las autoinfluencias.
                if i == j:
                    np.fill_diagonal(
                        phi_j, 
                        self._phi_panel_dobletes(
                            intensidades,
                        ))

                phi_i += np.sum(phi_j, axis=0)  # Se suma la contribución sobre cada punto de colocación.

                # Influencia por estela (vórtice)
                # -------------------------------

                th = conjunto.vortices[j].th(R_XY) # (M, N). Se obtienen los ángulos desde el vórtice de arranque hasta los puntos de evaluación.
                
                # Se desenvuelven los ángulos utilizando los de referencia, si es que se pasaron.
                if th_vortices_viejos:
                    th = self._desenvolverTheta(th, th_vortices_viejos[i][j])

                # Se calcula el potencial. Los puntos de colocación siempre son externos a los vórtices.   
                phi_j = self._phi_ext_vortices(
                    th,
                    conjunto.vortices[j].intensidades
                )

                th_vortices_nuevos_i.append(th)

                phi_i += phi_j[0,:] # Se suma la contribución sobre cada punto de colocación.
                # Se usa la primera fila, ya que es la única pues solo existe un vórtice por estela, el de arranque.
                
                # Influencia por estela (dobletes)
                # --------------------------------
                if not instanteInicial:
                    th1, th2 = conjunto.estelas[j].th12(R_XY)   # (M, N). Se obtienen los ángulos desde los nodos hasta los puntos de evaluación.
                    
                    # Se desenvuelven los ángulos utilizando los de referencia, si es que se pasaron y si contienen
                    # elementos válidos.
                    if (th12_estelas_viejos
                        and len(th12_estelas_viejos) > i    
                        and len(th12_estelas_viejos[i]) > j):

                        th1 = np.concatenate([th1[0:1,:], self._desenvolverTheta(th1[1:,:],th12_estelas_viejos[i][j][0])], axis=0)
                        th2 = np.concatenate([th2[0:1,:], self._desenvolverTheta(th2[1:,:], th12_estelas_viejos[i][j][1])], axis=0)
                    
                    th12_estelas_nuevos_i.append((th1, th2))
            
                    # Se calcula el potencial. Los puntos de colocación siempre son externos a los dobletes.   
                    intensidades = conjunto.estelas[j].intensidades
                    phi_j = self._phi_ext_dobletes(
                        th1,
                        th2,
                        intensidades,
                    )

                    phi_i += np.sum(phi_j, axis=0)  # Se suma la contribución sobre cada punto de colocación.

            phi.append(phi_i)
            th12_solidos_nuevos.append(th12_solidos_nuevos_i)
            th12_estelas_nuevos.append(th12_estelas_nuevos_i)
            th_vortices_nuevos.append(th_vortices_nuevos_i)

        return phi, th12_solidos_nuevos, th12_estelas_nuevos, th_vortices_nuevos

    def _DPhi_Dt(
            self,
            it: int,
            DtInicial: float = 1e-10
    ) -> List[np.ndarray]:
        """
        Método para calcular la "derivada material" del potencial inducido en los puntos de colocación. Esta derivada
        no es la material siguiendo una partícula de fluido, sino que sigue al punto de colocación.

        Parameters
        ----------
        it : int
            Índice del instante bajo estudio.
        DtInicial : float, optional
            Valor utilizado para representar una diferencia de tiempos inicial instantánea. 
            
            La diferencia de tiempos se utiliza para la derivación respecto del tiempo, dividiendo por esta cantidad
            a la diferencia de potenciales.
            
            En el instante inicial, esta diferencia de tiempos se puede resolver de dos formas, dependiendo de la
            configuración utilizada:
            - Si `MP2D.mpConfig['vorticeArranque`]`: la diferencia de tiempos está indeterminada y se usa `np.nan`.
            - De lo contrario: se impone el valor `DtInicial`, que por defecto es `1e-10`.
        
        Returns
        ------
        DPhi_Dt : List[np.ndarray] each shape (N_s[s],)
            Derivada material del potencial inducido en los puntos de colocación.        
        """
        it = int(it)
        conjunto_nuevo = self.conjuntos[it]

        DPhi_Dt = []
                
        instanteInicial = False
        if not conjunto_nuevo.estelas:
            instanteInicial = True
        
        # Instante inicial
        # ----------------
        if instanteInicial:
            # Primer instante estacionario, con lo cual la diferencia de tiempos está indeterminada.
            if self.mpConfig['vorticeArranque']:
                Dt = np.nan
            # Un infinitesimal antes, el sólido estaba en reposo. La diferencia de tiempos debe ser pequeñísima.
            else:
                Dt = DtInicial
            
            phi_nuevo, _, _, _= self._phiPC(it)  # Se calcula el nuevo potencial.    

            # Diferencias finitas.
            for i in range(self.N_S):
                DPhi_Dt.append(phi_nuevo[i] / Dt)

        # Instantes posteriores
        # --------------------- 
        else:
            Dt = self.t[it] - self.t[it-1]  # Se calcula el paso del tiempo.

            # Se calcula el potencial en el instante anterior.
            phi_viejo, th12_solidos_viejos, th12_estelas_viejos, th_vortices_viejos = self._phiPC(it-1)
            
            # Se calcula el potencial en el instante actual, desenvolviendo los ángulos con tomando de referencia
            # los viejos, para evitar diferencias por corte de ángulos.
            phi_nuevo, _, _, _= self._phiPC(it, th12_solidos_viejos, th12_estelas_viejos, th_vortices_viejos)   

            # Diferencias finitas.
            for i in range(self.N_S):
                DPhi_Dt.append((phi_nuevo[i] - phi_viejo[i]) / Dt)
        return DPhi_Dt   


# Calculo de potenciales y diferencias de potenciales
# =============================================================================

    def _desenvolverTheta(
            self,
            th_nuevo: np.ndarray,
            th_viejo: np.ndarray,
    ) -> np.ndarray:
        '''
        Función que devuelve un ángulo nuevo en radianes evitando cortes en relación a un ángulo viejo (de referencia).
        
        Parameters
        ----------
        th_nuevo : np.ndarray
            Arreglo de ángulos a corregir por cortes.
        th_viejo : np.ndarray
            Arreglo de ángulos viejos de referencia para corregir los nuevos. Debe tener la misma forma que `th_nuevo`.

        Returns
        -------
        th_nuevo_mod : np.ndarray
            Arreglo de ángulos corregido.
        '''
        dif = th_nuevo - th_viejo
        dif = (dif + np.pi) % (2 * np.pi) - np.pi
        return th_viejo + dif

    def _phi_ext_dobletes(
            self,
            th1: np.ndarray,      # (N, M)
            th2: np.ndarray,      # (N, M)
            intensidades: np.ndarray,    # (N,)
    ) -> np.ndarray:    # (N, M)
        '''
        Potencial inducido por un conjunto de N dobletes constantes contiguos entre sí, en un conjunto de M puntos exteriores a estos.
        
        Parameters
        ----------
        th1 : np.ndarray shape (N, M)
            Arreglos de ángulos th1 del potencial inducido en los puntos a evaluar.
        th2 : np.ndarray shape (N, M)
            Arreglos de ángulos th2 del potencial inducido en los puntos a evaluar.
        
        Returns
        -------
        phi : np.ndarray (N, M)
            Potencial inducido en cada punto y por cada panel.
        '''
        dth =  (th2 - th1 + np.pi) % (2*np.pi) - np.pi
        return - 0.5 * intensidades[:, None] / np.pi * dth

    def _phi_panel_dobletes(
            self,
            intensidades: np.ndarray,    # (N,)
            afuera: bool = True
    ) -> np.ndarray:    # (N,)    
        '''
        Potencial inducido en los puntos de colocación de una distribución de dobletes constantes.

        Para Neumann se considera el punto de colocación desde el exterior del perfil.

        Parameters
        ----------
        intensidades : np.ndarray shape (N,)
            Arreglo de intensidades de los paneles.
        afuera : bool
            Implica que en los casos de puntos de evaluación
            en las singularidades, se considere el límite por afuera del sólido
            si `True`, y por dentro en el caso contrario.
            Por defecto, `True`.
        '''
        if afuera:
            afuera = 1
        else:
            afuera = -1
        return - afuera * intensidades / 2

    def _phi_ext_vortices(
            self,
            th: np.ndarray,     # (N, M)
            intensidades: np.ndarray    # (N,)
    ) -> np.ndarray:    # (N, M)
        '''
        Potencial inducido por un conjunto de N vórtices puntuales, en un conjunto de M puntos exteriores a estos.
        
        Parameters
        ----------
        th : np.ndarray shape (N, M)
            Arreglos de ángulos th del potencial inducido en los puntos a evaluar.
        intensidades : np.ndarray shape (N,)
            Arreglo de intensidades de los vórtices.
        
        Returns
        -------
        phi : np.ndarray (N, M)
            Potencial inducido en cada punto y por cada vórtice.
        '''
        return 0.5 * intensidades[:, None] / np.pi * th

# =============================================================================
# Graficación
# =============================================================================
    def plotPaneles(
            self,
            it: int,
            ax: plt.Axes | None = None,
            mostrarRM: bool = False,
            xlabel: str = 'x, [m]',
            ylabel: str = 'y, [m]',
            markersize: float = 3.0,
            estelaDesconectada: bool = False,
    ) -> plt.Axes:
        """
        Método para graficar el conjunto en un instante dado.

        Parameters
        ----------
        it : int
            Índice del instante a graficar.
        ax : matplotlib.pyplot.Axes, optional
            Objeto de ejes en el cual graficar. Por defecto, se crea uno.
        mostrarRM : bool, optional.
            Si `True` y existe, se muestra el punto de toma de momentol. Por defecto, `False`.
        xlabel: str, optional.
            Rótulo del eje x.
        ylabel: str, optional.
            Rótulo del eje y. 
        estelaDesconectada: bool, optional
            Si `True`, solo se grafican los nodos de la estela. Si `False`, los nodos se conecta. Por defecto, `False`.
        Returns
        -------
        ax : matplotlib.pyplot.Axes
            Objeto de ejes en el cual se graficó.
        """
        it = int(it)
        if ax is None:
            _, ax = plt.subplots(1,1)

        conjunto = self.conjuntos[it]

        estelaLinestyle = '' if estelaDesconectada else '-'
        for s in range(self.N_S):
            R_XY = conjunto.solidos[s].r_xy
            ax.plot(R_XY[0,:], R_XY[1,:], 'k-o', markersize=markersize)
            if it == 0 or self.mpConfig['estacionario']:
                R_XY = conjunto.vortices[s].r0_xy
                ax.plot(R_XY[0,:], R_XY[1,:], color='gray', linestyle='', marker='o', markersize=markersize)
            if not self.mpConfig['estacionario'] and conjunto.estelas:
                R_XY = conjunto.estelas[s].r_xy
                ax.plot(R_XY[0,:], R_XY[1,:], color='gray', linestyle=estelaLinestyle, marker='o', markersize=markersize)
        ax.axis('equal')
        ax.grid(True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Leyenda personalizada
        legend_elements = [
            Line2D([0], [0], color='k', marker='o', markersize=3, label='Sólidos'),
            Line2D([0], [0], color='gray', marker='o', linestyle=estelaLinestyle, markersize=3, label='Estelas')
        ]

        if mostrarRM and len(self.RMRec) > 0:
            RM = self.RMRec[it]
            ax.plot(RM[0, 0], RM[1, 0], 'gx', markersize=10)
            legend_elements.append(Line2D([0], [0], color='g', marker='x', linestyle='', markersize=10, label='RM'))
        
        ax.legend(handles=legend_elements, loc='best')
        return ax
    
    def plotCp(
            self,
            it: int,
            ax: plt.Axes | None = None,
            xlabel: str = 'x, [m]',
    ) -> plt.Axes:
        """
        Método para graficar la distribución de Cp del conjunto en un instante dado.

        .. note::
            Actualmente, se utiliza al eje x como Cp = 0 y las coordenadas X de cada
            punto de colocación como abcisa de la distribución. Esto puede ocasionar
            visualizaciones del Cp incómodas para ángulos de ataque relativamente grandes.

        Parameters
        ----------
        it : int
            Índice del instante a graficar.
        ax : matplotlib.pyplot.Axes, optional
            Objeto de ejes en el cual graficar. Por defecto, se crea uno.
        xlabel: str, optional.
            Rótulo del eje x.
        Returns
        -------
        ax : matplotlib.pyplot.Axes
            Objeto de ejes en el cual se graficó.
        """
        it = int(it)
        if ax is None:
            _, ax = plt.subplots(1,1)

        ax.axhline(y=0., linewidth=2, color='k')

        conjunto = self.conjuntos[it]
        Cp = self.CpRec[it]
        x = []
        for i in range(self.N_S):
            x.append(conjunto.PC_XY[i][0,:])
            ax.plot(x[i], Cp[i], label=self.mpConfig['nombres'][i])

        x = np.concatenate(x, axis=0)
        ax.axis('auto')
        # Se invierte el eje y por convención.
        ymin, ymax = ax.get_ylim()
        if ymin < ymax:
            ax.set_ylim(ymax, ymin)
        ax.legend()
        ax.grid(True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Cp, [ ]')
        return ax      
      
    def plotV(
            self,
            it: int,
            ax: plt.Axes | None = None,
            escala: float | None = 1,
            VRel: bool | np.ndarray = False,
            xlabel: str = 'x, [m]',
            ylabel: str = 'y, [m]',
            quiverConfig: Tuple | None = None,
            estelaDesconectada: bool = False,
    ) -> plt.Axes:
        """
        Método para graficar las velocidades en los puntos de colocación del conjunto en un instante dado.

        Parameters
        ----------
        it : int
            Índice del instante a graficar.
        ax : matplotlib.pyplot.Axes, optional
            Objeto de ejes en el cual graficar. Por defecto, se crea uno.
        escala : float
            Valor con el que se escala la longitud de las flechas de las velocidades.
        VRel: bool | np.ndarray shape (2, 1)
            Si `True`, en modo estacionario resta la velocidad del flujo no perturbado a partir de
            alguna de las velocidades cinemáticas. 

            .. note::
                En modo no estacionario, la validez de lo graficado depende si el problema pudiera ser considerado
                estacionario. No se verifica.
            
            Si es un np.ndarray, debe ser un vector (2, 1) que se sustraerá a todo el campo.
        xlabel: str, optional.
            Rótulo del eje x.
        ylabel: str, optional.
            Rótulo del eje y.
        quiverConfig: tuple, optional.
            Tupla de configuracion del quiver con los elementos 'scale_units', 'width', 'cmap', 'norm' y 'scale'.
        estelaDesconectada: bool, optional
            Si `True`, solo se grafican los nodos de la estela. Si `False`, los nodos se conecta. Por defecto, `False`.
        Returns
        -------
        ax : matplotlib.pyplot.Axes
            Objeto de ejes en el cual se graficó.
        """
        it = int(it)
        ax = self.plotPaneles(it, ax, xlabel=xlabel, ylabel=ylabel, estelaDesconectada=estelaDesconectada)

        conjunto = self.conjuntos[it]
        PC = conjunto.PC_XY

        if quiverConfig is not None:
            scale_units, width, cmap, norm, scale = quiverConfig
        
        for i in range(self.N_S):
            x = PC[i][0, :]
            y = PC[i][1, :]

            VInd = conjunto.VInd[i]

            if isinstance(VRel, np.ndarray) or isinstance(VRel, bool):
                if isinstance(VRel, bool):
                    if VRel:
                        VCin = conjunto.VPC_XY[i]
                    else:
                        VCin = 0.
                else:
                    VCin = VRel
                VInd = VInd - VCin

            U = escala * VInd[0, :]
            V = escala * VInd[1, :]

            if quiverConfig is None:
                ax.quiver(x, y, U, V, color='green', angles='xy', scale_units='xy', scale=1, width=0.001)
            else:
                C = np.sqrt((VInd[0, :] ** 2) + (VInd[1, :] ** 2))
                ax.quiver(x, y, U, V, C, scale_units=scale_units, width=width, cmap=cmap, norm=norm, scale=scale, angles='xy', zorder=10)
        return ax

    def plotCpVect(
            self, 
            it: int, 
            ax: plt.Axes | None = None,
            escala: float | None = 1,
            xlabel: str = 'x, [m]',
            ylabel: str = 'y, [m]',
            estelaDesconectada: bool = False,
    ) -> plt.Axes:
        """
        Método para graficar la distribución de Cp vectorial del conjunto en un instante dado.
        - Naranja: succión.
        - Azul: sobrepresión.

        Parameters
        ----------
        it : int
            Índice del instante a graficar.
        ax : matplotlib.pyplot.Axes, optional
            Objeto de ejes en el cual graficar. Por defecto, se crea uno.
        escala : float
            Valor con el que se escala la longitud de las flechas del Cp vectorial.
        xlabel: str, optional.
            Rótulo del eje x.
        ylabel: str, optional.
            Rótulo del eje y.
        estelaDesconectada: bool, optional
            Si `True`, solo se grafican los nodos de la estela. Si `False`, los nodos se conecta. Por defecto, `False`. 
        Returns
        -------
        ax : matplotlib.pyplot.Axes
            Objeto de ejes en el cual se graficó.
        """
        it = int(it)
        ax = self.plotPaneles(it, ax, xlabel=xlabel, ylabel=ylabel, estelaDesconectada=estelaDesconectada)

        PC = self.conjuntos[it].PC_XY
        Cp = self.CpRec[it]
        normales = self.conjuntos[it].normales_XY

        for i in range(self.N_S):
            x = PC[i][0, :]
            y = PC[i][1, :]

            U = escala * np.abs(Cp[i]) * normales[i][0, :]
            V = escala * np.abs(Cp[i]) * normales[i][1, :]

            succion = (Cp[i] < 0)
            sobrepresion = ~succion

            if np.any(succion):
                ax.quiver(x[succion], y[succion], U[succion], V[succion], color='orange', scale=1, scale_units='xy',width=0.001, zorder=3)

            if np.any(sobrepresion):
                ax.quiver(
                    x[sobrepresion] + U[sobrepresion], 
                    y[sobrepresion] + V[sobrepresion], 
                    - U[sobrepresion], 
                    - V[sobrepresion], 
                    color='blue', scale=1, scale_units='xy',width=0.001, zorder=3
                )
        return ax

    def plotCAero(
            self,
            ax: Sequence[plt.Axes] | None = None,
            CxLims: Sequence[float] | None = None,
            CyLims: Sequence[float] | None = None,
            CmLims: Sequence[float] | None = None,
            total: bool = False
    ) -> plt.Axes:
        """
        Método para graficar los coeficientes aerodinámicos Cx, Cy y Cm vs. el tiempo.

        Parameters
        ----------
        ax : Sequence[matplotlib.pyplot.Axes], optional
            Secuencia de ejes en los cuales graficar. Por defecto, se crean.
        CxLims: Sequence[float] length 2
            Límites del eje y en el gráfico de Cx. Por defecto, se grafica según el comportamiento
            normal de matplotlib.pyplot.
        CyLims: Sequence[float] length 2
            Límites del eje y en el gráfico de Cy. Por defecto, se grafica según el comportamiento
            normal de matplotlib.pyplot.
        CmLims: Sequence[float] length 2
            Límites del eje y en el gráfico de Cm. Por defecto, se grafica según el comportamiento
            normal de matplotlib.pyplot.
        total : bool
            Si `True`, grafica los coeficientes netos. Si no, por cada sólido.

        Returns
        -------
        ax : Sequence[matplotlib.pyplot.Axes], optional
            Secuencia de ejes en los cuales se graficó.
        """
        if ax is None:
            if self.RM is not None:
                _, ax = plt.subplots(1, 3)
            else:
                _, ax = plt.subplots(1, 2)
        else:
            if self.RM is not None:
                ax = ax[:3]
            else:
                ax = ax[:2]

        if total:
            Cx = [Cxy[0] for Cxy in self.CxyTotalRec]
            Cy = [Cxy[1] for Cxy in self.CxyTotalRec]
            ax[0].plot(self.t, Cx, label='Total')
            ax[1].plot(self.t, Cy, label='Total')

        else:
            for i in range(self.N_S):
                Cx = [cxy[0, i] for cxy in self.CxyRec]
                Cy = [cxy[1, i] for cxy in self.CxyRec]
                ax[0].plot(self.t, Cx, label=self.mpConfig['nombres'][i])
                ax[1].plot(self.t, Cy, label=self.mpConfig['nombres'][i])

        if CxLims:
            ax[0].set_ylim(CxLims)
        if CyLims:
            ax[1].set_ylim(CyLims)

        if self.RM is not None: 
            if total:
                Cm = self.CmTotalRec
                ax[2].plot(self.t, Cm, label='Total')
            else:
                for i in range(self.N_S):
                    Cm = [Cm[i] for Cm in self.CmRec]
                    ax[2].plot(self.t, Cm, label=self.mpConfig['nombres'][i])

            ax[2].set_title('Cm')
            if CmLims:
                ax[2].set_ylim(CmLims)
        
        xlabel = self.mpConfig['rotulo_t']
        ylabel = ['Cx, [ ]', 'Cy, [ ]', 'Cm, [ ]']

        for i, ax_i in enumerate(ax):
            ax_i.set_title(ylabel[i][0:2])
            ax_i.set_ylabel(ylabel[i])
            ax_i.set_xlabel(xlabel)
            ax_i.grid(True)
            ax_i.legend()
        return ax
    
    def plotCampo_V(
            self,
            it: int,
            x: np.ndarray,
            y: np.ndarray,
            ax: plt.Axes | None = None,
            escala: float = 1.,
            VRel: bool | np.ndarray = False,
            mostrarPC: bool = True,
            radius: float | None = 0,
            xlabel: str = 'x, [m]',
            ylabel: str = 'y, [m]',
            Vlabel: str = '|V|, [m/s]',
            VMin: float | None = None,
            VMax: float | None = None,
            estelaDesconectada: bool = False,
    ) -> plt.Axes:
        """
        Método para graficar el campo vectorial de velocidades.

        .. note::

            Se grafica el campo de velocidades en el sistema fijo, con lo cual en la
            visualización no queda clara la condición de impenetrabilidad.

            Una en la que sí se aprecie esto no es posible en el caso general, ya que 
            cada sólido puede estar animádo con cinemáticas complejas y diferentes
            entre sí.

        Parameters
        ----------
        it : int
            Índice del instante a graficar.
        x : np.ndarray shape (Nx,)
            Valores de x con los cuales se construirá la grilla de evaluación.
        y : np.ndarray shape (Ny,)
            Valores de y con los cuales se construirá la grilla de evaluación.
        ax : matplotlib.pyplot.Axes, optional
            Objeto de ejes en el cual graficar. Por defecto, se crea uno.
        escala : float
            Valor con el que se escala la longitud de las flechas de las velocidades.
        VRel: bool | np.ndarray shape (2, 1)
            Si `True`, en modo estacionario resta la velocidad del flujo no perturbado a partir de
            alguna de las velocidades cinemáticas. 

            .. note::
                En modo no estacionario, la validez de lo graficado depende si el problema pudiera ser considerado
                estacionario. No se verifica.
            
            Si es un np.ndarray, debe ser un vector (2, 1) que se sustraerá a todo el campo.
        mostrarPC : bool
            Si `mostrarPC`: se muestra la velocidad en los puntos de colocación con la misma
            escala. Por defecto, `True`.
        radius : float
            Parametro como el del mismo nombre de `matplotlib.path.Path.contains_points`.

            Sirve para ajustar la categorización de los puntos, como internos o externos
            a los sólidos, para ajustar la visualización y evitar mostrar los defectos de
            la singularidad concentrada en los nodos, inherente a una discretización con 
            dobletes constantes.

            - Si es positivo, un punto de evaluación cercano a la frontera del lado externo, se
            puede considerar dentro.
            - Si es negativo, un punto de evaluación cercano a la frontera del lado interno, se
            puede considerar fuera.
        xlabel: str, optional.
            Rótulo del eje x.
        ylabel: str, optional.
            Rótulo del eje y. 
        VMin: float, optional
            Velocidad (módulo) mínima en la escala de colores. Por defecto, se utiliza la mínima calculada en el campo.
        VMax: float, optional
            Velocidad (módulo) máxima en la escala de colores. Por defecto, se utiliza la máxima calculada en el campo.
        estelaDesconectada: bool, optional
            Si `True`, solo se grafican los nodos de la estela. Si `False`, los nodos se conecta. Por defecto, `False`. 
        Returns
        -------
        ax : matplotlib.pyplot.Axes
            Objeto de ejes en el cual se graficó.
        """
        it = int(it)

        R_XY, shape = self.campo(it, x, y, radius=radius)
        
        Vx, Vy = self.V_campo(it, R_XY, shape, VRel)

        VNorm = np.sqrt(Vx**2 + Vy**2)

        m = ~ np.isnan(Vx.ravel())
        X = R_XY[0, m]
        Y = R_XY[1, m]

        U = Vx.ravel()[m] * escala
        V = Vy.ravel()[m] * escala
        
        C = VNorm.ravel()[m]

        ax = self.plotPaneles(it, ax, xlabel=xlabel, ylabel=ylabel, estelaDesconectada=estelaDesconectada)
        
        if VMin is None:
            vmin = C.min()
        else:
            vmin = VMin
        
        if VMax is None:
            vmax = C.max()
        else:
            vmax = VMax

        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        scale_units = 'xy'
        width = 0.001
        cmap = 'jet'
        norm = norm
        scale = 1.0
        
        q = ax.quiver(
            X, Y, U, V, C,
            scale_units=scale_units,
            width=width,
            cmap=cmap,
            norm=norm,
            scale=scale,
            angles='xy',
        )

        if mostrarPC:
            quiverConfig = (scale_units, width, cmap, norm, scale)
            self.plotV(it, ax, escala, VRel, xlabel, ylabel, quiverConfig, estelaDesconectada=estelaDesconectada)

        cbar = plt.colorbar(q, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
        cbar.set_label(Vlabel)

        ax.axis('equal')
        return ax

    def plotCampo_Cp(
            self,
            it: int,
            x: np.ndarray,
            y: np.ndarray,
            ax: plt.Axes | None = None,
            niveles: int | None = 200,
            mostrarPC: bool = True,
            radius: float | None = 0,
            xlabel: str = 'x, [m]',
            ylabel: str = 'y, [m]',
            CpMin: float | None = None,
            estelaDesconectada: bool = False,
    ) -> plt.Axes:
        """
        Método para graficar el campo escalar de coeficientes de presión.

        Parameters
        ----------
        it : int
            Índice del instante a graficar.
        x : np.ndarray shape (Nx,)
            Valores de x con los cuales se construirá la grilla de evaluación.
        y : np.ndarray shape (Ny,)
            Valores de y con los cuales se construirá la grilla de evaluación.
        ax : matplotlib.pyplot.Axes, optional
            Objeto de ejes en el cual graficar. Por defecto, se crea uno.
        niveles : int
            Número de niveles de colores que se utilizan en el gráfico.
        mostrarPC : bool
            Si `mostrarPC`: se muestra el Cp en los puntos de colocación con la misma
            escala de colores; si no, no. Por defecto, `True`.
        radius : float
            Parametro como el del mismo nombre de `matplotlib.path.Path.contains_points`.

            Sirve para ajustar la categorización de los puntos, como internos o externos
            a los sólidos, para ajustar la visualización y evitar mostrar los defectos de
            la singularidad concentrada en los nodos, inherente a una discretización con 
            dobletes constantes.

            - Si es positivo, un punto de evaluación cercano a la frontera del lado externo, se
            puede considerar dentro.
            - Si es negativo, un punto de evaluación cercano a la frontera del lado interno, se
            puede considerar fuera.
        xlabel: str, optional.
            Rótulo del eje x.
        ylabel: str, optional.
            Rótulo del eje y.

        CpMin: float, optional
            Valor mínimo de Cp en la escala de colores. Por defecto se usa el mínimo valor desarrollado sobre las superficies de los sólidos.
        estelaDesconectada: bool, optional
            Si `True`, solo se grafican los nodos de la estela. Si `False`, los nodos se conecta. Por defecto, `False`. 
        Returns
        -------
        ax : matplotlib.pyplot.Axes
            Objeto de ejes en el cual se graficó.
        """

        it = int(it)

        R_XY, shape = self.campo(it, x, y, radius=radius)
        
        Cp_sup = self.CpRec[it] # Cp en la superficie.
        Cp_sup_conc = np.concatenate(Cp_sup)

        CpMin_sup, CpMax_sup = np.min(Cp_sup_conc), np.max(Cp_sup_conc) # Mínimo y máximo Cp desarrollados en la superficie.

        if CpMin is None:
            CpMin = CpMin_sup
        
        if CpMax_sup > 1.0:
            warnings.warn(f'El valor máximo de Cp en la superficie ({CpMax_sup:.3f}) supera la unidad. La escala de color mantendrá el límite superior en la unidad.', RuntimeWarning)

        CpMax = 1.0

        CpLims = (CpMin, CpMax) # Límites del Cp. Fuera de la superficie, el Cp decae.

        Cp = self.Cp_campo(it, R_XY, shape, CpLims) # Se calcula el campo de Cp.

        norm = Normalize(vmin=CpMin, vmax=CpMax)    # Se normalizan los valores, con lo límites del Cp definidos.
        
        levels = np.linspace(CpMin, CpMax, niveles) 

        ax = self.plotPaneles(it, ax, xlabel=xlabel, ylabel=ylabel, markersize=0.5, estelaDesconectada=estelaDesconectada)

        # Se grafica el campo externo.
        cf = ax.contourf(
            R_XY[0,:].reshape(shape), 
            R_XY[1,:].reshape(shape), 
            Cp,
            levels=levels, 
            cmap='jet_r',
            norm=norm,
            zorder=0,
        )

        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label('Cp, [ ]')

        # Se grafica el campo en los puntos de colocación.
        if mostrarPC:
            PC = self.conjuntos[it].PC_XY
            for i in range(self.N_S):
                ax.scatter(
                PC[i][0,:], PC[i][1,:], 
                c=Cp_sup[i], 
                cmap=cf.cmap,
                norm=norm,
                edgecolor='k', 
                s=10, 
                linewidth=0,
                zorder=1
            )
        return ax
            
    def V_campo(
            self,
            it: int,
            R_XY: np.ndarray,
            shape: Tuple[int, int],
            VRel: bool = False,
    ) -> Tuple[np.ndarray]:
        """
        Método para calcular la velocidad inducida en una grilla de evaluación.

        Notes
        -----
        - Se asume que los puntos de la grilla no contienen a puntos sobre los paneles.
        - Está pensado para utilizarse con el método MP2D.campo(...) que devuelve `R_XY`
        y `shape`.

        Parameters
        ----------
        it : int
            Índice del instante bajo estudio.
        R_XY : np.ndarray shape (2, Nx * Ny)
            Coordenadas de evaluación donde coordenadas internas a los sólidos del conjunto 
            son reemplazadas por `np.nan`.
        shape : Tuple[int, int]
            Forma de la grilla `(Nx, Ny)` construida con `x` e `y`.
        VRel: bool | np.ndarray shape (2, 1)
            Si `True`, en modo estacionario resta la velocidad del flujo no perturbado a partir de
            alguna de las velocidades cinemáticas. 

            .. note::
                En modo no estacionario, la validez de lo graficado depende si el problema pudiera ser considerado
                estacionario. No se verifica.
            
            Si es un np.ndarray, debe ser un vector (2, 1) que se sustraerá a todo el campo.
        Returns
        -------
        Vx : np.ndarray shape (Nx, Ny)
            Componentes x de las velocidades inducidas en los puntos de evaluación.
        Vy : np.ndarray shape (Nx, Ny)
            Componentes y de las velocidades inducidas en los puntos de evaluación.
        """
        it = int(it)

        conjunto = self.conjuntos[it]

        n, m = shape
        M = R_XY.shape[1]

        VInd = np.zeros((2, M))

        instanteInicial = False
        if not conjunto.estelas:
            instanteInicial = True
        
        for i, solidoQueInfluencia in enumerate(conjunto.solidos):
            # Influencia por paneles sólidos
            VInd += np.sum(solidoQueInfluencia.VInd_xy(R_XY, ubicaciones=None, afuera=True, intensidadesUnitarias=False), axis=1)
            # Influencia por vórtices de eestla
            vorticeQueInfluencia = conjunto.vortices[i]
            VInd += np.sum(vorticeQueInfluencia.VInd_xy(R_XY, ubicaciones=None, intensidadesUnitarias=False), axis=1)

            # Influencia por dobletes de estela
            if not instanteInicial:
                estelaQueInfluencia = conjunto.estelas[i]
                VInd += np.sum(estelaQueInfluencia.VInd_xy(R_XY, ubicaciones=None, afuera=True, intensidadesUnitarias=False), axis=1)
        if isinstance(VRel, np.ndarray) or isinstance(VRel, bool):
            if isinstance(VRel, bool):
                if VRel:
                    VCin = self.conjuntos[it].TM.v2V_1TM(R_XY, np.zeros_like(R_XY), 0)
                else:
                    VCin = 0.
            else:
                VCin = VRel
            #VInf = self.conjuntos[it].VPC_XY[0][:, 0]    # Velocidad relativa al fluido en reposo de los sólidos.
            VInd = VInd - VCin

        Vx, Vy = VInd.reshape((2, n, m))
        return Vx, Vy       
    
    def Cp_campo(
            self,
            it: int,
            R_XY: np.ndarray,
            shape: Tuple[int, int],
            CpLims: Sequence | None = [-5, 1],
    ) -> np.ndarray:
        """
        Método para calcular el Cp en una grilla de evaluación.

        Notes
        -----
        - Se asume que los puntos de la grilla no contienen a puntos sobre los paneles.
        - Está pensado para utilizarse con el método MP2D.campo(...) que devuelve `R_XY`
        y `shape`.

        Parameters
        ----------
        it : int
            Índice del instante bajo estudio.
        R_XY : np.ndarray shape (2, Nx * Ny)
            Coordenadas de evaluación donde coordenadas internas a los sólidos del conjunto 
            son reemplazadas por `np.nan`.
        shape : Tuple[int, int]
            Forma de la grilla `(Nx, Ny)` construida con `x` e `y`.
        CpLims : Sequence[float] length 2
            Límites del Cp por fuera de los cuales se reemplazan los valores calculados por
            `np.nan`.

        Returns
        -------
        Cp : np.ndarray shape (Nx, Ny)
            Cp en la grilla de evaluación.
        """
        
        it = int(it)

        Vx, Vy = self.V_campo(it, R_XY, shape)  # Se calculan las velocidades en la grilla de evaluación.
        
        if self.mpConfig['estacionario'] or all([not self.mpConfig['estacionario'], it == 0, self.mpConfig['vorticeArranque']]):
            VInf = self.conjuntos[it].VPC_XY[0][:, 0]    # Velocidad relativa al fluido en reposo de los sólidos.
            Vx = Vx - VInf[0]
            Vy = Vy - VInf[1]
            p_pRef = - self.refConfig['rho'] * (Vx ** 2 + Vy ** 2 - self.refConfig['V'] ** 2) / 2  # Se calcula la diferencia de presiones.
        else:
            dPhi_dt = self._dPhi_dt_campo(it, R_XY, shape)  # Se calculan las derivadas parciales temporales del potencial en la grilla de evaluación.
            p_pRef = - self.refConfig['rho'] * ((Vx ** 2 + Vy ** 2) / 2 + dPhi_dt)  # Se calcula la diferencia de presiones.            

        Cp = p_pRef / self.refConfig['Q']   # Se calcula el Cp.

        # Se filtra el Cp.
        CpMin, CpMax = CpLims
        Cp[Cp < CpMin] = np.nan
        Cp[Cp > CpMax] = np.nan

        return Cp
    
    def campo(
            self,
            it: int,
            x: np.ndarray,
            y: np.ndarray,
            radius: float | None = 0,
    ) -> Tuple[np.ndarray]:
        """
        Método para construir una grilla de coordenadas de evaluación donde coordenadas internas a 
        los sólidos del conjunto son reemplazadas por `np.nan`.

        Parameters
        ----------
        it : int
            Índice del instante bajo estudio.
        x : np.ndarray shape (Nx,)
            Valores de x con los cuales se construirá la grilla de evaluación.
        y : np.ndarray shape (Ny,)
            Valores de y con los cuales se construirá la grilla de evaluación.
        radius : float
            Parametro como el del mismo nombre de `matplotlib.path.Path.contains_points`.

            Sirve para ajustar la categorización de los puntos, como internos o externos
            a los sólidos, para ajustar la visualización y evitar mostrar los defectos de
            la singularidad concentrada en los nodos, inherente a una discretización con 
            dobletes constantes.

            - Si es positivo, un punto de evaluación cercano a la frontera del lado externo, se
            puede considerar dentro.
            - Si es negativo, un punto de evaluación cercano a la frontera del lado interno, se
            puede considerar fuera.

        Returns
        -------
        R_XY : np.ndarray shape (2, Nx * Ny)
            Coordenadas de evaluación donde coordenadas internas a 
            los sólidos del conjunto son reemplazadas por `np.nan`.
        shape : np.ndarray shape (2,)
            Forma de la grilla `(Nx, Ny)` construida con `x` e `y`.
        """

        it = int(it)

        conjunto = self.conjuntos[it]

        X, Y = np.meshgrid(x, y)
        n, m = X.shape
        X = X.flatten()
        Y = Y.flatten()

        R_XY = np.stack([X, Y], axis=0)

        interiores = np.zeros(R_XY.shape[1], dtype=bool)

        for solido in conjunto.solidos:
            R_XY_solido = solido.r_xy

            path = Path(R_XY_solido.T)

            interiores |= path.contains_points(R_XY.T, radius=radius).reshape(interiores.shape)

        R_XY[:, interiores] = np.nan
        shape = (n, m)
        return R_XY, shape
    
    def _phi_campo(
            self,
            it: int,
            R_XY: np.ndarray,
            shape: Tuple[int, int],
            th12_solidos_viejos: List[List[Tuple[np.ndarray]]] = None,
            th12_estelas_viejos: List[List[Tuple[np.ndarray]]] = None,
            th_vortices_viejos: List[List[np.ndarray]] = None,
    ) -> Tuple[List[np.ndarray], List[List[Tuple[np.ndarray]]], List[List[Tuple[np.ndarray]]], List[List[np.ndarray]]]:
        """
        Metodo que calcula el potencial inducido en puntos de evaluación en un instante dado.

        Notes
        -----
        - Se asume que los puntos de evaluación no se encuentran sobre los paneles.
        - Los potenciales aquí calculados no son consistentes entre ellos, pues a lo sumo lo son con los
        ángulos de referencia pasado. Esto conlleva a utilizarlos para hacer una visualización de ello
        presentaría defectos debido a los cortes.
        - Está pensado para utilizarse con el método MP2D.campo(...) que devuelve `R_XY`
        y `shape`.

        Parameters
        ----------
        it : int
            Índice del instante bajo estudio.
        R_XY : np.ndarray shape (2, Nx * Ny)
            Puntos de evaluación.
        shape : Tuple[int, int]
            Forma de la grilla `(Nx, My)` construida con `x` e `y`.
        th12_solidos_viejos : List[List[Optional[Tuple[np.ndarray]]]] each shape (Nx * Ny, N_s[j]), optional
            Ángulos viejos (de refeferencia) para resolver los cortes que puedan existir. Relativo a los sólidos.
        th12_estelas_viejos : List[List[Optional[Tuple[np.ndarray]]]] each shape (Nx * Ny, N_s[j]), optional
            Ángulos viejos (de refeferencia) para resolver los cortes que puedan existir. Relativo a los dobletes de la estela.
        th_vortices_viejos : List[List[Optional[Tuple[np.ndarray]]]] each shape (Nx * Ny, N_s[j]), optional
            Ángulos viejos (de refeferencia) para resolver los cortes que puedan existir. Relativo a los vórtices de la estela.
        
        Returns
        -------
        phi: List[np.ndarray] each shape (Nx * Ny,)
            Potenciales inducidos en los puntos de evaluación.
        th12_solidos_nuevos : List[List[Tuple[np.ndarray]]] each shape (Nx * Ny, N_s[j]), optional
            Ángulos nuevos. Relativo a los sólidos.
        th12_estelas_nuevos : List[List[Tuple[np.ndarray]]] each shape (Nx * Ny, N_s[j]), optional
            Ángulos nuevos. Relativo a los dobletes de la estela.
        th_vortices_nuevos : List[List[Tuple[np.ndarray]]] each shape (Nx * Ny, N_s[j]), optional
            Ángulos nuevos. Relativo a los vórtices de la estela.
        """
        
        it = int(it)
        conjunto = self.conjuntos[it]

        instanteInicial = False
        if not conjunto.estelas:
            instanteInicial = True

        th12_solidos_nuevos = []
        th12_estelas_nuevos = []
        th_vortices_nuevos = []

        M = R_XY.shape[1]
        phi = np.zeros((M,))

        # Bucle sobre los sólidos que influencian.
        for j in range(self.N_S):

            # Influencia por sólido
            # ---------------------
            th1, th2 = conjunto.solidos[j].th12(R_XY)   # (M, N). Se obtienen los ángulos desde los nodos hasta los puntos de evaluación.

            # Se desenvuelven los ángulos utilizando los de referencia, si es que se pasaron.
            if th12_solidos_viejos:
                th1 = self._desenvolverTheta(th1,th12_solidos_viejos[j][0])
                th2 = self._desenvolverTheta(th2, th12_solidos_viejos[j][1])

            th12_solidos_nuevos.append((th1, th2))

            # Se calcula el potencial considerando a todos los puntos de evaluación como si fueran externos a los paneles.
            intensidades = conjunto.solidos[j].intensidades
            phi_j = self._phi_ext_dobletes(
                th1,
                th2,
                intensidades
            )

            phi += np.sum(phi_j, axis=0)    # Se suma la contribución.

            # Influencia por estela (vórtice)
            # -------------------------------

            th = conjunto.vortices[j].th(R_XY)  # (M, N). Se obtienen los ángulos desde el vórtice de arranque hasta los puntos de evaluación.
            
            # Se desenvuelven los ángulos utilizando los de referencia, si es que se pasaron.
            if th_vortices_viejos:
                th = self._desenvolverTheta(th, th_vortices_viejos[j])
            
            # Se calcula el potencial.
            phi_j = self._phi_ext_vortices(
                th,
                conjunto.vortices[j].intensidades
            )

            th_vortices_nuevos.append(th)

            phi += phi_j[0,:]   # Se suma la contribución.
            
            # Influencia por estela (dobletes)
            # --------------------------------
            if not instanteInicial:
                th1, th2 = conjunto.estelas[j].th12(R_XY)   # (M, N). Se obtienen los ángulos desde los nodos hasta los puntos de evaluación.

                # Se desenvuelven los ángulos utilizando los de referencia, si es que se pasaron y si contienen
                # elementos válidos.
                if (th12_estelas_viejos
                    and len(th12_estelas_viejos) > j):

                    th1 = np.concatenate([th1[0:1,:], self._desenvolverTheta(th1[1:,:],th12_estelas_viejos[j][0])], axis=0)
                    th2 = np.concatenate([th2[0:1,:], self._desenvolverTheta(th2[1:,:], th12_estelas_viejos[j][1])], axis=0)
                
                th12_estelas_nuevos.append((th1, th2))

                # Se calcula el potencial. Los puntos de colocación siempre son externos a los dobletes.
                intensidades = conjunto.estelas[j].intensidades

                phi_j = self._phi_ext_dobletes(
                    th1,
                    th2,
                    intensidades,
                )

                phi += np.sum(phi_j, axis=0)    # Se suma la contribución.

        phi = phi.reshape(shape)
        return phi, th12_solidos_nuevos, th12_estelas_nuevos, th_vortices_nuevos
      
    def _dPhi_dt_campo(
            self,
            it: int,
            R_XY: np.ndarray,
            shape: Tuple[int, int],
            DtInicial: float = 1e-10
    ) -> Tuple[np.ndarray]:
        """
        Metodo que calcula la derivada parcial del potencial inducido en puntos de evaluación en un instante dado.

        Notes
        -----
        - Se asume que los puntos de evaluación no se encuentran sobre los paneles.
        - Está pensado para utilizarse con el método MP2D.campo(...) que devuelve `R_XY`
        y `shape`.

        Parameters
        ----------
        it : int
            Índice del instante bajo estudio.
        R_XY : np.ndarray shape (2, Nx * Ny)
            Puntos de evaluación.
        shape : Tuple[int, int]
            Forma de la grilla `(Nx, Ny)` construida con `x` e `y`.
        DtInicial : float, optional
            Valor utilizado para representar una diferencia de tiempos inicial instantánea. 
            
            La diferencia de tiempos se utiliza para la derivación respecto del tiempo, dividiendo por esta cantidad
            a la diferencia de potenciales.
            
            En el instante inicial, esta diferencia de tiempos se puede resolver de dos formas, dependiendo de la
            configuración utilizada:
            - Si `MP2D.mpConfig['vorticeArranque`]`: la diferencia de tiempos está indeterminada y se usa `np.nan`.
            - De lo contrario: se impone el valor `DtInicial`, que por defecto es `1e-10`.

        Returns
        -------
        dPhi_dt : np.ndarray shape (Nx, Ny)
            Derivada parcial del potencial inducido en la grilla de evaluación.
        """
        conjunto_nuevo = self.conjuntos[it]
                
        instanteInicial = False
        if not conjunto_nuevo.estelas:
            instanteInicial = True

        dPhi_dt = np.zeros(shape)

        # Instante inicial
        # ----------------
        if instanteInicial:
            # Primer instante estacionario, con lo cual la diferencia de tiempos está indeterminada.
            if self.mpConfig['vorticeArranque']:
                Dt = np.nan
            else:
                Dt = DtInicial

            phi_nuevo, _, _, _= self._phi_campo(it, R_XY, shape)        # Se calcula el nuevo potencial.  

            # Diferencias finitas.
            dPhi_dt += phi_nuevo / Dt

        # Instantes posteriores
        # --------------------- 
        else:
            Dt = self.t[it] - self.t[it-1]  # Se calcula el paso del tiempo.

            # Se calcula el potencial en el instante anterior.
            phi_viejo, th12_solidos_viejos, th12_estelas_viejos, th_vortices_viejos = self._phi_campo(it - 1, R_XY, shape)
            
            # Se calcula el potencial en el instante actual, desenvolviendo los ángulos con tomando de referencia
            # los viejos, para evitar diferencias por corte de ángulos.
            phi_nuevo, _, _, _= self._phi_campo(it, R_XY, shape, th12_solidos_viejos, th12_estelas_viejos, th_vortices_viejos)   

            # Diferencias finitas.
            dPhi_dt += (phi_nuevo - phi_viejo) / Dt
        return dPhi_dt
    
# =============================================================================
# Guardado y carga
# =============================================================================

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('cinematicas', None)
        state.pop('RM', None)
        return state
    
    def guardar(self, rutaBase: str):
        """
        Método para guardar la instancia actual de `MP2D.MP2D` en un archivo.
        
        .. note::
            Los generadores en `MP2D.cinematicas` y `MP2D.RM` no forman parte del objeto serializado. 
            Por lo tanto, al cargar el `.pkl` aquí guardado no se puede reanudar la simulación a menos 
            que se reconstruyan y alineen los generadores de las cinemáticas y RM.

            Para esto, también puede ser útil el guardado de la cinemátimca en
            un `.npz` utilizando el argumento `rutaBase` en `_Cinematicas.cinematica(...)`;
            y los argumentos opcionales `cinematicas` y `RM` de `MP2D.cargar(...)`.

        Parameters
        ----------
        rutaBase : str
            Ruta de guardado, sin incluir extensión.
        """
        with open(rutaBase + '.pkl', 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def guardar_resultados(
            self, 
            rutaBase: str, 
            extension: Literal['npz', 'mat'] = 'npz'
    ):
        """
        Método para guardar los resultados disponibles en un archivo `.npz` de Numpy.

        Las variables guardadas son `t` y las `*Rec`.

        Además se añaden las variables `R_XYRec` y `PC_XYRec`, que son las coordenadas en el sistema fijo
        de los nodos de los sólidos y sus puntos de colocación, respectivamente, en cada instante.

        .. note::
            En un futuro y si hace falta, podrían guardarse más variables.
        
        Parameters
        ----------
        rutaBase: str
            Ruta de guardado, sin incluir extensión (`.npz`).
        extension: Literal['npz', 'mat']
            Extensión del tipo de archivo al cual guardar.
            - `'npz'`: preparado para ser cargado con `MP2D.cargar_resultados(...)`.
            - `'mat'`: preparado para ser importado en Matlab.
        """
        def _to_object_array(x) -> np.ndarray:
            """
            Convierte listas/tuplas (posiblemente anidadas) en np.ndarray dtype=object
            sin que NumPy intente apilar por forma. Los leaves numéricos se pasan con np.asarray.
            """
            if isinstance(x, (list, tuple)):
                out = np.empty(len(x), dtype=object)
                for i, xi in enumerate(x):
                    out[i] = _to_object_array(xi)
                return out
            elif isinstance(x, np.ndarray) and x.dtype == object:
                # Ya es object: copiar elemento a elemento para evitar intentos de broadcast raros
                out = np.empty(x.shape, dtype=object)
                it = np.nditer(x, flags=['multi_index', 'refs_ok'], op_flags=['readonly'])
                for _ in it:
                    out[it.multi_index] = _to_object_array(x[it.multi_index])
                return out
            else:
                # Leaf: dejalo como ndarray numérico (o escalar) para que MATLAB lo vea como double
                return np.asarray(x)

        def _drop_none(d: dict) -> dict:
            return {k: v for k, v in d.items() if v is not None}
        
        R_XYRec = []
        for conjunto in self.conjuntos:
            R_XYRec.append(tuple(solido.r_xy for solido in conjunto.solidos))

        PC_XYRec = []
        for conjunto in self.conjuntos:
            PC_XYRec.append(conjunto.PC_XY)

        if extension == 'npz':
            np.savez(
                rutaBase + '.npz',
                t=np.asarray(self.t, dtype=float),
                R_XYRec=_to_object_array(R_XYRec),
                PC_XYRec=_to_object_array(PC_XYRec),
                RMRec=_to_object_array(getattr(self, "RMRec", None)) if getattr(self, "RMRec", None) is not None else None,
                CpRec=_to_object_array(getattr(self, "CpRec", None)) if getattr(self, "CpRec", None) is not None else None,
                dFRec=_to_object_array(getattr(self, "dFRec", None)) if getattr(self, "dFRec", None) is not None else None,
                CxyTotalRec=_to_object_array(getattr(self, "CxyTotalRec", None)) if getattr(self, "CxyTotalRec", None) is not None else None,
                CmTotalRec=np.asarray(getattr(self, "CmTotalRec", None), dtype=float) if getattr(self, "CmTotalRec", None) is not None else None,
                CxyRec=_to_object_array(getattr(self, "CxyRec", None)) if getattr(self, "CxyRec", None) is not None else None,
                CmRec=_to_object_array(getattr(self, "CmRec", None)) if getattr(self, "CmRec", None) is not None else None,
            )

        elif extension == 'mat':
            mdict = {
                't': np.asarray(self.t, dtype=float),
                'R_XYRec': _to_object_array(R_XYRec),
                'PC_XYRec': _to_object_array(PC_XYRec),                
                'RMRec': _to_object_array(getattr(self, "RMRec", None)) if getattr(self, "RMRec", None) is not None else None,
                'CpRec': _to_object_array(getattr(self, "CpRec", None)) if getattr(self, "CpRec", None) is not None else None,
                'dFRec': _to_object_array(getattr(self, "dFRec", None)) if getattr(self, "dFRec", None) is not None else None,
                'CxyTotalRec': _to_object_array(getattr(self, "CxyTotalRec", None)) if getattr(self, "CxyTotalRec", None) is not None else None,
                'CmTotalRec': np.asarray(getattr(self, "CmTotalRec", None), dtype=float) if getattr(self, "CmTotalRec", None) is not None else None,
                'CxyRec': _to_object_array(getattr(self, "CxyRec", None)) if getattr(self, "CxyRec", None) is not None else None,
                'CmRec': _to_object_array(getattr(self, "CmRec", None)) if getattr(self, "CmRec", None) is not None else None,
            }
            savemat(rutaBase + '.mat', _drop_none(mdict))
        else:
            raise ValueError("extension debe ser 'npz' o 'mat'")

    def guardar_coeficientes_csv(
            self,
            ruta: str,
            sep: str = ',',
            precision: int = 10,
            encoding: str = 'utf-8-sig',
    ):
        """
        Guarda coeficientes aerodinámicos en CSV. La primera columna es 'alfa' (modo estacionario)
        o 't' (modo no estacionario).

        Parámetros
        ----------
        ruta : str
            Ruta del archivo a guardar. Si no termina en '.csv', se agrega.
        sep : str
            Separador. Por defecto, `,`.
        precision : int
            Precisión de escritura numérica. Por defecto, `10`.
        encoding : str
            Codificación de archivo. Por defecto `'utf-8-sig'`.
        """
        import os
        import csv
        import numpy as np
        from math import isnan

        ruta_csv = ruta if str(ruta).lower().endswith('.csv') else (str(ruta) + '.csv')

        n = len(getattr(self, 't', []))
        if n == 0:
            raise RuntimeError("No hay resultados para guardar (self.t está vacío).")

        # Encabezado de la primera columna
        estacionario = bool(self.mpConfig.get('estacionario', False))
        encabezado_t = 'alfa' if estacionario else 't'
        x_col = np.asarray(self.t, dtype=float)

        # Totales
        if not hasattr(self, 'CxyTotalRec') or len(self.CxyTotalRec) != n:
            raise RuntimeError("CxyTotalRec no disponible o longitud inconsistente.")

        Cx_total = np.array([cxy[0] for cxy in self.CxyTotalRec], dtype=float)
        Cy_total = np.array([cxy[1] for cxy in self.CxyTotalRec], dtype=float)

        hay_momentos = hasattr(self, 'CmTotalRec') and (self.CmTotalRec is not None) and (len(self.CmTotalRec) == n)
        if hay_momentos:
            Cm_total = np.asarray(self.CmTotalRec, dtype=float)

        headers = [encabezado_t, 'Cx', 'Cy'] + (['Cm'] if hay_momentos else [])
        columnas = [x_col, Cx_total, Cy_total] + ([Cm_total] if hay_momentos else [])

        # Por sólido (si hay más de uno)
        Ns = int(getattr(self, 'N_S', 1))
        if Ns > 1:
            if not hasattr(self, 'CxyRec') or len(self.CxyRec) != n:
                raise RuntimeError("CxyRec no disponible o longitud inconsistente.")
            nombres = list(self.mpConfig.get('nombres', [f"solido{i+1}" for i in range(Ns)]))
            if len(nombres) != Ns:
                nombres = [nombres[i] if i < len(nombres) else f"solido{i+1}" for i in range(Ns)]

            # Fuerzas por sólido
            for i in range(Ns):
                nombre = str(nombres[i])
                Cx_i = np.array([cxy[0, i] for cxy in self.CxyRec], dtype=float)
                Cy_i = np.array([cxy[1, i] for cxy in self.CxyRec], dtype=float)
                headers += [f'Cx_{nombre}', f'Cy_{nombre}']
                columnas += [Cx_i, Cy_i]

            # Momentos por sólido (si existen)
            if hay_momentos and hasattr(self, 'CmRec') and (self.CmRec is not None) and (len(self.CmRec) == n):
                for i in range(Ns):
                    nombre = str(nombres[i])
                    Cm_i = np.array([cm[i] for cm in self.CmRec], dtype=float)
                    headers.append(f'Cm_{nombre}')
                    columnas.append(Cm_i)

        # Apilado y guardado
        matriz = np.column_stack(columnas)

        formato = f'%.{precision}g'
        os.makedirs(os.path.dirname(ruta_csv) or ".", exist_ok=True)
        with open(ruta_csv, 'w', newline='', encoding=encoding) as f:
            w = csv.writer(f, delimiter=sep)
            w.writerow(headers)
            for fila in matriz:
                w.writerow([
                    "" if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else (formato % x)
                    for x in fila
                ])

        return ruta_csv
    
def cargarMP2D(
        rutaBase: str, 
        cinematicas: List[GeneradorCinematica] | List[GeneradorCinematicaAeroelasticidad] | None = None,
        RM : GeneradorRM | GeneradorRMAeroelasticidad | None = None,
        purgar: bool = True
) -> MP2D:
    """
    Función para cargar una instancia de `MP2D.MP2D` guardada en un archivo `.pkl`.
    
    .. note::
        Los generadores en `MP2D.cinematicas` y `MP2D.RM` no forman parte del objeto serializado. 
        Por lo tanto, al cargar el `.pkl` guardado con `MP2D.guardar(...)` no es posible reanudar
        la simulación de forma directa.

        Para ello se introducen los argumentos adicionales `cinematicas` y `RM`.

        Si no se desea el comportamiento de purga, definir los atributos manualmente.

    Parameters
    ----------
    rutaBase : str
        Ruta del archivo, sin incluir extensión.
    cinematicas : List[Tipos.GeneradorCinematica] | List[Tipos.GeneradorCinematicaAeroelasticidad] | None
        Conjunto de generadores obtenidos con `_Cinematicas.cinematica` con los que se construyan los 
        sólidos de la simulación.

        - En el caso no aeroelástico, se purgarán las cinemáticas hasta alinear el generador con el último
        paso de simulación resuelto si `purgar` es `True`.

        - En el caso aeroelástico, solo se efectuará normalmente la purga inicial del generador.

        Por defecto, es `None` y no se inicializa el atributo `cinematicas` de la instancia cargada.
    
    RM : Tipos.GeneradorRM | Tipos.GeneradorRMAeroelasticidad | None
        Conjunto de generadores obtenidos con `_Cinematicas.RM` con los que se obtienen los vectores de
        toma de momento para el cálculo de momentos aerodinámicos.

        Por defecto, no se utiliza y no se calculan momentos aerodinámicos.

        También puede ser None (por defecto) y en este caso no se computan momentos aerodinámicos.

        - En el caso no aeroelástico, se purgarán las cinemáticas hasta alinear el generador con el último
        paso de simulación resuelto si `purgar` es `True`.

        - En el caso aeroelástico, solo se efectuará normalmente la purga inicial del generador.

        Por defecto, es `None` y no se inicializa el atributo `RM` de la instancia cargada.

    purgar : bool
        Define si se purgan o no los generadores `cinematicas` y `RM` se cargaren.
    """
    with open(rutaBase + '.pkl', 'rb') as f:
        mp: MP2D = pickle.load(f)

    if cinematicas is not None:
        mp.cinematicas = cinematicas
        if mp.mpConfig['aeroelasticidad']:
            [next(cinematica) for cinematica in mp.cinematicas]
        else:
            if purgar:
                for _ in range(mp.it + 1):
                    [next(cinematica) for cinematica in mp.cinematicas]
    
    if RM is not None:
        mp.RM = RM
        if mp.mpConfig['RMAeroelasticidad']:
            next(mp.RM)
        else:
            if purgar:
                for _ in range(mp.it + 1):
                    next(mp.RM)
    return mp

def cargar_resultados(rutaBase: str):
    """
    Función para cargar los resultados guardados con `MP2D.guardar_resultados(...)`
    en un archivo `.npz` de Numpy.

    Parameters
    ----------
    rutaBase: str
        Ruta de guardado, sin incluir extensión (`.npz`).

    Returns
    -------
    resultados: dict
        Diccionario con las claves:
        - `t`
        - `RMRec`
        - `CpRec`
        - `dF_Rec`
        - `CxyTotalRec`
        - `CmTotalRec`
        - `CxyRec`
        - `CmRec`
    """
    return np.load(rutaBase + '.npz', allow_pickle=True)
        