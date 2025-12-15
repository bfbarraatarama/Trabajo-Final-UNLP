# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# =======================================================================================
# Proyecto: Trabajo Final - Método de Paneles Bidimensional Multielemento No estacionario
# Archivo: Ejemplos/rec/NACATN2465.py
# Autor: Bruno Francisco Barra Atarama
# Institución: 
#   Departamento de Ingeniería Aeroespacial
#   Facultad de Ingeniería
#   Universidad Nacional de La Plata
# Año: 2025
# Licencia: PolyForm Noncommercial License 1.0.0
# =======================================================================================

from numpy.typing import NDArray

import numpy as np
from scipy.special import hankel2
import matplotlib.pyplot as plt
import csv

"""
Funciones auxiliares del test de validación que contrasta con el NACA Technical Note no. 2465.
"""

# =============================================================================
# Curvas teóricas de Theodorsen
# =============================================================================
def theodorsenRotacion(k: NDArray, a0: float, c: float, c_rot: float, plotFlag: bool=False, ax_L=None, ax_M=None):
    a0 = np.deg2rad(a0)
    
    b = c / 2
    a = (c_rot - b) / b

    F, G = C_theodorsen(k)

    R_LP = - np.pi * a0 * (a * k ** 2 / 2 + F - (1 / 2 - a) * k * G)
    I_LP = - np.pi * a0 * (k / 2 + G + (1 / 2 - a) * k * F)

    R_MP = np.pi * a0 * (k ** 2 / 2 * (1 / 8 + a ** 2) + (1 / 2 + a) * (F - (1 / 2 - a) * k * G))
    I_MP = - np.pi * a0 * (k / 2 * (1 / 2 - a) - (1 / 2 + a) * (G + (1 / 2 - a) * k * F))

    mod_LP = np.sqrt(R_LP ** 2 + I_LP ** 2)
    fase_LP = np.rad2deg(np.atan2(I_LP, R_LP))

    mod_MP = np.sqrt(R_MP ** 2 + I_MP ** 2)
    fase_MP = np.rad2deg(np.atan2(I_MP, R_MP))

    LP = (k, mod_LP, fase_LP)
    MP = (k, mod_MP, fase_MP)
    aux = (R_LP, I_LP, R_MP, I_MP)

    return LP, MP, aux

def theodorsenTraslacion(k: NDArray, h0: float, c: float, c_rot: float, plotFlag: bool=False, ax_L=None, ax_M=None):
    b = c / 2
    a = (c_rot - b) / b

    F, G = C_theodorsen(k)

    R_LT = np.pi * h0 / b * (k ** 2 / 2 + k* G)
    I_LT = - np.pi * h0 / b * k * F

    R_MT = - np.pi * h0 / b * (a * k ** 2 / 2 + (1 / 2 + a) * k * G)
    I_MT = np.pi * h0 / b * (1 / 2 + a) * k * F

    mod_LT = np.sqrt(R_LT ** 2 + I_LT ** 2)
    fase_LT = np.rad2deg(np.atan2(I_LT, R_LT))

    mod_MT = np.sqrt(R_MT ** 2 + I_MT ** 2)
    fase_MT = np.rad2deg(np.atan2(I_MT, R_MT))

    LT = (k, mod_LT, fase_LT)
    MT = (k, mod_MT, fase_MT)

    aux = (R_LT, I_LT, R_MT, I_MT)

    return LT, MT, aux

def theodorsenRototraslacion(k: float, a0: float, h0: float, phi: NDArray, c: float, c_rot: float, plotFlag: bool, ax_L=None, ax_M=None):
    _, _, aux_P = theodorsenRotacion(k, a0, c, c_rot)
    _, _, aux_T = theodorsenTraslacion(k, h0, c, c_rot)
    
    R_LP, I_LP, R_MP, I_MP = aux_P
    R_LT, I_LT, R_MT, I_MT = aux_T

    R_LR = R_LT + R_LP * np.cos(phi) - I_LP * np.sin(phi)
    I_LR = I_LT + R_LP * np.sin(phi) + I_LP * np.cos(phi)

    R_MR = R_MT + R_MP * np.cos(phi) - I_MP * np.sin(phi)
    I_MR = I_MT + R_MP * np.sin(phi) + I_MP * np.cos(phi)

    mod_LR = np.sqrt(R_LR ** 2 + I_LR ** 2)
    fase_LR = np.rad2deg(np.atan2(I_LR, R_LR))

    mod_MR = np.sqrt(R_MR ** 2 + I_MR ** 2)
    fase_MR = np.rad2deg(np.atan2(I_MR, R_MR))

    LR = (k, mod_LR, fase_LR)
    MR = (k, mod_MR, fase_MR)

    return LR, MR

def C_theodorsen(k: NDArray) -> NDArray[np.complex128]:
    
    H0 = hankel2(0, k)
    H1 = hankel2(1, k)
    
    C = H1 / (H1 + 1j * H0)
    
    F = np.real(C)
    G = np.imag(C)
    return F, G

# =============================================================================
# Lectura de datos experimentales
# =============================================================================

def leerExperimento(ruta):
    with open(ruta, newline='') as f:
        reader = csv.DictReader(f)
        cols = {h: [] for h in reader.fieldnames}
        for row in reader:
            for h in reader.fieldnames:
                cols[h].append(row[h])
    for h, lst in cols.items():
        cols[h] = np.array(lst, dtype=float)
    return cols

# =============================================================================
# Manipulación de señales
# =============================================================================

def faseWrap(fase, fase_base):
    return (fase - fase_base) % 360 + fase_base

def analisisSenalSenoidal(t, y):
    t = np.array(t)
    y = np.array(y)
    idc_crucePorCero = np.where((y[:-1] * y[1:] < 0) | (y[:-1] == 0))[0]

    if y[idc_crucePorCero[-1]] == 0:
        idc_crucePorCero = idc_crucePorCero[:-1]

    idc_ceroPendPos = np.where(y[idc_crucePorCero + 1] - y[idc_crucePorCero] > 0)[0]

    idc_ceroPendPos = idc_crucePorCero[idc_ceroPendPos]

    if len(idc_ceroPendPos) >= 2:
        idc_ceroPendPos = idc_ceroPendPos[-2:]
    else:
        raise RuntimeError('No hay suficientes cruces por cero con pendiente positiva para estimar el desfase respecto de una senoidal')
    
    pendiente = []
    t0 = []
    for idc in idc_ceroPendPos:
        t1 = t[idc]
        t2 = t[idc + 1]

        y1 = y[idc]
        y2 = y[idc + 1]

        pendiente.append((y2 - y1) / (t2 - t1))
        t0.append(t1 - y1 / pendiente[-1])

    periodo = t0[1] - t0[0]
    pendiente = pendiente[1]
    t0 = t0[1]

    amplitud = periodo * pendiente / 2 / np.pi

    fase = (- 2 * np.pi / periodo * t0) % (2 * np.pi)
    if fase > np.pi:
        fase -= 2 * np.pi

    fase = np.rad2deg(fase)
    return amplitud, fase, periodo

# =============================================================================
# Graficación
# =============================================================================

def plotTeoricoExperimental(LTeo, MTeo, LSim, MSim, rutaExperimento, fase_base, titulo, fig, ax):
    [a.grid(True) for a in ax.flat]
    [a.set_axisbelow(True) for a in ax.flat]

    experimento = leerExperimento(rutaExperimento)

    teo = LTeo
    exp = (experimento['k'], experimento['L'], experimento['phi_L'])
    sim = LSim

    plotTeoExp(teo, exp, sim, fase_base[0], ax[0,:])

    teo = MTeo
    exp = (experimento['k'], experimento['M'], experimento['phi_M'])
    sim = MSim

    plotTeoExp(teo, exp, sim, fase_base[1], ax[1,:])

    ax[0,0].set_xlabel(r'$k = \frac{\omega b}{V}$')
    ax[0,0].set_ylabel(r'$\sqrt{R_L^2 + I_L^2}$')

    ax[0,1].set_xlabel(r'$k = \frac{\omega b}{V}$')
    ax[0,1].set_ylabel(r'$\phi_L$')

    ax[1,0].set_xlabel(r'$k = \frac{\omega b}{V}$')
    ax[1,0].set_ylabel(r'$\sqrt{R_M^2 + I_M^2}$')

    ax[1,1].set_xlabel(r'$k = \frac{\omega b}{V}$')
    ax[1,1].set_ylabel(r'$\phi_M$')

    fig.suptitle(titulo)
    return fig, ax

def plotTeoExp(teo, exp, sim, fase_base, ax):
    kteo = teo[0]
    modteo = teo[1]
    faseteo = teo[2]

    kexp = exp[0]
    modexp = exp[1]
    faseexp = exp[2]

    if sim is not None:
        ksim = sim[0]
        modsim = sim[1]
        fasesim = sim[2]

        fasesim = faseWrap(fasesim, fase_base)

    faseteo = faseWrap(faseteo, fase_base)
    faseexp = faseWrap(faseexp, fase_base)

    ax[0].plot(kteo, modteo, label='Theodorsen')
    ax[0].scatter(kexp, modexp, color='k', label='NACA TN 2465')

    ax[1].plot(kteo, faseteo, label='Theodorsen')
    ax[1].scatter(kexp, faseexp, color='k', label='NACA TN 2465')

    if sim is not None:
        ax[0].scatter(ksim, modsim, color='r', label='MP2D')
        ax[1].scatter(ksim, fasesim, color='r', label='MP2D')
    
    ax[0].legend()
    ax[1].legend()
    return ax

