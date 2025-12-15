# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# =======================================================================================
# Proyecto: Trabajo Final - Método de Paneles Bidimensional Multielemento No estacionario
# Archivo: src/__init__.py
# Autor: Bruno Francisco Barra Atarama
# Institución: 
#   Departamento de Ingeniería Aeroespacial
#   Facultad de Ingeniería
#   Universidad Nacional de La Plata
# Año: 2025
# Licencia: PolyForm Noncommercial License 1.0.0
# =======================================================================================

"""
Inicializa el paquete y expone los módulos MP2D, Cinematicas, Importacion y Tipos.
"""

from . import MP2D
from . import Cinematicas
from . import Importacion

__all__ = ['MP2D', 'Cinematicas', 'Importacion', 'Tipos']