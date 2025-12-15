
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# =======================================================================================
# Proyecto: Trabajo Final - Método de Paneles Bidimensional Multielemento No estacionario
# Archivo: gui/GUI.py
# Autor: Bruno Francisco Barra Atarama
# Institución: 
#   Departamento de Ingeniería Aeroespacial
#   Facultad de Ingeniería
#   Universidad Nacional de La Plata
# Año: 2025
# Licencia: PolyForm Noncommercial License 1.0.0
# =======================================================================================

"""
Interfaz gráfica.
"""

import numpy as np
import os, sys, re, uuid, pickle
import traceback
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Literal, Callable, Tuple, TypedDict, cast
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

from PySide6.QtCore import Qt, Signal, QLocale
from PySide6.QtGui import QDoubleValidator, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QLabel, QFormLayout,
    QLineEdit, QSpinBox, QHBoxLayout, QComboBox, QPushButton, QSplitter,
    QFrame, QListWidget, QStackedWidget, QCheckBox, QGroupBox, QSizePolicy, QFileDialog,
    QAbstractItemView, QListWidgetItem, QTextEdit, QDialog, QDialogButtonBox, QMessageBox,
    QScrollArea, QProgressBar, QGridLayout, QSlider, QTextBrowser
)

from src.Importacion import importarPerfil
from src import Cinematicas, Tipos
from src.MP2D import MP2D
from src._TernasMoviles2D import TernasMoviles2D


BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
ABOUT_IMAGE_PATH = "logos.png"
LICENSE_PATH = "../LICENSE"

def _resolve_path(p: str) -> str:
    if not p: return ""
    return p if os.path.isabs(p) else os.path.join(BASE_DIR, p)

def _num(edit: QLineEdit, placeholder="0.0"):
    v = QDoubleValidator(bottom=-1e12, top=1e12, decimals=6)
    v.setLocale(QLocale())
    v.setNotation(QDoubleValidator.ScientificNotation)
    edit.setLocale(QLocale())
    edit.setValidator(v)
    edit.setPlaceholderText(placeholder)

def _bbox_from_rxy_list(rxy_list: List[Tuple[str, np.ndarray]]) -> Optional[Tuple[float, float, float, float]]:
    if not rxy_list: return None
    xs, ys = [], []
    for _, r in rxy_list:
        if r is None or r.size == 0: continue
        xs.append(r[0, :]); ys.append(r[1, :])
    if not xs or not ys: return None
    X = np.concatenate(xs); Y = np.concatenate(ys)
    return (float(np.min(X)), float(np.max(X)), float(np.min(Y)), float(np.max(Y)))

PgIndex = Literal[0, 1, 2, 3, 4, 5, 6]

class GraficoConfig(TypedDict):
    pg: PgIndex
    idcFlag: bool
    escala: bool
    instLabel: str
    vrel: bool
    Rm: bool

class CommonsConfig(TypedDict):
    instante: int
    mostrar_RM: bool
    V_relativa: bool
    escala: Optional[float]

class CoefsCommonsConfig(TypedDict):
    instante: int
    invertir_m: bool
    totales: bool

class CpCampoConfig(CommonsConfig):
    radio: Optional[float]
    mostrar_superficies: bool
    niveles: int
    xmin: Optional[float]; xmax: Optional[float]; dx: Optional[float]
    ymin: Optional[float]; ymax: Optional[float]; dy: Optional[float]

class VCampoConfig(CommonsConfig):
    radio: Optional[float]
    mostrar_superficies: bool
    xmin: Optional[float]; xmax: Optional[float]; dx: Optional[float]
    ymin: Optional[float]; ymax: Optional[float]; dy: Optional[float]

class SeleccionConfig(TypedDict):
    tipo: str
    pg: int
    idcFlag: bool
    instLabel: str
    escala: bool
    vrel: bool
    Rm: bool

class AllGraphsConfig(TypedDict):
    seleccion: SeleccionConfig
    paneles: CommonsConfig
    coeficientes: CoefsCommonsConfig
    cp: CommonsConfig
    cp_vectorial: CommonsConfig
    v: CommonsConfig
    cp_campo: CpCampoConfig
    v_campo: VCampoConfig

@dataclass
class FlapConfig:
    cf: float
    df: float
    h_TEw_ROTf: float
    v_TEw_ROTf: float
    h_ROTf_BAf: float
    v_ROTf_MCf: float

@dataclass
class SolidConfig:
    id: str
    nombre: str
    perfil_path: Optional[str]
    formato: str
    c: float
    theta1_deg: float
    dx: float
    dy: float
    theta2_deg: float
    n_intra: int
    n_extra: int
    es_flap: bool
    flap: Optional[FlapConfig] = None
    cerrar_te: bool = False
    perfil_missing: bool = False

@dataclass
class ConjuntoConfig:
    theta1_deg: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    theta2_deg: float = 0.0

ModoSim = Literal["estacionario", "no_estacionario"]
SubmodoNE = Literal["mru", "rotacion", "traslacion", "archivo"]

@dataclass
class SimulationConfig:
    modo: ModoSim
    V: Optional[float] = None
    alfa_i: Optional[float] = None
    alfa_f: Optional[float] = None
    delta_alfa: Optional[float] = None
    submodo: Optional[SubmodoNE] = None
    alfa: Optional[float] = None
    w: Optional[float] = None
    h0: Optional[float] = None
    t_final: Optional[float] = None
    dt: Optional[float] = None
    archivo_path: Optional[str] = None
    rm_archivo_path: Optional[str] = None
    archivos_por_solido: Optional[Dict[str, str]] = None

@dataclass
class AppState:
    solidos: Dict[str, SolidConfig] = field(default_factory=dict)
    conjunto: ConjuntoConfig = field(default_factory=ConjuntoConfig)
    mp2d: Optional[object] = None
    l_ref: float = 0.6096
    V_ref: float = 35.76
    rho_ref: float = 1.225
    def to_dict(self) -> dict:
        return {"conjunto": asdict(self.conjunto), "solidos": [asdict(s) for s in self.solidos.values()],
                "referencia":{"l":self.l_ref,"V":self.V_ref,"rho":self.rho_ref}}

class LicenseDialog(QDialog):
    def __init__(self, parent=None, text: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Licencia")
        v = QVBoxLayout(self)
        self.te = QTextEdit(self); self.te.setReadOnly(True); self.te.setPlainText(text or "Texto de licencia no definido.")
        v.addWidget(self.te)
        btns = QDialogButtonBox(QDialogButtonBox.Close, self); btns.rejected.connect(self.reject); btns.accepted.connect(self.accept)
        v.addWidget(btns)

class AboutTab(QWidget):
    def __init__(self):
        super().__init__()

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        root.addWidget(self.scroll)

        content = QWidget()
        self.scroll.setWidget(content)
        v = QVBoxLayout(content)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        self.desc = QTextBrowser()
        self.desc.setFrameShape(QFrame.NoFrame)
        self.desc.setOpenExternalLinks(True)
        self.desc.setReadOnly(True)
        self.desc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.desc.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.desc.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.desc.setStyleSheet("QTextBrowser { font-size: 12pt; }")

        self.desc.setHtml("""
        <div style="line-height:1.25em;">
          <p style="margin:0 0 6px 0;"><b>Autor:</b> Bruno Francisco Barra Atarama</p>
          <p style="margin:0 0 6px 0;"><b>Institución:</b><br>
          Departamento de Ingeniería Aeroespacial<br>
          Facultad de Ingeniería<br>
          Universidad Nacional de La Plata</p>
          <p style="margin:0 0 6px 0;"><b>Año:</b> 2025</p>
          <p style="margin:0 0 6px 0;"><b>Descripción:</b></p>
          <p style="margin:0 0 6px 0;">
            Esta es una interfaz gráfica de la implementación del método de los paneles bidimensional,
            potencial, estacionario y no estacionario, con paneles rectos y distribuciones constantes
            de dobletes y vórtices de arranque puntuales, desarrollada en Python en el contexto de mi
            trabajo final desarrollado en la mencionada institución.
          </p>
          <p style="margin:0 0 6px 0;">
            Esta interfaz tiene el propósito de acercar la herramienta desarrollada a más estudiantes
            y usuarios interesados, que no necesariamente deseen hacer uso de los programas mediante la
            codificación en Python. Pese a su mayor accesibilidad y facilidad de uso, este formato
            presenta algunas limitaciones respecto a la versión original.
          </p>
          <p style="margin:0;">
            Por eso, en el repositorio de GIT donde también se encuentra el escrito de mi trabajo,
            los programas y ejemplos en Python y Jupyter Notebooks, se pone a disposición el código
            fuente de esta interfaz para poder ser ampliada y mejorada según necesidades particulares,
            así como el manual de uso de la versión más actualizada.
          </p>
          <p style="margin:8px 0 0 0;"><b>Repositorio:</b>
            <a href="https://github.com/bfbarraatarama/Trabajo-Final-UNLP">
              https://github.com/bfbarraatarama/Trabajo-Final-UNLP
            </a>
          </p>
          <p style="margin:0;">
            Si el enlace no fuera accesible, podés escribirme a:
            <a href="mailto:bfbarraatarama@gmail.com">bfbarraatarama@gmail.com</a>
          </p>
          <p style="margin:0 0 6px 0;"><b>Licencia:</b> PolyForm Noncommercial License 1.0.0</p>
        </div>
        """)
        v.addWidget(self.desc, 0)

        self.btnLic = QPushButton("Licencia…")
        self.btnLic.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        v.addWidget(self.btnLic, 0)

        self.imgLabel = QLabel()
        self.imgLabel.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.imgLabel.setFrameShape(QFrame.NoFrame)
        self.imgLabel.setScaledContents(False)
        self.imgLabel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        v.addWidget(self.imgLabel, 0)

        self.btnLic.clicked.connect(self._onLicense)
        self._license_text = ""
        self._orig_pm: Optional[QPixmap] = None
        self._load_license()
        self._load_fixed_image()

        self._update_desc_height()

    def _update_desc_height(self):
        doc = self.desc.document()
        viewport_w = max(100, self.desc.viewport().width())
        doc.setTextWidth(viewport_w)
        h = doc.size().height()
        extra = 4
        self.desc.setFixedHeight(int(h) + extra)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._update_desc_height()
        self._update_scaled_pixmap()

    def _load_license(self):
        path = _resolve_path(LICENSE_PATH)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                self._license_text = f.read()
        except Exception:
            self._license_text = "Texto de licencia no disponible."

    def _onLicense(self):
        LicenseDialog(self, self._license_text).exec()

    def _load_fixed_image(self):
        path = _resolve_path(ABOUT_IMAGE_PATH)
        if not os.path.exists(path):
            self._orig_pm = None
            self.imgLabel.setText("Imagen no disponible.\nEditá ABOUT_IMAGE_PATH.")
            self.imgLabel.adjustSize()
            return
        pm = QPixmap(path)
        if pm.isNull():
            self._orig_pm = None
            self.imgLabel.setText("No se pudo cargar la imagen.")
            self.imgLabel.adjustSize()
            return
        self._orig_pm = pm
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if not self._orig_pm:
            return
        avail_w = max(120, self.scroll.viewport().width() - 8)
        target_w = max(320, int(avail_w * 0.6))
        ratio = self._orig_pm.height() / self._orig_pm.width()
        target_h = max(60, int(target_w * ratio))
        scaled = self._orig_pm.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.imgLabel.setPixmap(scaled)
        self.imgLabel.setFixedSize(scaled.size())

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('MP2D - Bruno F. Barra Atarama - Departamento de Ingenieria Aeroespacial - FI UNLP')
        self.state = AppState()

        tabs = QTabWidget()
        self.tabSolidos = SolidosTab(self.state)
        self.tabSim = SimulacionTab(self.state)
        self.tabRes = ResTab()
        self.tabGuard = GuardadoTab(self.state)
        self.tabAbout = AboutTab()

        tabs.addTab(self.tabSolidos, 'Sólidos')
        tabs.addTab(self.tabSim, 'Simulación')
        tabs.addTab(self.tabRes, 'Resultados')
        tabs.addTab(self.tabGuard, 'Guardado y carga')
        tabs.addTab(self.tabAbout, 'Acerca de')
        self.setCentralWidget(tabs)

        self.tabSim.modoChanged.connect(self.tabRes.setContext)
        self.tabSim.simulateRequested.connect(self._onSimulate)
        self.tabRes.set_result_provider(self._provide_results)

        self.tabSolidos.solidsChanged.connect(self.tabSim.refresh_archivo_per_solid_rows)
        self.tabSim.refresh_archivo_per_solid_rows()

        self.resize(1200, 800)

    def _provide_results(self) -> Tuple[Optional[object], dict]:
        rxy_list = self.tabSolidos.build_all_rxy()
        meta = {"rxy_list": rxy_list}
        return (self.state.mp2d, meta)

    def _onSimulate(self, sim_cfg: SimulationConfig):
        try:
            self.state.l_ref = float(self.tabSim.edL.text()    or self.state.l_ref)
            self.state.V_ref = float(self.tabSim.edVref.text() or self.state.V_ref)
            self.state.rho_ref = float(self.tabSim.edRho.text()  or self.state.rho_ref)
            nu_ref = float(self.tabSim.edNu.text() or "1.5e-5")
            a_ref = float(self.tabSim.edA.text()  or "340.3") 
            g_ref = float(self.tabSim.edG.text()  or "9.81")

            faltantes_perfil = []
            for s in self.state.solidos.values():
                if (s.perfil_path is None) or s.perfil_missing or (not os.path.exists(s.perfil_path)):
                    faltantes_perfil.append(f"- {s.nombre}: {s.perfil_path or '(sin ruta)'}")

            if faltantes_perfil:
                QMessageBox.warning(
                    self, "Faltan archivos de coordenadas",
                    "No se puede iniciar la simulación porque faltan archivos de coordenadas:\n\n" +
                    "\n".join(faltantes_perfil)
                )
                return

            rxy_named = self.tabSolidos.build_all_rxy()
            rxy_list = [r for _, r in rxy_named]

            nombres = [nombre for nombre, _ in rxy_named]
            mpConfig: Tipos.MPConfig = {'mostrarProgreso': False, 'nombres' : nombres}
            refConfig: Tipos.RefConfig = {
                'l': self.state.l_ref,
                'V': self.state.V_ref,
                'rho': self.state.rho_ref,
                'nu': nu_ref,
                'a' : a_ref,
                'g' : g_ref
            }   

            # =========================
            #        ESTACIONARIO
            # =========================
            if sim_cfg.modo == 'estacionario':
                mpConfig['estacionario'] = True
                mpConfig['rotulo_t']     = 'α, [°]'
                mpConfig['invertirCm'] = True

                if (sim_cfg.delta_alfa or 0) == 0:
                    alfa = np.array([float(sim_cfg.alfa_i or 0)], dtype=float)
                else:
                    alfa = np.arange(float(sim_cfg.alfa_i or 0), float(sim_cfg.alfa_f or 0) + 1e-12,
                                    float(sim_cfg.delta_alfa or 0.0), dtype=float)
                nPasos = len(alfa)
                RM = (np.zeros((2, 1)) for _ in range(len(alfa)))

                cinematicas = []
                for r_xy in rxy_list:
                    params: Tipos.AOAParams = {
                        'alfa': alfa,
                        'r_xy': r_xy,
                        'V': float(sim_cfg.V or self.state.V_ref)
                    }
                    cinematicas.append(Cinematicas.cinematica('aoa', params))

            # =========================
            #     NO ESTACIONARIO
            # =========================
            else:
                mpConfig['estacionario'] = False
                mpConfig['rotulo_t']     = 't, [s]'
                mpConfig['invertirCm'] = False

                submodo = sim_cfg.submodo

                if submodo != 'archivo':
                    V = float(sim_cfg.V or self.state.V_ref)
                    alfa = float(sim_cfg.alfa or 0.0)
                    w = float(sim_cfg.w or 0.0)
                    h0 = float(sim_cfg.h0 or 0.0)
                    t_final= sim_cfg.t_final
                    dt = sim_cfg.dt
                    t = np.arange(0, t_final + 1e-15, dt)
                else:
                    rutas_por_sid = sim_cfg.archivos_por_solido or {}
                    faltantes_kins = []

                    if sim_cfg.rm_archivo_path:
                        if not os.path.exists(sim_cfg.rm_archivo_path):
                            faltantes_kins.append(f"- RM: {sim_cfg.rm_archivo_path}")

                    id_a_nombre = { sid: s.nombre for sid, s in self.state.solidos.items() }
                    for sid, ruta in rutas_por_sid.items():
                        if ruta and (not os.path.exists(ruta)):
                            nombre = id_a_nombre.get(sid, f"Sólido (id {sid})")
                            faltantes_kins.append(f"- {nombre}: {ruta}")

                    if faltantes_kins:
                        QMessageBox.warning(
                            self, "Faltan archivos de cinemáticas (No estacionario)",
                            "No se puede iniciar la simulación porque faltan archivos de cinemática:\n\n" +
                            "\n".join(faltantes_kins)
                        )
                        return

                cinematicas: list = []
                if submodo == "mru":
                    V = float(sim_cfg.V or self.state.V_ref)
                    alfa = float(sim_cfg.alfa or 0.0)
                    t_final = sim_cfg.t_final
                    dt = sim_cfg.dt
                    t = np.arange(0, t_final + 1e-15, dt)
                    for r_xy in rxy_list:
                        params: Tipos.MRUParams = {'alfa': alfa, 'r_xy': r_xy, 't': t, 'V': V}
                        cinematicas.append(Cinematicas.cinematica('mru', params))
                    RM = Cinematicas.cinematica('mru', params)
                    RM = Cinematicas.RM('desdeCinematicaRO', {'cinematica' : RM})

                elif submodo == "rotacion":
                    V = float(sim_cfg.V or self.state.V_ref)
                    alfa = float(sim_cfg.alfa or 0.0)
                    w = float(sim_cfg.w or 0.0)
                    t_final = sim_cfg.t_final
                    dt = sim_cfg.dt
                    t = np.arange(0, t_final + 1e-15, dt)
                    for r_xy in rxy_list:
                        params: Tipos.RotacionArmonicaParams = {'a0': alfa, 'r_xy': r_xy, 't': t, 'V': V, 'w': w}
                        cinematicas.append(Cinematicas.cinematica('rotacionArmonica', params))
                    RM = Cinematicas.cinematica('rotacionArmonica', params)
                    RM = Cinematicas.RM('desdeCinematicaRO', {'cinematica' : RM})

                elif submodo == "traslacion":
                    V = float(sim_cfg.V or self.state.V_ref)
                    h0 = float(sim_cfg.h0 or 0.0)
                    w = float(sim_cfg.w or 0.0)
                    t_final = sim_cfg.t_final
                    dt = sim_cfg.dt
                    t = np.arange(0, t_final + 1e-15, dt)
                    for r_xy in rxy_list:
                        params: Tipos.TraslacionArmonicaParams = {'h0': h0, 'r_xy': r_xy, 't': t, 'V': V, 'w': w}
                        cinematicas.append(Cinematicas.cinematica('traslacionArmonica', params))
                    RM = Cinematicas.cinematica('traslacionArmonica', params)
                    RM = Cinematicas.RM('desdeCinematicaRO', {'cinematica' : RM})

                elif submodo == "archivo":
                    rxy_named = self.tabSolidos.build_all_rxy()  # [(nombre, r_xy), ...]
                    rutas_por_sid = sim_cfg.archivos_por_solido or {}

                    rutas_por_nombre = { s.nombre: rutas_por_sid.get(sid) for sid, s in self.state.solidos.items() }

                    relativizar_por_sid = self.tabSim.get_relativizar_flags()  # {sid: bool}
                    relativizar_por_nombre = { s.nombre: relativizar_por_sid.get(sid, True) for sid, s in self.state.solidos.items() }

                    for nombre, r_xy in rxy_named:
                        ruta = rutas_por_nombre.get(nombre)
                        ruta, _ = os.path.splitext(ruta)
                        if relativizar_por_nombre.get(nombre, True):
                            params: Tipos.CuerpoRigidoCSVParams = {'r_xy' : r_xy, 'rutaBase' : ruta}
                            t, RO, theta, _, _, _, _ = Cinematicas._cuerpoRigidoCSV(r_xy, ruta)
                            r_xy = TernasMoviles2D(RO[0], np.array(theta[0:1])).R2r(r_xy)
                        params: Tipos.CuerpoRigidoCSVParams = {'r_xy' : r_xy, 'rutaBase' : ruta}
                        cinematicas.append(Cinematicas.cinematica('cuerpoRigidoCSV', params))
                    try:
                        t
                    except:
                        t, _, _, _, _, _, _ = Cinematicas._cuerpoRigidoCSV(r_xy, ruta)
                    ruta , _ = os.path.splitext(sim_cfg.rm_archivo_path)
                    RMParams: Tipos.RMCSVParams = {'rutaBase' : ruta}
                    RM = Cinematicas.RM('csv', RMParams)
                else:
                    cinematicas = [None for _ in rxy_list]

                nPasos = len(t)

            mp = MP2D(cinematicas, RM, mpConfig, refConfig)
            self.tabSim.start_progress(nPasos)
            for i in range(nPasos):
                mp.simular(1)
                self.tabSim.set_progress(i + 1)
                QApplication.processEvents()
            self.tabSim.finish_progress()

            self.state.mp2d = mp
            inst_count = len(mp.t)

            self.tabRes.setContext(sim_cfg.modo == 'estacionario', inst_count)

            self.tabSim.update_adim_info_from_mp(mp)

        except Exception:
            traceback.print_exc()
            QMessageBox.warning(self, "Advertencia", "La simulación no pudo ser completada con éxito")


class AdimInfoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        self.title = QLabel("Relaciones adimensionales:")
        self.title.setStyleSheet("font-weight: 600;")
        v.addWidget(self.title)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        v.addLayout(grid)

        def ro(qss: str = ""):
            e = QLineEdit(); e.setReadOnly(True)
            if qss:
                e.setStyleSheet(qss)
            return e

        # ---- Fila 1
        self.lblRe = QLabel("Re")
        self.edRe  = ro()
        self.lblMa = QLabel("Ma")
        self.edMa = ro()
        self.lblFr = QLabel("Fr")
        self.edFr = ro()

        grid.addWidget(self.lblRe, 0, 0); grid.addWidget(self.edRe, 0, 1)
        grid.addWidget(self.lblMa, 0, 2); grid.addWidget(self.edMa, 0, 3)
        grid.addWidget(self.lblFr, 0, 4); grid.addWidget(self.edFr, 0, 5)

        # ---- Fila 2
        self.lblInvRe = QLabel("1/Re")
        self.edInvRe = ro()
        self.lblInvFr2 = QLabel("1/Fr<sup>2</sup>"); self.lblInvFr2.setTextFormat(Qt.RichText)
        self.edInvFr2 = ro()

        grid.addWidget(self.lblInvRe, 1, 0); grid.addWidget(self.edInvRe, 1, 1)
        grid.addWidget(self.lblInvFr2, 1, 2); grid.addWidget(self.edInvFr2, 1, 3)

        # ---- Fila 3
        self.lblMa2 = QLabel("Ma<sup>2</sup>"); self.lblMa2.setTextFormat(Qt.RichText)
        self.edMa2 = ro()
        self.lblMa2Re = QLabel("Ma<sup>2</sup>/Re"); self.lblMa2Re.setTextFormat(Qt.RichText)
        self.edMa2Re = ro()
        self.lblMa2Fr2 = QLabel("Ma<sup>2</sup>/Fr<sup>2</sup>"); self.lblMa2Fr2.setTextFormat(Qt.RichText)
        self.edMa2Fr2 = ro()

        grid.addWidget(self.lblMa2, 2, 0); grid.addWidget(self.edMa2, 2, 1)
        grid.addWidget(self.lblMa2Re, 2, 2); grid.addWidget(self.edMa2Re, 2, 3)
        grid.addWidget(self.lblMa2Fr2, 2, 4); grid.addWidget(self.edMa2Fr2, 2, 5)

        self.setVisible(False)

    def set_data(self, d: Optional[dict]):
        if not d:
            self.setVisible(False)
            for ed in (self.edRe, self.edMa, self.edFr, self.edInvRe, self.edInvFr2,
                       self.edMa2, self.edMa2Re, self.edMa2Fr2):
                ed.clear()
            return

        def fmt(x):
            try:
                return f"{float(x):.6g}"
            except Exception:
                return ""

        self.edRe.setText(fmt(d.get("Re")))
        self.edMa.setText(fmt(d.get("Ma")))
        self.edFr.setText(fmt(d.get("Fr")))
        self.edInvRe.setText(fmt(d.get("_Re")))
        self.edInvFr2.setText(fmt(d.get("_Fr2")))
        self.edMa2.setText(fmt(d.get("Ma2")))
        self.edMa2Re.setText(fmt(d.get("Ma2_Re")))
        self.edMa2Fr2.setText(fmt(d.get("Ma2_Fr2")))

        self.setVisible(True)

class ParametrosComunesBox(QWidget):
    indiceChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)

        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignRight)
        form.setFormAlignment(Qt.AlignTop)

        self.lblGrupo = QLabel("Selección de instante:", self)
        self.lblGrupo.setStyleSheet("font-weight: 600;")
        form.addRow(self.lblGrupo)

        self.lblInstante = QLabel("Índice instante [ ]:", self)

        row_idx = QWidget(self)
        lay_idx = QHBoxLayout(row_idx)
        lay_idx.setContentsMargins(0, 0, 0, 0)
        lay_idx.setSpacing(6)

        self.spInstante = QSpinBox(row_idx)
        self.spInstante.setRange(0, 0)
        self.spInstante.setSingleStep(1)
        self.spInstante.setButtonSymbols(QSpinBox.PlusMinus)

        self.slInstante  = QSlider(Qt.Horizontal, row_idx)
        self.slInstante.setRange(0, 0)
        self.slInstante.setSingleStep(1)
        self.slInstante.setPageStep(1)
        self.slInstante.setTickPosition(QSlider.NoTicks)

        lay_idx.addWidget(self.spInstante, 0)
        lay_idx.addWidget(self.slInstante, 1)
        form.addRow(self.lblInstante, row_idx)

        self.lblSel = QLabel("t [s]:", self)
        self.edSel  = QLineEdit(self)
        self.edSel.setReadOnly(True)
        self.edSel.setPlaceholderText("")
        form.addRow(self.lblSel, self.edSel)

        self.lblRm = QLabel('Mostrar RM:', self); self.chkRm = QCheckBox(self)
        self.lblVRel = QLabel("V relativa:", self); self.vrel = QCheckBox(self); self.vrel.setChecked(True)
        self.lblEscala = QLabel("Escala:", self); self.escala = QLineEdit(self); _num(self.escala, '1.0'); self.escala.setText("1.0")
        form.addRow(self.lblRm, self.chkRm)
        form.addRow(self.lblVRel, self.vrel)
        form.addRow(self.lblEscala, self.escala)

        self._value_provider: Optional[Callable[[int], Optional[float]]] = None

        self.slInstante.valueChanged.connect(self.spInstante.setValue)
        self.spInstante.valueChanged.connect(self.slInstante.setValue)

        self.slInstante.valueChanged.connect(self._emit_index_changed)
        self.spInstante.valueChanged.connect(self._emit_index_changed)

        self.slInstante.valueChanged.connect(self._update_selected_value)

    def setRangoInstante(self, max_idx: int):
        max_idx = max(0, int(max_idx))
        self.slInstante.setRange(0, max_idx)
        self.spInstante.setRange(0, max_idx)
        cur = min(self.slInstante.value(), max_idx)
        self.slInstante.setValue(cur)
        self._update_selected_value()

    def configurar(self, *, mostrarInstante: bool, etiquetaInstante: str,
                   mostrarEscala: bool, mostrarVRel: bool, mostrarRm: bool):
        self._apply_label_mode_from_current(etiquetaInstante)
        for w, on in [
            (self.lblGrupo, mostrarInstante),
            (self.lblInstante, mostrarInstante),
            (self.spInstante, mostrarInstante),
            (self.slInstante, mostrarInstante),
            (self.lblSel, mostrarInstante),
            (self.edSel, mostrarInstante),
            (self.lblRm, mostrarRm), (self.chkRm, mostrarRm),
            (self.lblVRel, mostrarVRel), (self.vrel, mostrarVRel),
            (self.lblEscala, mostrarEscala), (self.escala, mostrarEscala),
        ]:
            w.setVisible(on)
        self._update_selected_value()

    def setValueProvider(self, provider: Optional[Callable[[int], Optional[float]]], unit_label: str):
        self._value_provider = provider
        self.lblSel.setText(unit_label)
        self._apply_label_mode(unit_label)
        self._update_selected_value()

    def getIndex(self) -> int:
        return int(self.slInstante.value())

    def setIndex(self, i: int):
        i = int(max(self.slInstante.minimum(), min(self.slInstante.maximum(), int(i))))
        self.slInstante.setValue(i)
        self._update_selected_value()

    def _apply_label_mode(self, unit_label: str):
        if "α" in unit_label:
            self.lblGrupo.setText("Selección de α:")
            self.lblInstante.setText("Índice α [ ]:")
        else:
            self.lblGrupo.setText("Selección de instante:")
            self.lblInstante.setText("Índice instante [ ]:")

    def _apply_label_mode_from_current(self, etiquetaInstante_backup: str):
        cur = self.lblSel.text() or ""
        if "α" in cur or "t" in cur:
            self._apply_label_mode(cur)
        else:
            if "α" in etiquetaInstante_backup:
                self.lblGrupo.setText("Selección de α:")
            else:
                self.lblGrupo.setText("Selección de instante:")
            self.lblInstante.setText(etiquetaInstante_backup or self.lblInstante.text())

    def _update_selected_value(self):
        idx = int(self.slInstante.value())
        val_text = ""
        if self._value_provider is not None:
            try:
                val = self._value_provider(idx)
                if val is not None:
                    val_text = f"{val}"
            except Exception:
                val_text = ""
        self.edSel.setText(val_text)

    def _emit_index_changed(self, v: int):
        self.indiceChanged.emit(int(v))

class ParametrosCoefBox(QWidget):
    indiceChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)

        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignRight)
        form.setFormAlignment(Qt.AlignTop)

        self.lblGrupo = QLabel("Selección de instante:", self)
        self.lblGrupo.setStyleSheet("font-weight: 600;")
        form.addRow(self.lblGrupo)

        self.lblInstante = QLabel("Índice instante [ ]:", self)

        row_idx = QWidget(self)
        lay_idx = QHBoxLayout(row_idx); lay_idx.setContentsMargins(0, 0, 0, 0); lay_idx.setSpacing(6)

        self.spInstante = QSpinBox(row_idx)
        self.spInstante.setRange(0, 0)
        self.spInstante.setSingleStep(1)
        self.spInstante.setButtonSymbols(QSpinBox.PlusMinus)

        self.slInstante  = QSlider(Qt.Horizontal, row_idx)
        self.slInstante.setRange(0, 0)
        self.slInstante.setSingleStep(1)
        self.slInstante.setPageStep(1)
        self.slInstante.setTickPosition(QSlider.NoTicks)

        lay_idx.addWidget(self.spInstante, 0)
        lay_idx.addWidget(self.slInstante, 1)
        form.addRow(self.lblInstante, row_idx)

        self.lblSel = QLabel("t [s]:", self)
        self.edSel = QLineEdit(self); self.edSel.setReadOnly(True)
        form.addRow(self.lblSel, self.edSel)

        self.chkInvertM = QCheckBox("Invertir momento")
        self.chkTot = QCheckBox("Totales")
        form.addRow(self.chkInvertM)
        form.addRow(self.chkTot)

        self._value_provider: Optional[Callable[[int], Optional[float]]] = None

        self.slInstante.valueChanged.connect(self.spInstante.setValue)
        self.spInstante.valueChanged.connect(self.slInstante.setValue)
        self.slInstante.valueChanged.connect(self._update_selected_value)
        self.slInstante.valueChanged.connect(self._emit_index_changed)
        self.spInstante.valueChanged.connect(self._emit_index_changed)

    def setRangoInstante(self, max_idx: int):
        max_idx = max(0, int(max_idx))
        self.slInstante.setRange(0, max_idx)
        self.spInstante.setRange(0, max_idx)
        cur = min(self.slInstante.value(), max_idx)
        self.slInstante.setValue(cur)
        self._update_selected_value()

    def configurar(self, *, mostrarInstante: bool, etiquetaInstante: str):
        self._apply_label_mode_from_current(etiquetaInstante)
        for w, on in [
            (self.lblGrupo, mostrarInstante),
            (self.lblInstante, mostrarInstante),
            (self.spInstante, mostrarInstante),
            (self.slInstante, mostrarInstante),
            (self.lblSel, mostrarInstante),
            (self.edSel, mostrarInstante),
        ]:
            w.setVisible(on)
        self._update_selected_value()

    def setValueProvider(self, provider: Optional[Callable[[int], Optional[float]]], unit_label: str):
        self._value_provider = provider
        self.lblSel.setText(unit_label)
        self._apply_label_mode(unit_label)
        self._update_selected_value()

    def getIndex(self) -> int:
        return int(self.slInstante.value())

    def setIndex(self, i: int):
        i = int(max(self.slInstante.minimum(), min(self.slInstante.maximum(), int(i))))
        self.slInstante.setValue(i)
        self._update_selected_value()

    def _apply_label_mode(self, unit_label: str):
        if "α" in unit_label:
            self.lblGrupo.setText("Selección de α:")
            self.lblInstante.setText("Índice α [ ]:")
        else:
            self.lblGrupo.setText("Selección de instante:")
            self.lblInstante.setText("Índice instante [ ]:")

    def _apply_label_mode_from_current(self, etiquetaInstante_backup: str):
        cur = self.lblSel.text() or ""
        if "α" in cur or "t" in cur:
            self._apply_label_mode(cur)
        else:
            if "α" in etiquetaInstante_backup:
                self.lblGrupo.setText("Selección de α:")
            else:
                self.lblGrupo.setText("Selección de instante:")
            self.lblInstante.setText(etiquetaInstante_backup or self.lblInstante.text())

    def _update_selected_value(self):
        idx = int(self.slInstante.value())
        val_text = ""
        if self._value_provider is not None:
            try:
                val = self._value_provider(idx)
                if val is not None:
                    val_text = f"{val}"
            except Exception:
                val_text = ""
        self.edSel.setText(val_text)

    def _emit_index_changed(self, v: int):
        self.indiceChanged.emit(int(v))

class ResTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self); layout.setContentsMargins(8, 8, 8, 8); layout.setSpacing(8)

        tipoRow = QHBoxLayout(); tipoRow.setContentsMargins(0, 0, 0, 0); tipoRow.setSpacing(8)
        tipoRow.addWidget(QLabel('Tipo de gráfico:'))
        self.cbTipoGrafico = QComboBox()
        tipoRow.addWidget(self.cbTipoGrafico, 1)
        layout.addLayout(tipoRow)

        self.stackParams = QStackedWidget(); self.stackParams.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addWidget(self.stackParams); layout.setAlignment(self.stackParams, Qt.AlignTop)

        # Paneles
        pgPaneles = QWidget(); self.formPaneles = QFormLayout(pgPaneles); self._compactForm(self.formPaneles)
        self._commonsPaneles = ParametrosComunesBox(pgPaneles); self.formPaneles.addRow(self._commonsPaneles)

        # Coeficientes
        pgCoef = QWidget(); self.formCoef = QFormLayout(pgCoef); self._compactForm(self.formCoef)
        self._coefBox = ParametrosCoefBox(pgCoef); self.formCoef.addRow(self._coefBox)

        # Cp
        pgCp = QWidget(); self.formCp = QFormLayout(pgCp); self._compactForm(self.formCp)
        self._commonsCp = ParametrosComunesBox(pgCp); self.formCp.addRow(self._commonsCp)

        # Cp vectorial
        pgCpVect = QWidget(); self.formCpVect = QFormLayout(pgCpVect); self._compactForm(self.formCpVect)
        self._commonsCpVect = ParametrosComunesBox(pgCpVect); self.formCpVect.addRow(self._commonsCpVect)
        self._commonsCpVect.escala.setText('0.1')

        # V
        pgV = QWidget(); self.formV = QFormLayout(pgV); self._compactForm(self.formV)
        self._commonsV = ParametrosComunesBox(pgV); self.formV.addRow(self._commonsV)
        self._commonsV.escala.setText("0.001")

        # Cp campo
        pgCpCampo = QWidget(); self.formCpCampo = QFormLayout(pgCpCampo); self._compactForm(self.formCpCampo)
        self._commonsCpCampo = ParametrosComunesBox(pgCpCampo)
        self.lblRadioCp = QLabel("Radio de exclusión:", pgCpCampo); self.edRadioCp = QLineEdit(pgCpCampo); _num(self.edRadioCp, "-1e-3"); self.edRadioCp.setText("-1e-3")
        self.lblSuperf = QLabel("Mostrar en superficies:", pgCpCampo); self.chkSuperf = QCheckBox(pgCpCampo); self.chkSuperf.setChecked(True)
        self.lblNiv = QLabel("Niveles:", pgCpCampo); self.spNiveles = QSpinBox(pgCpCampo); self.spNiveles.setRange(1, 999); self.spNiveles.setValue(200)

        gx1 = QHBoxLayout(); self.edXminCp=QLineEdit(); _num(self.edXminCp)
        self.edXmaxCp=QLineEdit(); _num(self.edXmaxCp)
        self.edDxCp=QLineEdit(); _num(self.edDxCp)
        gx1.addWidget(QLabel("x mín:")); gx1.addWidget(self.edXminCp)
        gx1.addWidget(QLabel("x máx:")); gx1.addWidget(self.edXmaxCp)
        gx1.addWidget(QLabel("Δx:"));   gx1.addWidget(self.edDxCp)

        gy1 = QHBoxLayout(); self.edYminCp=QLineEdit(); _num(self.edYminCp)
        self.edYmaxCp=QLineEdit(); _num(self.edYmaxCp)
        self.edDyCp=QLineEdit(); _num(self.edDyCp)
        gy1.addWidget(QLabel("y mín:")); gy1.addWidget(self.edYminCp)
        gy1.addWidget(QLabel("y máx:")); gy1.addWidget(self.edYmaxCp)
        gy1.addWidget(QLabel("Δy:"));   gy1.addWidget(self.edDyCp)

        self.btnResetGridCp = QPushButton("Reiniciar grilla")
        self.btnResetGridCp.setToolTip("Borra xmin/xmax/Δx/ymin/ymax/Δy para que al presionar 'Mostrar' se recalculen automáticamente con el instante elegido.")
        self.btnResetGridCp.clicked.connect(self._onResetGridCp)

        self.formCpCampo.addRow(self._commonsCpCampo)
        self.formCpCampo.addRow(self.lblRadioCp, self.edRadioCp)
        self.formCpCampo.addRow(self.lblSuperf, self.chkSuperf)
        self.formCpCampo.addRow(self.lblNiv, self.spNiveles)
        self.formCpCampo.addRow(gx1); self.formCpCampo.addRow(gy1)
        self.formCpCampo.addRow(self.btnResetGridCp)

        # V campo
        pgVCampo = QWidget(); self.formVCampo = QFormLayout(pgVCampo); self._compactForm(self.formVCampo)
        self._commonsVCampo = ParametrosComunesBox(pgVCampo)
        self.formVCampo.addRow(self._commonsVCampo)
        self._commonsVCampo.escala.setText("0.00005")

        self.lblRadioV = QLabel("Radio de exclusión:", pgVCampo)
        self.edRadioV = QLineEdit(pgVCampo); _num(self.edRadioV, "-1e-3"); self.edRadioV.setText("-1e-3")

        self.lblSuperfV = QLabel("Mostrar en superficies:", pgVCampo)
        self.chkSuperfV = QCheckBox(pgVCampo); self.chkSuperfV.setChecked(True)

        gx2 = QHBoxLayout()
        self.edXminV = QLineEdit(); _num(self.edXminV)
        self.edXmaxV = QLineEdit(); _num(self.edXmaxV)
        self.edDxV = QLineEdit(); _num(self.edDxV)
        gx2.addWidget(QLabel("x mín:")); gx2.addWidget(self.edXminV)
        gx2.addWidget(QLabel("x máx:")); gx2.addWidget(self.edXmaxV)
        gx2.addWidget(QLabel("Δx:"));   gx2.addWidget(self.edDxV)

        gy2 = QHBoxLayout()
        self.edYminV = QLineEdit(); _num(self.edYminV)
        self.edYmaxV = QLineEdit(); _num(self.edYmaxV)
        self.edDyV = QLineEdit(); _num(self.edDyV)
        gy2.addWidget(QLabel("y mín:")); gy2.addWidget(self.edYminV)
        gy2.addWidget(QLabel("ymáx:"));  gy2.addWidget(self.edYmaxV)   
        gy2.addWidget(QLabel("Δy:"));    gy2.addWidget(self.edDyV)

        self.btnResetGridV = QPushButton("Reiniciar grilla")
        self.btnResetGridV.setToolTip("Borra xmin/xmax/Δx/ymin/ymax/Δy para que al presionar 'Mostrar' se recalcule automáticamente con el instante elegido.")
        self.btnResetGridV.clicked.connect(self._onResetGridV)

        self.formVCampo.addRow(self._commonsVCampo)
        self.formVCampo.addRow(self.lblRadioV,  self.edRadioV)
        self.formVCampo.addRow(self.lblSuperfV, self.chkSuperfV)
        self.formVCampo.addRow(gx2)
        self.formVCampo.addRow(gy2)
        self.formVCampo.addRow(self.btnResetGridV)

        self.stackParams.addWidget(pgPaneles)
        self.stackParams.addWidget(pgCoef)
        self.stackParams.addWidget(pgCp)
        self.stackParams.addWidget(pgCpVect)
        self.stackParams.addWidget(pgV)
        self.stackParams.addWidget(pgCpCampo)
        self.stackParams.addWidget(pgVCampo)

        # ---- Botón Mostrar
        acc = QHBoxLayout(); acc.setContentsMargins(0,0,0,0); acc.setSpacing(8)
        self.btnMostrar = QPushButton('Mostrar'); acc.addWidget(self.btnMostrar); acc.addStretch(1)
        layout.addLayout(acc)

        # ---- Zona de plot
        self.plotBox = QGroupBox("Gráfico")
        self.plotBoxLayout = QVBoxLayout(self.plotBox)
        layout.addWidget(self.plotBox, 1)

        # ---- Estado/contexto
        self.cbTipoGrafico.currentIndexChanged.connect(self._onTipoGraficoChanged)
        self.btnMostrar.clicked.connect(self._onMostrar)
        self._context = {'estacionario': True, 'instantes': 1}
        self._result_provider: Optional[Callable[[], Tuple[Optional[object], dict]]] = None
        self._tipoSegunContexto()

        self._shared_index: int = 0 
        self._broadcasting: bool = False
        self._wire_shared_index()

    def set_result_provider(self, provider: Callable[[], Tuple[Optional[object], dict]]):
        self._result_provider = provider

    def _compactForm(self, form: QFormLayout):
        form.setContentsMargins(0, 0, 0, 0); form.setSpacing(6)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignRight); form.setFormAlignment(Qt.AlignTop)

    def setContext(self, estacionario: bool, instantes: int | list = 1):
        self._context = {'estacionario': estacionario, 'instantes': instantes}
        self._tipoSegunContexto()

        inst = self._context.get("instantes", 1)
        max_idx = (inst - 1) if isinstance(inst, int) else max(0, len(inst) - 1)
        for commons in (self._commonsPaneles, self._commonsCp, self._commonsCpVect,
                        self._commonsV, self._commonsCpCampo, self._commonsVCampo):
            commons.setRangoInstante(max_idx)
        self._coefBox.setRangoInstante(max_idx)

        unit_label = "α [°]:" if estacionario else "t [s]:"
        for commons in (self._commonsPaneles, self._commonsCp, self._commonsCpVect,
                        self._commonsV, self._commonsCpCampo, self._commonsVCampo):
            commons.setValueProvider(self._get_mp_t_value, unit_label)
        self._coefBox.setValueProvider(self._get_mp_t_value, unit_label)

        if estacionario:
            self._coefBox.chkInvertM.setChecked(False)
            self._coefBox.chkInvertM.setVisible(False)
        else:
            self._coefBox.chkInvertM.setText("Invertir Cz")
            self._coefBox.chkInvertM.setVisible(True)

        self._coefBox.chkTot.setChecked(False)

        self._shared_index = max(0, min(self._shared_index, max_idx))
        self._broadcast_shared_index()

    def _tipoSegunContexto(self):
        self.cbTipoGrafico.blockSignals(True)
        self.cbTipoGrafico.clear()
        estacionario = self._context.get('estacionario', True)

        nombre_coefs = 'Cd, Cl y Cm' if estacionario else 'Cx, Cy y Cz'

        if estacionario:
            opciones = [
                ('Paneles',{'pg':0,'idcFlag':True,'escala':False,'instLabel':'Índice α [ ]:','vrel':False,'Rm':True}),
                (nombre_coefs,{'pg':1,'idcFlag':False,'escala':False,'instLabel':'','vrel':False,'Rm':False}),
                ('Cp',{'pg':2,'idcFlag':True,'escala':False,'instLabel':'Índice α [ ]:','vrel':False,'Rm':False}),
                ('Cp vectorial',{'pg':3,'idcFlag':True,'escala':True,'instLabel':'Índice α [ ]:','vrel':False,'Rm':False}),
                ('Cp campo',{'pg':5,'idcFlag':True,'escala':False,'instLabel':'Índice α [ ]:','vrel':False,'Rm':False}),
                ('V',{'pg':4,'idcFlag':True,'escala':True,'instLabel':'Índice α [ ]:','vrel':True,'Rm':False}),
                ('V campo',{'pg':6,'idcFlag':True,'escala':True,'instLabel':'Índice α [ ]:','vrel':True,'Rm':False}),
            ]
        else:
            opciones = [
                ('Paneles',{'pg':0,'idcFlag':True,'escala':False,'instLabel':'Índice instante [ ]:','vrel':False,'Rm':True}),
                (nombre_coefs,{'pg':1,'idcFlag':False,'escala':False,'instLabel':'','vrel':False,'Rm':False}),
                ('Cp',{'pg':2,'idcFlag':True,'escala':False,'instLabel':'Índice instante [ ]:','vrel':False,'Rm':False}),
                ('Cp vectorial',{'pg':3,'idcFlag':True,'escala':True,'instLabel':'Índice instante [ ]:','vrel':False,'Rm':False}),
                ('Cp campo',{'pg':5,'idcFlag':True,'escala':False,'instLabel':'Índice instante [ ]:','vrel':False,'Rm':False}),
                ('V',{'pg':4,'idcFlag':True,'escala':True,'instLabel':'Índice instante [ ]:','vrel':True,'Rm':False}),
                ('V campo',{'pg':6,'idcFlag':True,'escala':True,'instLabel':'Índice instante [ ]:','vrel':True,'Rm':False}),
            ]

        for nombre, data in opciones:
            self.cbTipoGrafico.addItem(nombre, data)

        inst = self._context.get("instantes", 1)
        max_idx = (inst - 1) if isinstance(inst, int) else max(0, len(inst) - 1)
        for commons in (self._commonsPaneles, self._commonsCp, self._commonsCpVect,
                        self._commonsV, self._commonsCpCampo, self._commonsVCampo):
            commons.setRangoInstante(max_idx)
        self._coefBox.setRangoInstante(max_idx)

        self._coefBox.chkTot.setChecked(False)

        self.cbTipoGrafico.blockSignals(False)
        self._onTipoGraficoChanged(self.cbTipoGrafico.currentIndex())

    def _onTipoGraficoChanged(self, _idc: int):
        data = cast(GraficoConfig, self.cbTipoGrafico.currentData() or {})
        if not data: return
        pg = int(data.get('pg', 0))
        self.stackParams.setCurrentIndex(pg)

        mostrarInst = bool(data.get('idcFlag', False))
        instLabel = str(data.get('instLabel', 'Índice instante [ ]:'))
        mostrarEsc = bool(data.get('escala', False))
        mostrarVRel = bool(data.get('vrel', False))
        mostrarRm = bool(data.get('Rm', False))

        for w in (self._commonsPaneles, self._coefBox, self._commonsCp,
                  self._commonsCpVect, self._commonsV, self._commonsCpCampo, self._commonsVCampo):
            w.setVisible(False)

        commons_map = {
            0:self._commonsPaneles, 1:self._coefBox, 2:self._commonsCp,
            3:self._commonsCpVect, 4:self._commonsV, 5:self._commonsCpCampo, 6:self._commonsVCampo
        }
        commons = commons_map.get(pg)
        if commons is None: return
        commons.setVisible(True)

        if pg == 1:
            self._coefBox.configurar(mostrarInstante=mostrarInst, etiquetaInstante=instLabel or "Índice instante [ ]:")
        else:
            commons: ParametrosComunesBox
            commons.configurar(mostrarInstante=mostrarInst, etiquetaInstante=instLabel,
                               mostrarEscala=mostrarEsc, mostrarVRel=mostrarVRel, mostrarRm=mostrarRm)

    def _clear_plot_area(self):
        while self.plotBoxLayout.count():
            item = self.plotBoxLayout.takeAt(0); w = item.widget()
            if w is not None: w.deleteLater()

    def _add_figure(self, fig: plt.Figure, canvas_first: bool = True):
        canvas = FigureCanvasQTAgg(fig); toolbar = NavigationToolbar2QT(canvas, self)
        if canvas_first:
            self.plotBoxLayout.addWidget(canvas); self.plotBoxLayout.addWidget(toolbar)
        else:
            self.plotBoxLayout.addWidget(toolbar); self.plotBoxLayout.addWidget(canvas)
        canvas.draw()

    def _add_figure_return_canvas(self, fig: plt.Figure, canvas_first: bool = True):
        canvas = FigureCanvasQTAgg(fig); toolbar = NavigationToolbar2QT(canvas, self)
        if canvas_first:
            self.plotBoxLayout.addWidget(canvas); self.plotBoxLayout.addWidget(toolbar)
        else:
            self.plotBoxLayout.addWidget(toolbar); self.plotBoxLayout.addWidget(canvas)
        canvas.draw()
        return canvas

    def _flt(self, txt: str) -> Optional[float]:
        try:
            s = (txt or "").strip()
            if not s: return None
            return float(s)
        except Exception:
            return None

    @staticmethod
    def _flt_or_none(s: str) -> Optional[float]:
        try:
            t = (s or "").strip()
            return None if t == "" else float(t)
        except Exception:
            return None

    def _snap_commons(self, commons: ParametrosComunesBox) -> CommonsConfig:
        return {
            'instante': commons.getIndex(),                 # <-- toma el índice actual (compartido)
            'mostrar_RM': commons.chkRm.isChecked(),
            'V_relativa': commons.vrel.isChecked(),
            'escala': self._flt(commons.escala.text()),
        }

    def _snap_coefs(self) -> CoefsCommonsConfig:
        return {
            'instante': self._coefBox.getIndex(),
            'invertir_m': self._coefBox.chkInvertM.isChecked(),
            'totales': self._coefBox.chkTot.isChecked(),
        }

    def _collect_all_graph_configs(self, nombre_actual: str, data_actual: GraficoConfig) -> AllGraphsConfig:
        cfg: AllGraphsConfig = {
            'seleccion': {
                'tipo': nombre_actual, 'pg': int(data_actual.get('pg', 0)),
                'idcFlag': bool(data_actual.get('idcFlag', False)),
                'instLabel': str(data_actual.get('instLabel', '')),
                'escala': bool(data_actual.get('escala', False)),
                'vrel': bool(data_actual.get('vrel', False)),
                'Rm': bool(data_actual.get('Rm', False))
            },
            'paneles': self._snap_commons(self._commonsPaneles),
            'coeficientes': self._snap_coefs(),
            'cp': self._snap_commons(self._commonsCp),
            'cp_vectorial': self._snap_commons(self._commonsCpVect),
            'v': self._snap_commons(self._commonsV),
            'cp_campo': {
                **self._snap_commons(self._commonsCpCampo),
                'radio': self._flt(self.edRadioCp.text()),
                'mostrar_superficies': self.chkSuperf.isChecked(),
                'niveles': self.spNiveles.value(),
                'xmin': self._flt(self.edXminCp.text()),
                'xmax': self._flt(self.edXmaxCp.text()),
                'dx':   self._flt(self.edDxCp.text()),
                'ymin': self._flt(self.edYminCp.text()),
                'ymax': self._flt(self.edYmaxCp.text()),
                'dy':   self._flt(self.edDyCp.text()),
            },
            'v_campo': {
                **self._snap_commons(self._commonsVCampo),
                'radio': self._flt(self.edRadioV.text()),
                'mostrar_superficies': self.chkSuperfV.isChecked(),   # NUEVO
                'xmin': self._flt(self.edXminV.text()),
                'xmax': self._flt(self.edXmaxV.text()),
                'dx':   self._flt(self.edDxV.text()),
                'ymin': self._flt(self.edYminV.text()),
                'ymax': self._flt(self.edYmaxV.text()),
                'dy':   self._flt(self.edDyV.text()),
            },
        }
        return cfg

    @staticmethod
    def _autoscale_axes(axes: np.ndarray | List[plt.Axes]):
        try:
            for ax in axes:
                if not ax.lines and not ax.collections:
                    continue
                ax.relim()
                ax.autoscale_view(True, True, True)
                ax.set_xmargin(0.0)
                ax.set_ymargin(0.0)
        except Exception:
            pass
    @staticmethod
    def _apply_default_ylim(ax: plt.Axes, pad_fraction: float = 0.25):
        y0, y1 = ax.get_ylim()
        ymin, ymax = (min(y0, y1), max(y0, y1))
        if ymin >= 0.0 and ymax >= 0.0:
            pad = (ymax - 0.0) * pad_fraction
            ax.set_ylim(bottom=0.0, top=ymax + pad)
        elif ymin <= 0.0 and ymax <= 0.0:
            pad = (0.0 - ymin) * pad_fraction
            ax.set_ylim(bottom=ymin - pad, top=0.0)
        else:
            span = max(1e-15, ymax - ymin)
            pad = span * pad_fraction
            ax.set_ylim(bottom=ymin - pad, top=ymax + pad)

    @staticmethod
    def _apply_x_padding(ax: plt.Axes, frac: float = 0.05):
        x0, x1 = ax.get_xlim()
        xmin, xmax = (min(x0, x1), max(x0, x1))
        span = xmax - xmin
        if span <= 0:
            return
        pad = span * frac
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)
        ax.set_ymargin(0.0)

    @staticmethod
    def _lock_axes(ax: plt.Axes):
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_autoscale_on(False)
        ax.set_xmargin(0.0); ax.set_ymargin(0.0)

    @staticmethod
    def _grid_from_limits(xmin: float, xmax: float, dx: Optional[float],
                          ymin: float, ymax: float, dy: Optional[float]) -> Tuple[np.ndarray, np.ndarray, float, float]:
        if dx is None or dx <= 0.0:
            nx = 300
            dx = (xmax - xmin) / max(1, (nx - 1))
        if dy is None or dy <= 0.0:
            ny = 300
            dy = (ymax - ymin) / max(1, (ny - 1))
        x = np.arange(xmin, xmax + 0.5*dx, dx)
        y = np.arange(ymin, ymax + 0.5*dy, dy)
        return x, y, dx, dy

    def _onMostrar(self):
        if self._result_provider is None: return
        mp2d, meta = self._result_provider()
        data = cast(GraficoConfig, self.cbTipoGrafico.currentData() or {})
        if not data: return

        self._clear_plot_area()
        nombre = self.cbTipoGrafico.currentText()
        cfg: AllGraphsConfig = self._collect_all_graph_configs(nombre, data)

        if int(data.get('pg', -1)) == 1:
            fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharex=False)
            self._draw_coefs(mp2d, meta, cfg, axes)
            self._add_figure(fig)
            return

        if nombre not in ('Cp campo', 'V campo'):
            fig, ax = plt.subplots(1, 1)
            if nombre == 'Paneles': self._draw_paneles(mp2d, meta, cfg, ax)
            elif nombre == 'Cp': self._draw_cp(mp2d, meta, cfg, ax)
            elif nombre == 'Cp vectorial': self._draw_cp_vectorial(mp2d, meta, cfg, ax)
            elif nombre == 'V': self._draw_v(mp2d, meta, cfg, ax)
            self._add_figure(fig)
            return

        fig, ax = plt.subplots(1, 1)
        canvas = self._add_figure_return_canvas(fig)

        if nombre == 'Cp campo':
            manual = self._limits_from_fields_cp()
            if manual is not None:
                xmin, xmax, ymin, ymax, dx, dy = manual
                x, y, dx2, dy2 = self._grid_from_limits(xmin, xmax, dx, ymin, ymax, dy)
                self._draw_cp_campo_from_xy(mp2d, cfg, ax, x, y)
                ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
                self._lock_axes(ax)
                self._fill_fields_cp_from_limits(xmin, xmax, dx2, ymin, ymax, dy2)
                canvas.draw(); return

            if mp2d is not None and hasattr(mp2d, "plotPaneles"):
                try:
                    mp2d.plotPaneles(it=cfg['cp_campo']['instante'], ax=ax, mostrarRM=cfg['cp_campo']['mostrar_RM'])
                except TypeError:
                    mp2d.plotPaneles(cfg['cp_campo']['instante'], ax)
            else:
                rxy_list: List[Tuple[str, np.ndarray]] = meta.get("rxy_list", [])
                for nombre_s, r_xy in rxy_list:
                    ax.plot(r_xy[0, :], r_xy[1, :], label=nombre_s)
                if cfg['cp_campo']['mostrar_RM']:
                    ax.plot(0, 0, 'gx', markersize=10, label='Rm')
                ax.axis('equal'); ax.grid(True)

            canvas.draw()
            xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim(); ax.cla()
            x, y, dx2, dy2 = self._grid_from_limits(xmin, xmax, cfg['cp_campo']['dx'], ymin, ymax, cfg['cp_campo']['dy'])
            self._draw_cp_campo_from_xy(mp2d, cfg, ax, x, y)
            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
            self._lock_axes(ax); self._fill_fields_cp_from_limits(xmin, xmax, dx2, ymin, ymax, dy2)
            canvas.draw(); return

        if nombre == 'V campo':
            manual = self._limits_from_fields_v()
            if manual is not None:
                xmin, xmax, ymin, ymax, dx, dy = manual
                x, y, dx2, dy2 = self._grid_from_limits(xmin, xmax, dx, ymin, ymax, dy)
                self._draw_v_campo_from_xy(mp2d, cfg, ax, x, y)
                ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
                self._lock_axes(ax); self._fill_fields_v_from_limits(xmin, xmax, dx2, ymin, ymax, dy2)
                canvas.draw(); return

            if mp2d is not None and hasattr(mp2d, "plotPaneles"):
                try:
                    mp2d.plotPaneles(it=cfg['v_campo']['instante'], ax=ax, mostrarRM=cfg['v_campo']['mostrar_RM'])
                except TypeError:
                    mp2d.plotPaneles(cfg['v_campo']['instante'], ax)
            else:
                rxy_list: List[Tuple[str, np.ndarray]] = meta.get("rxy_list", [])
                for nombre_s, r_xy in rxy_list:
                    ax.plot(r_xy[0, :], r_xy[1, :], label=nombre_s)
                if cfg['v_campo']['mostrar_RM']:
                    ax.plot(0, 0, 'gx', markersize=10, label='Rm')
                ax.axis('equal'); ax.grid(True)

            canvas.draw()
            xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim(); ax.cla()
            x, y, dx2, dy2 = self._grid_from_limits(xmin, xmax, cfg['v_campo']['dx'], ymin, ymax, cfg['v_campo']['dy'])
            self._draw_v_campo_from_xy(mp2d, cfg, ax, x, y)
            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
            self._lock_axes(ax); self._fill_fields_v_from_limits(xmin, xmax, dx2, ymin, ymax, dy2)
            canvas.draw(); return

    def _draw_paneles(self, mp2d: Optional[MP2D], meta: dict, cfg: AllGraphsConfig, ax: plt.Axes):
        if mp2d is not None and hasattr(mp2d, "plotPaneles"):
            try:
                mp2d.plotPaneles(it=cfg['paneles']['instante'], ax=ax, mostrarRM=cfg['paneles']['mostrar_RM']); return
            except TypeError:
                mp2d.plotPaneles(cfg['paneles']['instante'], ax); return
        rxy_list: List[Tuple[str, np.ndarray]] = meta.get("rxy_list", [])
        for nombre, r_xy in rxy_list:
            ax.plot(r_xy[0, :], r_xy[1, :], label=nombre)
        if cfg['paneles']['mostrar_RM']:
            ax.plot(0, 0, 'gx', markersize=10, label='Rm')
        ax.legend(); ax.set_xlabel('x, [m]'); ax.set_ylabel('y, [m]'); ax.axis('equal'); ax.grid(True)

    def _draw_coefs(self, mp2d: Optional[MP2D], meta: dict, cfg: AllGraphsConfig, axes: np.ndarray):
        for ax in axes: ax.grid(True)
        if mp2d is not None and hasattr(mp2d, "plotCAero"):
            try:
                mp2d.plotCAero(ax=axes, total=cfg['coeficientes']['totales']); used_plotCAero = True
            except TypeError:
                try:
                    mp2d.plotCAero(axes); used_plotCAero = True
                except Exception:
                    self._plot_coefs_manual(mp2d, axes, cfg['coeficientes']['totales'])
        elif mp2d is not None:
            self._plot_coefs_manual(mp2d, axes, cfg['coeficientes']['totales'])

        if self._context.get('estacionario', True):
            titles = ("Cd", "Cl", "Cm")
        else:
            titles = ("Cx", "Cy", "Cz")
        axes[0].set_title(titles[0]); axes[1].set_title(titles[1]); axes[2].set_title(titles[2])

        if not self._context.get('estacionario', True) and cfg['coeficientes']['invertir_m']:
            try:
                for l in axes[2].lines:
                    y = np.asarray(l.get_ydata(), dtype=float)
                    l.set_ydata(-y)
            except Exception:
                pass

        self._autoscale_axes(axes)
        for ax in axes:
            self._apply_x_padding(ax, frac=0.05)
            self._apply_default_ylim(ax, pad_fraction=0.25)

    def _plot_coefs_manual(self, mp2d: MP2D, axes: np.ndarray, totales: bool):
        x=Cx=Cy=Cm=None
        try:
            data = mp2d.CAero() if callable(getattr(mp2d,"CAero",None)) else getattr(mp2d,"CAero",None)
        except Exception:
            data=None
        if isinstance(data, dict):
            x = data.get("alpha_deg") or data.get("t") or data.get("x")
            Cx = data.get("Cx"); Cy = data.get("Cy"); Cm = data.get("Cm")
            if totales and isinstance(data.get("totales"), dict):
                Cx = data["totales"].get("Cx", Cx); Cy = data["totales"].get("Cy", Cy); Cm = data["totales"].get("Cm", Cm)
        if x is None: x = getattr(mp2d,"alpha_deg",None) or getattr(mp2d,"t",None)
        if Cx is None: Cx = getattr(mp2d,"CX",None) or getattr(mp2d,"Cx",None)
        if Cy is None: Cy = getattr(mp2d,"CY",None) or getattr(mp2d,"Cy",None) or getattr(mp2d,"CL",None)
        if Cm is None: Cm = getattr(mp2d,"CM",None) or getattr(mp2d,"Cm",None)
        if x is None or Cx is None or Cy is None or Cm is None: return
        axes[0].plot(x, Cx, label="Cx"); axes[1].plot(x, Cy, label="Cy"); axes[2].plot(x, Cm, label="Cm")
        for ax in axes: ax.legend(loc="best")
        self._autoscale_axes(axes)

    def _grid_from_cfg(self, cfg: dict, meta: dict) -> Tuple[np.ndarray, np.ndarray]:
        xmin=cfg.get('xmin'); xmax=cfg.get('xmax'); dx=cfg.get('dx')
        ymin=cfg.get('ymin'); ymax=cfg.get('ymax'); dy=cfg.get('dy')
        if xmin is None or xmax is None or ymin is None or ymax is None:
            bbox = _bbox_from_rxy_list(meta.get("rxy_list", []))
            if bbox is None:
                x = np.linspace(-1.0, 1.0, 300); y = np.linspace(-1.0, 1.0, 300); return x, y
            bxmin, bxmax, bymin, bymax = bbox
            dxw = (bxmax - bxmin) or 1.0; dyw = (bymax - bymin) or 1.0
            pad_x = 0.1*dxw; pad_y = 0.1*dyw
            xmin = bxmin - pad_x; xmax = bxmax + pad_x; ymin = bymin - pad_y; ymax = bymax + pad_y
        if dx is None or dx <= 0:
            nx = 300; dx = (xmax - xmin)/max(1, (nx-1))
        if dy is None or dy <= 0:
            ny = 300; dy = (ymax - ymin)/max(1, (ny-1))
        x = np.arange(xmin, xmax + 0.5*dx, dx); y = np.arange(ymin, ymax + 0.5*dy, dy)
        return x, y

    def _draw_cp(self, mp2d: Optional[MP2D], meta: dict, cfg: AllGraphsConfig, ax: plt.Axes):
        if mp2d is not None and hasattr(mp2d, "plotCp"):
            try: mp2d.plotCp(cfg['cp']['instante'], ax); return
            except TypeError: mp2d.plotCp(it=cfg['cp']['instante'], ax=ax); return
        ax.set_title("Cp"); ax.set_xlabel("x"); ax.set_ylabel("Cp"); ax.grid(True)

    def _draw_cp_vectorial(self, mp2d: Optional[MP2D], meta: dict, cfg: AllGraphsConfig, ax: plt.Axes):
        if mp2d is not None and hasattr(mp2d, "plotCpVect"):
            try: mp2d.plotCpVect(cfg['cp_vectorial']['instante'], ax, cfg['cp_vectorial']['escala']); return
            except TypeError: mp2d.plotCpVect(it=cfg['cp_vectorial']['instante'], ax=ax, escala=cfg['cp_vectorial']['escala']); return
        ax.set_title('Cp vectorial'); ax.grid(True)

    def _draw_v(self, mp2d: Optional[MP2D], meta: dict, cfg: AllGraphsConfig, ax: plt.Axes):
        if mp2d is not None and hasattr(mp2d, "plotV"):
            try: mp2d.plotV(cfg['v']['instante'], ax, cfg['v']['escala'], cfg['v']['V_relativa']); return
            except TypeError: mp2d.plotV(it=cfg['v']['instante'], ax=ax, escala=cfg['v']['escala'], v_relativa=cfg['v']['V_relativa']); return
        ax.set_title('V'); ax.grid(True)

    def _draw_cp_campo_from_xy(self, mp2d: Optional[MP2D], cfg: AllGraphsConfig, ax: plt.Axes,
                               x: np.ndarray, y: np.ndarray):
        if mp2d is not None and hasattr(mp2d, "plotCampo_Cp"):
            mp2d.plotCampo_Cp(
                it=cfg['cp_campo']['instante'], x=x, y=y, ax=ax,
                niveles=cfg['cp_campo']['niveles'],
                mostrarPC=cfg['cp_campo']['mostrar_superficies'],
                radius=cfg['cp_campo']['radio']
            ); return
        ax.set_title('Cp campo'); ax.grid(True)

    def _draw_v_campo_from_xy(self, mp2d: Optional[MP2D], cfg: AllGraphsConfig, ax: plt.Axes,
                              x: np.ndarray, y: np.ndarray):
        if mp2d is not None and hasattr(mp2d, "plotCampo_V"):
            mp2d.plotCampo_V(
                it=cfg['v_campo']['instante'], x=x, y=y, ax=ax,
                escala=cfg['v_campo']['escala'],
                VRel=cfg['v_campo']['V_relativa'],
                radius=cfg['v_campo']['radio'],
                mostrarPC=cfg['v_campo']['mostrar_superficies']
            ); return
        ax.set_title('V campo'); ax.grid(True)

    def _limits_from_fields_cp(self) -> Optional[Tuple[float,float,float,float,Optional[float],Optional[float]]]:
        try:
            xmin = float(self.edXminCp.text()); xmax = float(self.edXmaxCp.text())
            ymin = float(self.edYminCp.text()); ymax = float(self.edYmaxCp.text())
            dx = float(self.edDxCp.text()) if self.edDxCp.text().strip() else None
            dy = float(self.edDyCp.text()) if self.edDyCp.text().strip() else None
            return xmin, xmax, ymin, ymax, dx, dy
        except Exception:
            return None

    def _limits_from_fields_v(self) -> Optional[Tuple[float,float,float,float,Optional[float],Optional[float]]]:
        try:
            xmin = float(self.edXminV.text()); xmax = float(self.edXmaxV.text())
            ymin = float(self.edYminV.text()); ymax = float(self.edYmaxV.text())
            dx = float(self.edDxV.text()) if self.edDxV.text().strip() else None
            dy = float(self.edDyV.text()) if self.edDyV.text().strip() else None
            return xmin, xmax, ymin, ymax, dx, dy
        except Exception:
            return None

    def _fill_fields_cp_from_limits(self, xmin: float, xmax: float, dx: float,
                                    ymin: float, ymax: float, dy: float):
        self.edXminCp.setText(f"{xmin}"); self.edXmaxCp.setText(f"{xmax}"); self.edDxCp.setText(f"{dx}")
        self.edYminCp.setText(f"{ymin}"); self.edYmaxCp.setText(f"{ymax}"); self.edDyCp.setText(f"{dy}")

    def _fill_fields_v_from_limits(self, xmin: float, xmax: float, dx: float,
                                   ymin: float, ymax: float, dy: float):
        self.edXminV.setText(f"{xmin}"); self.edXmaxV.setText(f"{xmax}"); self.edDxV.setText(f"{dx}")
        self.edYminV.setText(f"{ymin}"); self.edYmaxV.setText(f"{ymax}"); self.edDyV.setText(f"{dy}")

    def apply_all_graph_configs(self, cfg: AllGraphsConfig):
        sel = cfg.get("seleccion", {}) or {}
        nombre_objetivo = sel.get("tipo", "")
        pg_objetivo = int(sel.get("pg", 0))
        idx_encontrado = -1
        for i in range(self.cbTipoGrafico.count()):
            if self.cbTipoGrafico.itemText(i) == nombre_objetivo:
                idx_encontrado = i
                break
        if idx_encontrado < 0:
            for i in range(self.cbTipoGrafico.count()):
                data = self.cbTipoGrafico.itemData(i) or {}
                if int(data.get("pg", -1)) == pg_objetivo:
                    idx_encontrado = i
                    break
        if idx_encontrado >= 0:
            self.cbTipoGrafico.setCurrentIndex(idx_encontrado)
            self._onTipoGraficoChanged(idx_encontrado)

        def _first_idx():
            paths = [
                ('paneles', self._commonsPaneles),
                ('cp', self._commonsCp),
                ('v', self._commonsV),
                ('cp_campo', self._commonsCpCampo),
                ('v_campo', self._commonsVCampo),
                ('coeficientes', self._coefBox),
                ('cp_vectorial', self._commonsCpVect),
            ]
            for key, _ in paths:
                d = cfg.get(key) or {}
                if "instante" in d:
                    try:
                        return int(d.get("instante", 0))
                    except Exception:
                        pass
            return None

        idx_unico = _first_idx()
        if idx_unico is None:
            idx_unico = 0

        def _fill_commons(box, data):
            if not data: data = {}
            try:
                box.setIndex(int(data.get("instante", idx_unico)))
            except Exception:
                box.setIndex(idx_unico)

        _fill_commons(self._commonsPaneles, cfg.get("paneles"))
        _fill_commons(self._commonsCp, cfg.get("cp"))
        _fill_commons(self._commonsCpVect, cfg.get("cp_vectorial"))
        _fill_commons(self._commonsV, cfg.get("v"))

        coefs = cfg.get("coeficientes") or {}
        self._coefBox.chkInvertM.setChecked(bool(coefs.get("invertir_m", False)))
        self._coefBox.chkTot.setChecked(bool(coefs.get("totales", False)))

        # ---- Cp campo
        cpc = cfg.get("cp_campo") or {}
        _fill_commons(self._commonsCpCampo, cpc)

        def _put(line: QLineEdit, key: str, src: dict):
            val = src.get(key, None)
            line.setText("" if val in (None, "") else f"{val}")

        _put(self.edRadioCp, "radio", cpc)
        _put(self.edXminCp, "xmin", cpc)
        _put(self.edXmaxCp, "xmax", cpc)
        _put(self.edDxCp,   "dx",   cpc)
        _put(self.edYminCp, "ymin", cpc)
        _put(self.edYmaxCp, "ymax", cpc)
        _put(self.edDyCp,   "dy",   cpc)

        if "mostrar_superficies" in cpc:
            self.chkSuperf.setChecked(bool(cpc.get("mostrar_superficies", True)))

        if "niveles" in cpc and cpc.get("niveles") not in (None, ""):
            try:
                self.spNiveles.setValue(int(cpc.get("niveles")))
            except Exception:
                pass
        else:
            self.spNiveles.setValue(self.spNiveles.minimum())

        # ---- V campo
        vc = cfg.get("v_campo") or {}
        _fill_commons(self._commonsVCampo, vc)

        _put(self.edRadioV, "radio", vc)
        _put(self.edXminV,  "xmin", vc)
        _put(self.edXmaxV,  "xmax", vc)
        _put(self.edDxV,    "dx",   vc)
        _put(self.edYminV,  "ymin", vc)
        _put(self.edYmaxV,  "ymax", vc)
        _put(self.edDyV,    "dy",   vc)

        if "mostrar_superficies" in vc and hasattr(self, "chkSuperfV"):
            self.chkSuperfV.setChecked(bool(vc.get("mostrar_superficies", True)))

        if "niveles" in vc and vc.get("niveles") not in (None, "") and hasattr(self, "spNivelesV"):
            try:
                self.spNivelesV.setValue(int(vc.get("niveles")))
            except Exception:
                pass
        elif hasattr(self, "spNivelesV"):
            self.spNivelesV.setValue(self.spNivelesV.minimum())

        for ed in (self.edXminCp, self.edXmaxCp, self.edDxCp, self.edYminCp, self.edYmaxCp, self.edDyCp):
            if ed.text().strip() == "":
                ed.clear()
        for ed in (self.edXminV, self.edXmaxV, self.edDxV, self.edYminV, self.edYmaxV, self.edDyV):
            ed.clear()


    # ---- Acciones de "Reiniciar grilla" ----
    def _onResetGridCp(self):
        for ed in (self.edXminCp, self.edXmaxCp, self.edDxCp, self.edYminCp, self.edYmaxCp, self.edDyCp):
            ed.clear()

    def _onResetGridV(self):
        for ed in (self.edXminV, self.edXmaxV, self.edDxV, self.edYminV, self.edYmaxV, self.edDyV):
            ed.clear()

    def _get_mp_t_value(self, idx: int) -> Optional[float]:
        if self._result_provider is None:
            return None
        try:
            mp2d, _ = self._result_provider()
            if mp2d is None:
                return None
            t = getattr(mp2d, "t", None)
            if t is None:
                return None
            if idx < 0 or idx >= len(t):
                return None
            return float(t[idx])
        except Exception:
            return None
        
    def _wire_shared_index(self):
        boxes = [
            self._commonsPaneles, self._coefBox, self._commonsCp,
            self._commonsCpVect, self._commonsV, self._commonsCpCampo, self._commonsVCampo
        ]
        for b in boxes:
            try:
                b.indiceChanged.connect(self._on_box_index_changed)
            except Exception:
                pass

    def _on_box_index_changed(self, v: int):
        if self._broadcasting:
            return
        self._shared_index = int(v)
        self._broadcast_shared_index()

    def _broadcast_shared_index(self):
        self._broadcasting = True
        try:
            boxes = [
                self._commonsPaneles, self._coefBox, self._commonsCp,
                self._commonsCpVect, self._commonsV, self._commonsCpCampo, self._commonsVCampo
            ]
            for b in boxes:
                try:
                    b.setIndex(self._shared_index)
                except Exception:
                    pass
        finally:
            self._broadcasting = False

class GuardadoTab(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # -----------------------------
        # Bloque: Cargar sesión
        # -----------------------------
        gbLoad = QGroupBox("Cargar sesión")
        gbLoadLayout = QVBoxLayout(gbLoad)

        rowLoad = QHBoxLayout()
        self.btnElegirLoad = QPushButton("Elegir archivo…")
        self.edRutaLoad = QLineEdit()
        self.edRutaLoad.setReadOnly(True)
        self.edRutaLoad.setPlaceholderText("Ningún archivo seleccionado")
        rowLoad.addWidget(self.btnElegirLoad)
        rowLoad.addWidget(self.edRutaLoad, 1)
        gbLoadLayout.addLayout(rowLoad)

        self.btnCargar = QPushButton("Cargar sesión")
        gbLoadLayout.addWidget(self.btnCargar)

        layout.addWidget(gbLoad)

        # -----------------------------
        # Bloque: Guardar sesión (1 archivo)
        # -----------------------------
        gbSim = QGroupBox("Guardar sesión")
        gbSimLayout = QVBoxLayout(gbSim)

        rowSim = QHBoxLayout()
        self.btnElegirSim = QPushButton("Elegir ubicación…")
        self.edRutaSim = QLineEdit()
        self.edRutaSim.setReadOnly(True)
        self.edRutaSim.setPlaceholderText("Ningún archivo seleccionado")
        rowSim.addWidget(self.btnElegirSim)
        rowSim.addWidget(self.edRutaSim, 1)
        gbSimLayout.addLayout(rowSim)

        self.btnGuardarSim = QPushButton("Guardar sesión")
        gbSimLayout.addWidget(self.btnGuardarSim)

        layout.addWidget(gbSim)

        # -----------------------------
        # Bloque: Guardar coeficientes
        # -----------------------------
        gbCoef = QGroupBox("Guardar coeficientes")
        gbCoefLayout = QVBoxLayout(gbCoef)

        rowCsv = QHBoxLayout()
        self.btnElegirCoef = QPushButton("Elegir CSV…")
        self.edRutaCoef = QLineEdit()
        self.edRutaCoef.setReadOnly(True)
        self.edRutaCoef.setPlaceholderText("Ningún archivo seleccionado")
        rowCsv.addWidget(self.btnElegirCoef)
        rowCsv.addWidget(self.edRutaCoef, 1)
        gbCoefLayout.addLayout(rowCsv)

        self.btnGuardarCoef = QPushButton("Guardar coeficientes (.csv)")
        gbCoefLayout.addWidget(self.btnGuardarCoef)

        layout.addWidget(gbCoef)
        layout.addStretch(1)

        # Conexiones
        self.btnElegirLoad.clicked.connect(self._onElegirRutaLoadSesion)
        self.btnCargar.clicked.connect(self._onCargarSesion)

        self.btnElegirSim.clicked.connect(self._onElegirRutaSesion)
        self.btnGuardarSim.clicked.connect(self._onGuardarSesion)

        self.btnElegirCoef.clicked.connect(self._onElegirRutaCoef)
        self.btnGuardarCoef.clicked.connect(self._onGuardarCoef)

    # ==========================
    #  UTILITARIOS/ACCESOS
    # ==========================
    def _main_window(self):
        return self.window()

    def _tabs(self):
        win = self._main_window()
        if not win:
            return None, None, None
        sol = getattr(win, 'tabSolidos', None)
        sim = getattr(win, 'tabSim', None)
        res = getattr(win, 'tabRes', None)
        return sol, sim, res

    # ==========================
    #  SNAPSHOT (para guardar)
    # ==========================
    def _snapshot_sesion(self) -> dict:
        sol_tab, sim_tab, res_tab = self._tabs()

        sesion = {"app_state": self.state.to_dict()}

        # --- Simulación UI ---
        sim_dict: Dict[str, object] = {}
        if sim_tab is not None:
            sim_dict["referencias"] = {
                "l":   float(sim_tab.edL.text()    or f"{self.state.l_ref}"),
                "V":   float(sim_tab.edVref.text() or f"{self.state.V_ref}"),
                "rho": float(sim_tab.edRho.text()  or f"{self.state.rho_ref}"),
                "nu":  float(sim_tab.edNu.text()   or "1.5e-5"),
                "a":   float(sim_tab.edA.text()    or "340.3"),
                "g":   float(sim_tab.edG.text()    or "9.81"),
            }
            sim_dict["modo_index"] = int(sim_tab.cbModo.currentIndex())
            sim_dict["modo"] = "estacionario" if sim_tab.cbModo.currentIndex() == 0 else "no_estacionario"

            sim_dict["estacionario"] = {
                "V": sim_tab.est_V.text(),
                "alfa_i": sim_tab.est_alfa_i.text(),
                "alfa_f": sim_tab.est_alfa_f.text(),
                "delta_alfa": sim_tab.est_delta_alfa.text(),
            }

            sim_dict["no_est_submodo_index"] = int(sim_tab.cbCinematicaNoEst.currentIndex())
            sim_dict["no_est_MRU"] = {
                "V": sim_tab.ne_mru_V.text(),
                "alfa": sim_tab.ne_mru_alfa.text(),
                "t_final": sim_tab.ne_mru_tfinal.text(),
                "dt": sim_tab.ne_mru_dt.text(),
            }
            sim_dict["no_est_rotacion"] = {
                "V": sim_tab.ne_rot_V.text(),
                "alfa0": sim_tab.ne_rot_alfa0.text(),
                "w": sim_tab.ne_rot_w.text(),
                "t_final": sim_tab.ne_rot_tfinal.text(),
                "dt": sim_tab.ne_rot_dt.text(),
            }
            sim_dict["no_est_traslacion"] = {
                "V": sim_tab.ne_tras_V.text(),
                "h0": sim_tab.ne_tras_h0.text(),
                "w": sim_tab.ne_tras_w.text(),
                "t_final": sim_tab.ne_tras_tfinal.text(),
                "dt": sim_tab.ne_tras_dt.text(),
            }
            rutas_por_sid = dict(getattr(sim_tab, "_ne_archivo_paths", {}) or {})
            rm_path = getattr(sim_tab, "_ne_rm_file", None)
            try:
                rel_flags = sim_tab.get_relativizar_flags()
            except Exception:
                rel_flags = {}
            sim_dict["no_est_archivo"] = {
                "rm_archivo_path": rm_path,
                "archivos_por_solido": rutas_por_sid,
                "relativizar_por_solido": rel_flags,
            }
        sesion["simulacion_ui"] = sim_dict

        # --- Resultados UI ---
        res_dict: Dict[str, object] = {}
        if res_tab is not None:
            try:
                nombre_actual = str(res_tab.cbTipoGrafico.currentText())
                data_actual = res_tab.cbTipoGrafico.currentData() or {}
                res_dict["seleccion_nombre"] = nombre_actual
                res_dict["config_graficos"] = res_tab._collect_all_graph_configs(nombre_actual, data_actual)
            except Exception:
                res_dict["seleccion_nombre"] = None
                res_dict["config_graficos"] = None
        sesion["resultados_ui"] = res_dict

        return sesion

    # ==========================
    #  GUARDAR SESIÓN (.pkl)
    # ==========================
    def _onElegirRutaSesion(self):
        ruta, _ = QFileDialog.getSaveFileName(
            self, "Guardar sesión como…", "", "Sesión MP2D (*.pkl);;Todos los archivos (*)"
        )
        if ruta:
            if not ruta.lower().endswith(".pkl"):
                ruta += ".pkl"
            self.edRutaSim.setText(ruta)

    def _onGuardarSesion(self):
        if not self.edRutaSim.text():
            self._onElegirRutaSesion()
        ruta = (self.edRutaSim.text() or "").strip()
        if not ruta:
            return
        if not ruta.lower().endswith(".pkl"):
            ruta += ".pkl"
            self.edRutaSim.setText(ruta)

        sesion = self._snapshot_sesion()
        mp = getattr(self.state, 'mp2d', None)

        try:
            if mp is not None:
                # Adjuntar snapshot de UI al objeto MP2D antes de serializar
                try:
                    setattr(mp, "sesion_ui", sesion)
                except Exception:
                    pass

                base, _ = os.path.splitext(ruta)
                try:
                    mp.guardar(base)  # crea base+'.pkl'
                    ruta_final = base + ".pkl"
                except Exception:
                    with open(ruta, "wb") as f:
                        pickle.dump(mp, f, protocol=pickle.HIGHEST_PROTOCOL)
                    ruta_final = ruta

                QMessageBox.information(
                    self, "Guardar sesión",
                    f"Sesión guardada con éxito en:\n{ruta_final}\n\n"
                    "Contenido: simulación MP2D + configuración completa de la UI."
                )
                print(f"[Guardado] Sesión (MP2D+UI) -> {ruta_final}")

            else:
                # Sin simulación: guardar sólo la UI
                contenedor = {"_tipo": "solo_ui", "sesion_ui": sesion}
                with open(ruta, "wb") as f:
                    pickle.dump(contenedor, f, protocol=pickle.HIGHEST_PROTOCOL)

                QMessageBox.information(
                    self, "Guardar sesión",
                    f"No había simulación activa.\nSe guardó la sesión (UI) en:\n{ruta}"
                )
                print(f"[Guardado] Sesión (solo UI) -> {ruta}")

        except Exception as e:
            traceback.print_exc()
            QMessageBox.warning(self, "Guardar sesión", f"No se pudo completar el guardado:\n{e}")

    # ==========================
    #  CARGAR SESIÓN (.pkl)
    # ==========================
    def _onElegirRutaLoadSesion(self):
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Cargar sesión", "", "Sesión MP2D (*.pkl);;Todos los archivos (*)"
        )
        if ruta:
            self.edRutaLoad.setText(ruta)

    def _onCargarSesion(self):
        if not self.edRutaLoad.text():
            self._onElegirRutaLoadSesion()
        ruta = (self.edRutaLoad.text() or "").strip()
        if not ruta:
            return

        try:
            with open(ruta, "rb") as f:
                obj = pickle.load(f)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.warning(self, "Cargar sesión", f"No se pudo abrir el archivo:\n{e}")
            return

        mp_loaded = None
        sesion = None
        try:
            if hasattr(obj, "sesion_ui"):
                mp_loaded = obj
                sesion = getattr(obj, "sesion_ui", None)
            elif isinstance(obj, dict) and ("sesion_ui" in obj):
                sesion = obj.get("sesion_ui")
            else:
                raise ValueError("El archivo no contiene datos de sesión reconocibles.")
        except Exception:
            traceback.print_exc()
            QMessageBox.warning(self, "Cargar sesión", "Formato de sesión no reconocido.")
            return

        if not isinstance(sesion, dict):
            QMessageBox.warning(self, "Cargar sesión", "La sesión cargada no tiene el formato esperado.")
            return

        try:
            self._apply_sesion_ui(sesion, mp_loaded)
            QMessageBox.information(self, "Cargar sesión", "Sesión restaurada correctamente.")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.warning(self, "Cargar sesión", f"Error al aplicar la sesión:\n{e}")

    # --------------------------
    #  Aplicar sesión cargada
    # --------------------------
    def _apply_sesion_ui(self, sesion: dict, mp_loaded):
        sol_tab, sim_tab, res_tab = self._tabs()

        # --- 1) Restaurar AppState ---
        app_state = sesion.get("app_state", {}) or {}
        ref = (app_state.get("referencia", {}) or {})
        self.state.l_ref = float(ref.get("l", self.state.l_ref))
        self.state.V_ref = float(ref.get("V", self.state.V_ref))
        self.state.rho_ref = float(ref.get("rho", self.state.rho_ref))

        # Conjunto
        cj = (app_state.get("conjunto", {}) or {})
        self.state.conjunto = ConjuntoConfig(
            theta1_deg=float(cj.get("theta1_deg", 0.0)),
            dx=float(cj.get("dx", 0.0)),
            dy=float(cj.get("dy", 0.0)),
            theta2_deg=float(cj.get("theta2_deg", 0.0)),
        )

        # Sólidos
        self.state.solidos.clear()
        faltantes_perfil: list[str] = []
        for s in (app_state.get("solidos", []) or []):
            flap = s.get("flap") or None
            flap_cfg = None
            if flap is not None:
                flap_cfg = FlapConfig(
                    cf=float(flap.get("cf", 0.0)),
                    df=float(flap.get("df", 0.0)),
                    h_TEw_ROTf=float(flap.get("h_TEw_ROTf", 0.0)),
                    v_TEw_ROTf=float(flap.get("v_TEw_ROTf", 0.0)),
                    h_ROTf_BAf=float(flap.get("h_ROTf_BAf", 0.0)),
                    v_ROTf_MCf=float(flap.get("v_ROTf_MCf", 0.0)),
                )
            sc = SolidConfig(
                id=str(s.get("id")),
                nombre=str(s.get("nombre")),
                perfil_path=s.get("perfil_path"),
                formato=str(s.get("formato", "selig")),
                c=float(s.get("c", 0.0)),
                theta1_deg=float(s.get("theta1_deg", 0.0)),
                dx=float(s.get("dx", 0.0)),
                dy=float(s.get("dy", 0.0)),
                theta2_deg=float(s.get("theta2_deg", 0.0)),
                n_intra=int(s.get("n_intra", 0)),
                n_extra=int(s.get("n_extra", 0)),
                es_flap=bool(s.get("es_flap", False)),
                flap=flap_cfg,
                cerrar_te=bool(s.get("cerrar_te", False)),
                perfil_missing=False,  # se ajusta abajo si falta
            )
            if sc.perfil_path and (not os.path.exists(sc.perfil_path)):
                sc.perfil_missing = True
                faltantes_perfil.append(f"- {sc.nombre}: {sc.perfil_path}")
            self.state.solidos[sc.id] = sc

        if sol_tab is not None:
            sol_tab.reload_from_state()

        # --- 2) Restaurar campos de Simulación ---
        sim_ui = sesion.get("simulacion_ui", {}) or {}
        if sim_tab is not None:
            sim_tab.apply_sim_ui(sim_ui)

        # --- 3) MP2D (si vino) y CONTEXTO de Resultados ---
        self.state.mp2d = mp_loaded if mp_loaded is not None else None

        if sim_tab is not None:
            sim_tab.update_adim_info_from_mp(self.state.mp2d)

        estacionario = (sim_ui.get("modo", "estacionario") == "estacionario")
        inst_count = 1
        if self.state.mp2d is not None and hasattr(self.state.mp2d, "t"):
            try:
                inst_count = len(getattr(self.state.mp2d, "t", []))
            except Exception:
                inst_count = 1

        if res_tab is not None:
            res_tab.setContext(estacionario, inst_count)

        # --- 4) Re-aplicar configuración de visualizaciones (luego del contexto) ---
        res_ui = sesion.get("resultados_ui", {}) or {}
        cfg_all = res_ui.get("config_graficos", None)
        if res_tab is not None and cfg_all:
            res_tab.apply_all_graph_configs(cfg_all)

        # --- 5) Advertencias únicas por faltantes (perfiles y cinemáticas "archivo") ---
        #    a) perfiles ya colectados en 'faltantes_perfil'
        faltantes_archivo_noest: list[str] = []
        try:
            arch = sim_ui.get("no_est_archivo", {}) or {}
            rm_path = arch.get("rm_archivo_path", None)
            if rm_path and (not os.path.exists(rm_path)):
                faltantes_archivo_noest.append(f"- RM: {rm_path}")

            rutas_por_sid = arch.get("archivos_por_solido", {}) or {}
            for sid, ruta in rutas_por_sid.items():
                if ruta and (not os.path.exists(ruta)):
                    nombre = self.state.solidos.get(sid).nombre if sid in self.state.solidos else f"Sólido (id {sid})"
                    faltantes_archivo_noest.append(f"- {nombre}: {ruta}")
        except Exception:
            pass

        # Mostrar una única advertencia, si corresponde
        if faltantes_perfil or faltantes_archivo_noest:
            partes = []
            if faltantes_perfil:
                partes.append("No se encontraron los archivos de coordenadas de los siguientes sólidos (se restauraron igualmente):\n" +
                            "\n".join(faltantes_perfil))
            if faltantes_archivo_noest:
                partes.append("No se encontraron archivos de cinemática (modo No estacionario - \"Desde archivo de texto\") en:\n" +
                            "\n".join(faltantes_archivo_noest))
            texto = "\n\n".join(partes)
            QMessageBox.warning(self, "Advertencia de sesión cargada", texto)



    # ==========================
    #  GUARDAR COEFICIENTES
    # ==========================
    def _onElegirRutaCoef(self):
        ruta, _ = QFileDialog.getSaveFileName(
            self, "Guardar coeficientes como…", "", "CSV (*.csv);;Todos los archivos (*)"
        )
        if ruta:
            self.edRutaCoef.setText(ruta)

    def _onGuardarCoef(self):
        if not self.edRutaCoef.text():
            self._onElegirRutaCoef()
        ruta = self.edRutaCoef.text().strip()
        if not ruta:
            return

        mp = getattr(self.state, 'mp2d', None)
        if mp is None:
            QMessageBox.information(self, "Guardar coeficientes",
                                    "No hay una simulación disponible para exportar.")
            return

        if not ruta.lower().endswith('.csv'):
            ruta = ruta + '.csv'
            self.edRutaCoef.setText(ruta)

        try:
            ruta_out = mp.guardar_coeficientes_csv(
                ruta, sep=',', precision=10, encoding='utf-8-sig'
            )

            try:
                estacionario = bool(getattr(mp, "mpConfig", {}).get("estacionario", True))
                self._postprocess_coef_csv_headers(ruta_out, estacionario)
            except Exception:
                pass

            QMessageBox.information(
                self, "Guardar coeficientes",
                f"Coeficientes guardados en:\n{ruta_out}"
            )
            print(f"[Guardado] Coeficientes -> {ruta_out}")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.warning(
                self, "Guardar coeficientes",
                f"No se pudo guardar el CSV:\n{e}"
            )

    def _postprocess_coef_csv_headers(self, ruta_csv: str, estacionario: bool):
        if (ruta_csv or "").strip() == "" or (not os.path.exists(ruta_csv)):
            return

        with open(ruta_csv, "r", encoding="utf-8-sig", errors="ignore") as f:
            contenido = f.read()

        if not contenido:
            return

        nl_pos = contenido.find("\n")
        if nl_pos == -1:
            header = contenido
            resto = ""
        else:
            header = contenido[:nl_pos]
            resto = contenido[nl_pos+1:]

        cols = header.split(",")

        def repl_est(c: str) -> str:
            if c.startswith("Cx"):
                return "Cd" + c[2:]
            if c.startswith("Cy"):
                return "Cl" + c[2:]
            return c

        def repl_noest(c: str) -> str:
            if c.startswith("Cm"):
                return "Cz" + c[2:]
            return c

        cols_out = [ (repl_est(col) if estacionario else repl_noest(col)) for col in cols ]
        header_out = ",".join(cols_out)

        with open(ruta_csv, "w", encoding="utf-8-sig", errors="ignore") as f:
            if resto:
                f.write(header_out + "\n" + resto)
            else:
                f.write(header_out)

class SolidosTab(QWidget):
    solidsChanged = Signal()
    _re_solido = re.compile(r"^\s*Sólido\s+(\d+)\s*$", re.IGNORECASE)

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._editing_id: Optional[str] = None
        self._editing_original_name: Optional[str] = None

        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # ----- Columna izquierda con secciones scrollables -----
        izq = QWidget(); izqLayout = QVBoxLayout(izq)
        izqLayout.setContentsMargins(0, 0, 0, 0)
        izqLayout.setSpacing(8)

        # ======================================================
        # Sección PERFIL
        # ======================================================
        perfilContainer = QWidget(); perfilLayout = QVBoxLayout(perfilContainer)
        perfilLayout.setContentsMargins(0, 0, 0, 0)
        perfilLayout.setSpacing(8)

        perfilLayout.addWidget(QLabel('Perfil:'))
        row = QHBoxLayout()
        self.btnCargarPerfil = QPushButton("Cargar perfil…")
        self.edPerfilPath = QLineEdit(); self.edPerfilPath.setReadOnly(True)
        self.edPerfilPath.setPlaceholderText("Ningún archivo seleccionado")
        row.addWidget(self.btnCargarPerfil); row.addWidget(self.edPerfilPath, 1)
        perfilLayout.addLayout(row)

        formatoRow = QHBoxLayout()
        lblFormato = QLabel("Formato:")
        self.cbFormato = QComboBox(); self.cbFormato.addItems(["Selig", "Lednicer"]); self.cbFormato.setCurrentIndex(0)
        formatoRow.addWidget(lblFormato); formatoRow.addWidget(self.cbFormato, 1)
        perfilLayout.addLayout(formatoRow)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)
        self.nombreInput = QLineEdit(); form.addRow(QLabel('Nombre:'), self.nombreInput)
        self.cInput = QLineEdit(); self.cInput.setText("0.508")
        self.theta1Input = QLineEdit(); self.theta1Input.setText("0")
        self.xInput = QLineEdit(); self.xInput.setText("0")
        self.yInput = QLineEdit(); self.yInput.setText("0")
        self.theta2Input = QLineEdit(); self.theta2Input.setText("0")
        self.nIntraInput = QLineEdit(); self.nIntraInput.setText("200")
        self.nExtraInput = QLineEdit(); self.nExtraInput.setText("200")
        form.addRow(QLabel('Cuerda (cw) [m]:'), self.cInput)
        form.addRow(QLabel('Primera rotación [°]:'), self.theta1Input)
        form.addRow(QLabel('Desplazamiento x [m]:'), self.xInput)
        form.addRow(QLabel('Desplazamiento y [m]:'), self.yInput)
        form.addRow(QLabel('Segunda rotación [°]:'), self.theta2Input)
        form.addRow(QLabel('Nro. paneles intradós [ ]:'), self.nIntraInput)
        form.addRow(QLabel('Nro. paneles extradós [ ]:'), self.nExtraInput)
        perfilLayout.addLayout(form)

        self.esFlap = QCheckBox('Es flap'); self.esFlap.setChecked(False)
        perfilLayout.addWidget(self.esFlap)

        perfilScroll = QScrollArea(); perfilScroll.setWidgetResizable(True); perfilScroll.setFrameShape(QFrame.NoFrame)
        perfilScroll.setWidget(perfilContainer)
        izqLayout.addWidget(perfilScroll)

        # ======================================================
        # Sección FLAP
        # ======================================================
        flapInner = QWidget(); flapForm = QFormLayout(flapInner)
        flapForm.setContentsMargins(0, 0, 0, 0)
        flapForm.setSpacing(6)

        self.chkFlapCenterOrigin = QCheckBox("Posicionar punto de abisagramiento en el origen"); self.chkFlapCenterOrigin.setChecked(True); flapForm.addRow(self.chkFlapCenterOrigin)
        self.cfInput = QLineEdit(); _num(self.cfInput); self.cfInput.setText("0.1016")
        self.dfInput = QLineEdit(); _num(self.dfInput); self.dfInput.setText("10.0")
        self.h_TEw_ROTf_Input = QLineEdit(); _num(self.h_TEw_ROTf_Input); self.h_TEw_ROTf_Input.setText("0.03")
        self.v_TEw_ROTf_Input = QLineEdit(); _num(self.v_TEw_ROTf_Input); self.v_TEw_ROTf_Input.setText("0.054")
        self.h_ROTf_BAf_Input = QLineEdit(); _num(self.h_ROTf_BAf_Input); self.h_ROTf_BAf_Input.setText("0.24")
        self.v_ROTf_MCf_Input = QLineEdit(); _num(self.v_ROTf_MCf_Input); self.v_ROTf_MCf_Input.setText("0.1")
        flapForm.addRow('cf [m]:', self.cfInput); flapForm.addRow('df [°]:', self.dfInput)
        flapForm.addRow('h_TEw_ROTf [m/cw]:', self.h_TEw_ROTf_Input); flapForm.addRow('v_TEw_ROTf [m/cw]:', self.v_TEw_ROTf_Input)
        flapForm.addRow('h_ROTf_BAf [m/cf]:', self.h_ROTf_BAf_Input); flapForm.addRow('v_ROTf_MCf [m/cf]:', self.v_ROTf_MCf_Input)

        flapBox = QGroupBox('Flap'); flapBoxLayout = QVBoxLayout(flapBox); flapBoxLayout.setContentsMargins(8, 8, 8, 8)
        flapBoxLayout.addWidget(flapInner)

        flapScroll = QScrollArea(); flapScroll.setWidgetResizable(True); flapScroll.setFrameShape(QFrame.NoFrame)
        flapScroll.setWidget(flapBox)

        self.flapBoxScroll = flapScroll
        self.flapBoxScroll.setVisible(False)
        izqLayout.addWidget(self.flapBoxScroll)

        # ======================================================
        # Sección CONJUNTO
        # ======================================================
        linea2 = QFrame(); linea2.setFrameShape(QFrame.HLine); linea2.setFrameShadow(QFrame.Sunken)
        izqLayout.addWidget(linea2)

        conjuntoContainer = QWidget()
        conjuntoVBox = QVBoxLayout(conjuntoContainer)
        conjuntoVBox.setContentsMargins(0, 0, 0, 0)
        conjuntoVBox.setSpacing(6)

        conjuntoVBox.addWidget(QLabel('Conjunto:'))

        formConjunto = QFormLayout()
        formConjunto.setContentsMargins(0, 0, 0, 0)
        formConjunto.setSpacing(6)

        self.theta1ConjuntoInput = QLineEdit(); self.theta1ConjuntoInput.setText(f"{self.state.conjunto.theta1_deg}")
        self.xConjuntoInput = QLineEdit(); self.xConjuntoInput.setText(f"{self.state.conjunto.dx}")
        self.yConjuntoInput = QLineEdit(); self.yConjuntoInput.setText(f"{self.state.conjunto.dy}")
        self.theta2ConjuntoInput = QLineEdit(); self.theta2ConjuntoInput.setText(f"{self.state.conjunto.theta2_deg}")

        formConjunto.addRow(QLabel('Primera rotación [°]:'), self.theta1ConjuntoInput)
        formConjunto.addRow(QLabel('Desplazamiento x [m]:'), self.xConjuntoInput)
        formConjunto.addRow(QLabel('Desplazamiento y [m]:'), self.yConjuntoInput)
        formConjunto.addRow(QLabel('Segunda rotación [°]:'), self.theta2ConjuntoInput)

        conjuntoVBox.addLayout(formConjunto)

        self.btnAplicarConjunto = QPushButton("Aplicar conjunto")
        conjuntoVBox.addWidget(self.btnAplicarConjunto)

        conjuntoScroll = QScrollArea()
        conjuntoScroll.setWidgetResizable(True)
        conjuntoScroll.setFrameShape(QFrame.NoFrame)
        conjuntoScroll.setWidget(conjuntoContainer)

        izqLayout.addWidget(conjuntoScroll)

        # ======================================================
        # Lista de SÓLIDOS + acciones
        # ======================================================
        linea = QFrame(); linea.setFrameShape(QFrame.HLine); linea.setFrameShadow(QFrame.Sunken)
        izqLayout.addWidget(linea)

        izqLayout.addWidget(QLabel('Sólidos agregados:'))
        self.listaSolidos = QListWidget(); self.listaSolidos.setSelectionMode(QAbstractItemView.SingleSelection)
        izqLayout.addWidget(self.listaSolidos, 1)

        row2 = QHBoxLayout(); self.btnEliminar = QPushButton('Eliminar selección'); self.btnEditar = QPushButton('Editar')
        row2.addWidget(self.btnEliminar); row2.addWidget(self.btnEditar); row2.addStretch(1); izqLayout.addLayout(row2)

        # Botones agregar/guardar
        rowAddSave = QHBoxLayout()
        self.btnAgregar = QPushButton('Agregar')
        self.btnGuardarCambios = QPushButton('Guardar cambios'); self.btnGuardarCambios.setEnabled(False)
        rowAddSave.addWidget(self.btnAgregar); rowAddSave.addWidget(self.btnGuardarCambios)

        izqLayout.insertLayout(2, rowAddSave)

        splitter.addWidget(izq)

        # ----- Columna derecha (preview interactivo) -----
        der = QWidget(); derLayout = QVBoxLayout(der)
        self.btnMostrar = QPushButton('Mostrar conjunto'); derLayout.addWidget(self.btnMostrar)
        self.previewBox = QGroupBox("Vista previa del conjunto")
        self.previewLayout = QVBoxLayout(self.previewBox)
        derLayout.addWidget(self.previewBox, 1)
        splitter.addWidget(der); splitter.setSizes([380, 600])

        layout.addWidget(splitter)

        # Conexiones
        self.btnCargarPerfil.clicked.connect(self._onCargarPerfil)
        self.btnAgregar.clicked.connect(self._onAgregarSolido)
        self.btnGuardarCambios.clicked.connect(self._onGuardarCambios)
        self.btnEliminar.clicked.connect(self._onEliminarSeleccion)
        self.btnEditar.clicked.connect(self._onEditarSeleccion)
        self.btnAplicarConjunto.clicked.connect(self._onAplicarConjunto)
        self.btnMostrar.clicked.connect(self._onMostrarConjunto)

        # Toggle de visibilidad para FLAP
        self.esFlap.toggled.connect(self.flapBoxScroll.setVisible)

        self._refrescar_sugerencia_si_corresponde(force=True)
        for s in self.state.solidos.values():
            self._add_list_item_for_solid(s)

    def _indices_ocupados(self):
        usados=set()
        for i in range(self.listaSolidos.count()):
            name = self.listaSolidos.item(i).text()
            m = self._re_solido.match(name)
            if m: 
                usados.add(int(m.group(1)))
        return usados
    
    def _siguiente_indice_libre(self):
        usados = self._indices_ocupados()
        n = 0
        while n in usados: 
            n += 1
        return n
    
    def _sugerencia(self): 
        return f"Sólido {self._siguiente_indice_libre()}"
    
    def _es_sugerencia_actual(self, text: str) -> bool: 
        return text.strip().lower() == self._sugerencia().strip().lower()
    
    def _refrescar_sugerencia_si_corresponde(self, force: bool = False):
        txt = self.nombreInput.text().strip()

        if force or txt == "" or self._es_sugerencia_actual(txt): 
            self.nombreInput.setText(self._sugerencia())

    def _add_list_item_for_solid(self, s: SolidConfig):
        item = QListWidgetItem(s.nombre)
        item.setData(Qt.UserRole, s.id)
        self.listaSolidos.addItem(item)

    def _get_solids(self) -> List[SolidConfig]:
        solids=[]
        for i in range(self.listaSolidos.count()):
            it = self.listaSolidos.item(i)
            sid = it.data(Qt.UserRole)
            s = self.state.solidos.get(sid)
            if s is not None: 
                solids.append(s)
        return solids
    
    def _nombre_unico(self, nombre: str, excluir_item: Optional[QListWidgetItem] = None) -> str:
        existentes = {self.listaSolidos.item(i).text().strip() for i in range(self.listaSolidos.count())
                    if (excluir_item is None or self.listaSolidos.item(i) is not excluir_item)}
        if nombre not in existentes: 
            return nombre
        
        m = self._re_solido.match(nombre)
        if m: 
            nombre = self._sugerencia(); existentes.add(nombre)
        base = nombre
        k = 2
        while f"{base} ({k})" in existentes: 
            k += 1
        return f"{base} ({k})"

    def _onCargarPerfil(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Cargar perfil", "", "Perfiles (*.dat *.txt *.csv);;Todos los archivos (*)")
        if ruta: 
            self.edPerfilPath.setText(ruta)

    def _leer_flap(self) -> Optional[FlapConfig]:
        if not self.esFlap.isChecked(): 
            return None
        return FlapConfig(cf=float(self.cfInput.text() or "0"), df=float(self.dfInput.text() or "0"),
                          h_TEw_ROTf=float(self.h_TEw_ROTf_Input.text() or "0"), v_TEw_ROTf=float(self.v_TEw_ROTf_Input.text() or "0"),
                          h_ROTf_BAf=float(self.h_ROTf_BAf_Input.text() or "0"), v_ROTf_MCf=float(self.v_ROTf_MCf_Input.text() or "0"))
    
    def _load_form_from_solid(self, s: SolidConfig):
        self.edPerfilPath.setText(s.perfil_path or "")
        self.cbFormato.setCurrentIndex(0 if (s.formato or "").lower()=="selig" else 1)
        self.nombreInput.setText(s.nombre)
        self.cInput.setText(f"{s.c}")
        self.theta1Input.setText(f"{s.theta1_deg}")
        self.xInput.setText(f"{s.dx}")
        self.yInput.setText(f"{s.dy}")
        self.theta2Input.setText(f"{s.theta2_deg}")
        self.nIntraInput.setText(f"{s.n_intra}")
        self.nExtraInput.setText(f"{s.n_extra}")
        self.esFlap.setChecked(s.es_flap); self.flapBoxScroll.setVisible(s.es_flap)
        if s.es_flap and s.flap is not None:
            self.cfInput.setText(f"{s.flap.cf}")
            self.dfInput.setText(f"{s.flap.df}")
            self.h_TEw_ROTf_Input.setText(f"{s.flap.h_TEw_ROTf}")
            self.v_TEw_ROTf_Input.setText(f"{s.flap.v_TEw_ROTf}")
            self.h_ROTf_BAf_Input.setText(f"{s.flap.h_ROTf_BAf}")
            self.v_ROTf_MCf_Input.setText(f"{s.flap.v_ROTf_MCf}")

    def _solid_from_inputs(self, name_override: Optional[str] = None, keep_id: Optional[str] = None, cerrar_te: bool = False) -> SolidConfig:
        sid = keep_id or str(uuid.uuid4())
        nombre = name_override if name_override is not None else (self.nombreInput.text().strip() or self._sugerencia())
        perfil_path_txt = self.edPerfilPath.text().strip() or None
        formato_txt = self.cbFormato.currentText().strip().lower()
        es_flap = self.esFlap.isChecked()
        flap_cfg = self._leer_flap() if es_flap else None
        return SolidConfig(
            id=sid, nombre=nombre, perfil_path=perfil_path_txt, formato=formato_txt,
            c=float(self.cInput.text() or "0.508"),
            theta1_deg=float(self.theta1Input.text() or "0"),
            dx=float(self.xInput.text() or "0"),
            dy=float(self.yInput.text() or "0"),
            theta2_deg=float(self.theta2Input.text() or "0"),
            n_intra=int(self.nIntraInput.text() or "0"),
            n_extra=int(self.nExtraInput.text() or "0"),
            es_flap=es_flap, flap=flap_cfg,
            cerrar_te=cerrar_te
        )

    # Cierre BF
    @staticmethod
    def _line_intersection(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> Optional[np.ndarray]:
        x1, y1 = p0; x2, y2 = p1
        x3, y3 = q0; x4, y4 = q1
        den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(den) < 1e-14:
            return None
        numx = (x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)
        numy = (x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)
        return np.array([numx/den, numy/den], dtype=float)

    @classmethod
    def _close_te_by_extension_coords(cls, x: List[List[float]], y: List[List[float]], c: float) -> Optional[Tuple[List[List[float]], List[List[float]]]]:
        if len(x) != 2 or len(y) != 2: return None
        if len(x[0]) < 2 or len(x[1]) < 2: return None
        p0 = np.array([x[0][0], y[0][0]], dtype=float)
        p1 = np.array([x[0][1], y[0][1]], dtype=float)
        q1 = np.array([x[1][-2], y[1][-2]], dtype=float)
        q0 = np.array([x[1][-1], y[1][-1]], dtype=float)
        inter = cls._line_intersection(p0, p1, q0, q1)
        if inter is None:
            return None
        x_mod = [list(xx) for xx in x]
        y_mod = [list(yy) for yy in y]
        x_mod[0][0], y_mod[0][0]   = float(inter[0]), float(inter[1])
        x_mod[1][-1], y_mod[1][-1] = float(inter[0]), float(inter[1])

        cuerda0 = np.sqrt((x_mod[0][0] - x_mod[0][-1]) ** 2 + (y_mod[0][0] - y_mod[0][-1]) ** 2)
        factor = c/cuerda0
        x_mod = [[xyi * factor for xyi in xy] for xy in x_mod]
        y_mod = [[xyi * factor for xyi in xy] for xy in y_mod]
        return x_mod, y_mod

    @staticmethod
    def _discretizar_desde_coords(x: List[List[float]], y: List[List[float]],
                                  nIntrados: int, nExtrados: int,
                                  espaciamiento: Literal['cos','tanh']='cos') -> np.ndarray:
        c = x[0][0] - x[0][-1]
        n = [nIntrados+1, nExtrados+1]
        xInt: List[float] = []
        yInt: List[float] = []
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
                raise ValueError(f"Espaciamiento inválido: {espaciamiento!r}")
            xi = list(x[i]); yi = list(y[i])
            if i == 0:
                xx = np.flip(xx); xx = xx[0:-1]
                xi.reverse(); yi.reverse()
            yy = np.interp(xx, xi, yi)
            xInt += xx.tolist(); yInt += yy.tolist()
        return np.stack([np.array(xInt), np.array(yInt)], axis=0)

    def _import_coords_or_warn(self, ruta: str, formato: str, cuerda: float, bf_cerrado: bool=False) -> Optional[Tuple[List[List[float]], List[List[float]]]]:
        try:
            x, y = importarPerfil(ruta=ruta, formato=formato, cuerda=cuerda, bordeDeFugaCerrado=bf_cerrado)
            return x, y
        except Exception:
            QMessageBox.warning(self, "Advertencia", "El perfil no pudo ser cargado desde el archivo de coordenadas")
            return None

    def _check_te_and_get_coords(self, ruta: str, formato: str, cuerda: float,
                                 n_intra: int, n_extra: int) -> Optional[Tuple[List[List[float]], List[List[float]], bool]]:
        pair = self._import_coords_or_warn(ruta, formato, cuerda, bf_cerrado=False)
        if pair is None:
            return None
        x, y = pair
        bf_intra = (x[0][0], y[0][0]); bf_extra = (x[1][-1], y[1][-1])
        cerrado = (bf_intra[0] == bf_extra[0]) and (bf_intra[1] == bf_extra[1])
        if cerrado:
            return x, y, False
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Borde de fuga no cerrado")
        msg.setText("Borde de fuga no cerrado en el archivo de coordenadas. ¿Desea cerrarlo?")
        btn_no = msg.addButton("No, omitir sólido", QMessageBox.RejectRole)
        btn_si = msg.addButton("Sí", QMessageBox.AcceptRole)
        msg.exec()
        if msg.clickedButton() is btn_si:
            closed = self._close_te_by_extension_coords(x, y, cuerda)
            if closed is None:
                return None
            x_mod, y_mod = closed
            return x_mod, y_mod, True
        else:
            return None

    def _auto_center_conjunto_if_flap(self, s: SolidConfig):
        if not (s.es_flap and self.chkFlapCenterOrigin.isChecked() and s.flap is not None):
            return
        pair = self._import_coords_or_warn(
            ruta=s.perfil_path, formato=s.formato,
            cuerda=(s.flap.cf if s.es_flap else s.c),
            bf_cerrado=False
        )
        if pair is None:
            return
        x, y = pair
        if s.cerrar_te:
            cuerda = (s.flap.cf if s.es_flap else s.c)
            closed = self._close_te_by_extension_coords(x, y, cuerda)
            if closed is not None:
                x, y = closed
        r_xy = self._discretizar_desde_coords(x, y, s.n_intra, s.n_extra, 'cos')
        r_xy = Cinematicas.moverCoordenadas(r_xy, 0.0, 0.0, s.theta1_deg)
        r_xy = Cinematicas.moverCoordenadas(r_xy, s.dx, s.dy, 0.0)
        r_xy = Cinematicas.moverCoordenadas(r_xy, 0.0, 0.0, s.theta2_deg)
        flapParams = {'cf':s.flap.cf,'cw':s.c,'df':s.flap.df,'h_ROTf_BAf':s.flap.h_ROTf_BAf,'h_TEw_ROTf':s.flap.h_TEw_ROTf,
                      'v_ROTf_MCf':s.flap.v_ROTf_MCf,'v_TEw_ROTf':s.flap.v_TEw_ROTf,'r_xy':r_xy}
        try:
            _, r_ROT_xy = Cinematicas.flap(flapParams)
            cx = float(r_ROT_xy[0, 0]); cy = float(r_ROT_xy[1, 0])
            self.xConjuntoInput.setText(f"{-cx}"); self.yConjuntoInput.setText(f"{-cy}")
        except Exception:
            pass

    def _onAgregarSolido(self):
        if not self.edPerfilPath.text().strip():
            return
        nombre = (self.nombreInput.text().strip() or self._sugerencia())
        nombre = self._nombre_unico(nombre)
        ruta = self.edPerfilPath.text().strip()
        formato = self.cbFormato.currentText().strip().lower()
        cuerda = float(self.cInput.text() or "0.508")
        n_extra = int(self.nExtraInput.text() or "0")
        n_intra = int(self.nIntraInput.text() or "0")
        res = self._check_te_and_get_coords(ruta, formato, cuerda, n_intra, n_extra)
        if res is None:
            return
        x_mod, y_mod, cerrar_flag = res
        s = self._solid_from_inputs(name_override=nombre, cerrar_te=cerrar_flag)
        self.state.solidos[s.id] = s
        self._add_list_item_for_solid(s)
        self._refrescar_sugerencia_si_corresponde(force=True)
        self._auto_center_conjunto_if_flap(s)
        self.solidsChanged.emit()

    def _onGuardarCambios(self):
        if self._editing_id is None:
            return
        old_id = self._editing_id
        old_name = self._editing_original_name or ""
        if old_id not in self.state.solidos:
            self._editing_id = None
            self._editing_original_name = None
            self.btnGuardarCambios.setEnabled(False)
            return
        new_name_input = (self.nombreInput.text().strip() or self._sugerencia())
        name_changed = (new_name_input != old_name)
        cerrar_flag = self.state.solidos[old_id].cerrar_te
        if not name_changed:
            s = self._solid_from_inputs(name_override=old_name, keep_id=old_id, cerrar_te=cerrar_flag)
            self.state.solidos[old_id] = s
            for i in range(self.listaSolidos.count()):
                it = self.listaSolidos.item(i)
                if it.data(Qt.UserRole) == old_id:
                    it.setText(s.nombre)
                    break
            self._auto_center_conjunto_if_flap(s)
        else:
            new_name = self._nombre_unico(new_name_input)
            s = self._solid_from_inputs(name_override=new_name, cerrar_te=cerrar_flag)
            self.state.solidos[s.id] = s
            self._add_list_item_for_solid(s)
            del self.state.solidos[old_id]
            for i in range(self.listaSolidos.count()):
                it = self.listaSolidos.item(i)
                if it.data(Qt.UserRole) == old_id:
                    self.listaSolidos.takeItem(i)
                    break
            self._auto_center_conjunto_if_flap(s)
        self._editing_id = None
        self._editing_original_name = None
        self.btnGuardarCambios.setEnabled(False)
        self.solidsChanged.emit()

    def _onEliminarSeleccion(self):
        row = self.listaSolidos.currentRow()
        if row < 0:
            return
        item = self.listaSolidos.item(row)
        sid = item.data(Qt.UserRole)
        if sid in self.state.solidos:
            del self.state.solidos[sid]
        self.listaSolidos.takeItem(row)
        self._refrescar_sugerencia_si_corresponde(force=True)
        if self._editing_id == sid:
            self._editing_id = None
            self._editing_original_name = None
            self.btnGuardarCambios.setEnabled(False)
        self.solidsChanged.emit()

    def _onEditarSeleccion(self):
        item=self.listaSolidos.currentItem()
        if not item: return
        sid=item.data(Qt.UserRole); s=self.state.solidos.get(sid)
        if s is None: return
        self._load_form_from_solid(s); self._editing_id=sid; self._editing_original_name=s.nombre; self.btnGuardarCambios.setEnabled(True)

    def _onAplicarConjunto(self):
        self.state.conjunto = ConjuntoConfig(
            theta1_deg=float(self.theta1ConjuntoInput.text() or "0"),
            dx=float(self.xConjuntoInput.text() or "0"),
            dy=float(self.yConjuntoInput.text() or "0"),
            theta2_deg=float(self.theta2ConjuntoInput.text() or "0")
        )

    def _clear_preview_area(self):
        while self.previewLayout.count():
            item=self.previewLayout.takeAt(0); w=item.widget()
            if w is not None: w.deleteLater()

    def _onMostrarConjunto(self):
        solids=self._get_solids()
        if not solids: return
        self._clear_preview_area(); self._render_preview(solids, self.state.conjunto)

    def _render_preview(self, solidosConfig: List[SolidConfig], conjuntoConfig: ConjuntoConfig):
        fig, ax = plt.subplots(1, 1)
        for nombre, r_xy in self.build_all_rxy(solidosConfig, conjuntoConfig):
            ax.plot(r_xy[0, :], r_xy[1, :], label=nombre)
        ax.plot(0, 0, 'gx', markersize=10, label='Rm')
        ax.legend(); ax.set_xlabel('x, [m]'); ax.set_ylabel('y, [m]')
        ax.axis('equal'); ax.grid(True)
        canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar2QT(canvas, self)
        self.previewLayout.addWidget(toolbar)
        self.previewLayout.addWidget(canvas)

    def build_rxy_for_solid(self, s: SolidConfig, conjunto: ConjuntoConfig):
        if (s.perfil_path is None) or s.perfil_missing or (not os.path.exists(s.perfil_path)):
            return np.zeros((2, 2), dtype=float)

        pair = self._import_coords_or_warn(
            ruta=s.perfil_path, formato=s.formato, cuerda=(s.flap.cf if s.es_flap else s.c), bf_cerrado=False
        )
        if pair is None:
            return np.zeros((2, 2), dtype=float)

        x, y = pair
        if s.cerrar_te:
            cuerda = (s.flap.cf if s.es_flap else s.c)
            closed = self._close_te_by_extension_coords(x, y, cuerda)
            if closed is not None:
                x, y = closed

        r_xy = self._discretizar_desde_coords(x, y, s.n_intra, s.n_extra, 'cos')
        r_xy = Cinematicas.moverCoordenadas(r_xy, 0.0, 0.0, s.theta1_deg)
        r_xy = Cinematicas.moverCoordenadas(r_xy, s.dx, s.dy, 0.0)
        r_xy = Cinematicas.moverCoordenadas(r_xy, 0.0, 0.0, s.theta2_deg)

        if s.es_flap and s.flap is not None:
            flapParams={'cf':s.flap.cf,'cw':s.c,'df':s.flap.df,'h_ROTf_BAf':s.flap.h_ROTf_BAf,'h_TEw_ROTf':s.flap.h_TEw_ROTf,
                        'v_ROTf_MCf':s.flap.v_ROTf_MCf,'v_TEw_ROTf':s.flap.v_TEw_ROTf,'r_xy':r_xy}
            try:
                r_xy,_=Cinematicas.flap(flapParams)
            except Exception:
                pass

        r_xy = Cinematicas.moverCoordenadas(r_xy, 0.0, 0.0, conjunto.theta1_deg)
        r_xy = Cinematicas.moverCoordenadas(r_xy, conjunto.dx, conjunto.dy, 0.0)
        r_xy = Cinematicas.moverCoordenadas(r_xy, 0.0, 0.0, conjunto.theta2_deg)
        return r_xy

    def build_all_rxy(self, solidos: List[SolidConfig] | None = None, conjunto: ConjuntoConfig | None = None) -> List[Tuple[str, np.ndarray]]:
        if solidos is None: solidos=self._get_solids()
        if conjunto is None: conjunto=self.state.conjunto
        out=[]
        for s in solidos:
            r_xy=self.build_rxy_for_solid(s, conjunto); out.append((s.nombre, r_xy))
        return out

    def reload_from_state(self):
        self.listaSolidos.clear()
        for s in self.state.solidos.values():
            self._add_list_item_for_solid(s)
        self.theta1ConjuntoInput.setText(f"{self.state.conjunto.theta1_deg}")
        self.xConjuntoInput.setText(f"{self.state.conjunto.dx}")
        self.yConjuntoInput.setText(f"{self.state.conjunto.dy}")
        self.theta2ConjuntoInput.setText(f"{self.state.conjunto.theta2_deg}")
        self._refrescar_sugerencia_si_corresponde(force=True)
        self.solidsChanged.emit()

class SimulacionTab(QWidget):
    modoChanged = Signal(bool, int)
    simulateRequested = Signal(SimulationConfig)

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        layout = QVBoxLayout(self)

        # ----- Referencias -----
        gbRef = QGroupBox("Magnitudes de referencia"); refForm = QFormLayout(gbRef)
        self.edL = QLineEdit(); _num(self.edL); self.edL.setText(f"{self.state.l_ref}")
        self.edVref = QLineEdit(); _num(self.edVref); self.edVref.setText(f"{self.state.V_ref}")
        self.edRho = QLineEdit(); _num(self.edRho); self.edRho.setText(f"{self.state.rho_ref}")
        refForm.addRow('l [m]:', self.edL)
        refForm.addRow('V [m/s]:', self.edVref)
        refForm.addRow('\u03C1 [kg/m^3]:', self.edRho)
        self.edNu = QLineEdit(); _num(self.edNu); self.edNu.setText("1.5e-5")
        self.edA  = QLineEdit(); _num(self.edA);  self.edA.setText("340.3")
        self.edG  = QLineEdit(); _num(self.edG);  self.edG.setText("9.81")

        lblNu = QLabel('ν [m<sup>2</sup>/s]:'); lblNu.setTextFormat(Qt.RichText)
        lblG  = QLabel('g [m/s<sup>2</sup>]:'); lblG.setTextFormat(Qt.RichText)

        refForm.addRow(lblNu, self.edNu)
        refForm.addRow('a [m/s]:', self.edA)
        refForm.addRow(lblG, self.edG)

        layout.addWidget(gbRef)

        # ----- Modo -----
        modoRow = QHBoxLayout()
        modoRow.addWidget(QLabel('Modo:'))
        self.cbModo = QComboBox()
        self.cbModo.addItems(['Estacionario', 'No estacionario'])
        modoRow.addWidget(self.cbModo, 1)
        layout.addLayout(modoRow)

        self.stackCinematica = QStackedWidget()
        layout.addWidget(self.stackCinematica)

        # =======================
        #       ESTACIONARIO
        # =======================
        est = QWidget(); estLayout = QVBoxLayout(est); estForm = QFormLayout()
        self.est_V = QLineEdit(); _num(self.est_V, '35.76'); self.est_V.setText("35.76")
        self.est_alfa_i = QLineEdit(); self.est_alfa_i.setText("-5")
        self.est_alfa_f = QLineEdit(); self.est_alfa_f.setText("5")
        self.est_delta_alfa = QLineEdit(); self.est_delta_alfa.setText("1")
        estForm.addRow('V [m/s]:', self.est_V)
        estForm.addRow('αi [°]:', self.est_alfa_i)
        estForm.addRow('αf [°]:', self.est_alfa_f)
        estForm.addRow('Δα [°]:', self.est_delta_alfa)
        estLayout.addLayout(estForm)
        self.btnSimEst = QPushButton('Simular')
        self.btnSimEst.clicked.connect(self._onSimular)
        estLayout.addWidget(self.btnSimEst)
        self.pbEst = QProgressBar(); self.pbEst.setTextVisible(True); self.pbEst.setValue(0)
        estLayout.addWidget(self.pbEst)

        self.adimEst = AdimInfoWidget()
        estLayout.addWidget(self.adimEst)
        estLayout.addStretch(1)

        # =======================
        #    NO ESTACIONARIO
        # =======================
        noEst = QWidget(); noEstLayout = QVBoxLayout(noEst)
        noEstRow = QHBoxLayout()
        noEstRow.addWidget(QLabel('Cinemática:'))
        self.cbCinematicaNoEst = QComboBox()
        self.cbCinematicaNoEst.addItems(['Movimiento rectilíneo uniforme','Rotación armónica','Traslación armónica','Desde archivos de texto'])
        noEstRow.addWidget(self.cbCinematicaNoEst, 1)
        noEstLayout.addLayout(noEstRow)

        self.stackNoEst = QStackedWidget()
        noEstLayout.addWidget(self.stackNoEst)

        # ---- MRU ----
        noEstMRU = QWidget(); noEstMRULayout = QVBoxLayout(noEstMRU); noEstMRUForm = QFormLayout()
        self.ne_mru_V = QLineEdit(); _num(self.ne_mru_V,'35.76'); self.ne_mru_V.setText("35.76")
        self.ne_mru_alfa = QLineEdit(); _num(self.ne_mru_alfa,'0.0'); self.ne_mru_alfa.setText("0.0")
        self.ne_mru_tfinal = QLineEdit(); _num(self.ne_mru_tfinal,'0.01'); self.ne_mru_tfinal.setText("0.01")
        self.ne_mru_dt = QLineEdit(); self.ne_mru_dt.setText("0.0001")
        noEstMRUForm.addRow('V [m/s]:', self.ne_mru_V)
        noEstMRUForm.addRow('α [°]:', self.ne_mru_alfa)
        noEstMRUForm.addRow('t final [s]:', self.ne_mru_tfinal)
        noEstMRUForm.addRow('Δt [s]:', self.ne_mru_dt)
        noEstMRULayout.addLayout(noEstMRUForm)
        self.btnSimMRU = QPushButton('Simular')
        self.btnSimMRU.clicked.connect(self._onSimular)
        noEstMRULayout.addWidget(self.btnSimMRU)
        self.pbMRU = QProgressBar(); self.pbMRU.setTextVisible(True); self.pbMRU.setValue(0)
        noEstMRULayout.addWidget(self.pbMRU)
        self.adimMRU = AdimInfoWidget()
        noEstMRULayout.addWidget(self.adimMRU)
        noEstMRULayout.addStretch(1)

        # ---- Rotación armónica ----
        noEstRot = QWidget(); noEstRotLayout = QVBoxLayout(noEstRot); noEstRotForm = QFormLayout()
        self.ne_rot_V = QLineEdit(); _num(self.ne_rot_V,'35.76'); self.ne_rot_V.setText("35.76")
        self.ne_rot_alfa0 = QLineEdit(); _num(self.ne_rot_alfa0,'6.74'); self.ne_rot_alfa0.setText("6.74")
        self.ne_rot_w = QLineEdit(); _num(self.ne_rot_w,'200.0'); self.ne_rot_w.setText("200.0")
        self.ne_rot_tfinal = QLineEdit(); _num(self.ne_rot_tfinal,'0.01'); self.ne_rot_tfinal.setText("0.01")
        self.ne_rot_dt = QLineEdit(); self.ne_rot_dt.setText("0.0001")
        noEstRotForm.addRow('V [m/s]:', self.ne_rot_V)
        noEstRotForm.addRow('α0 [°]:', self.ne_rot_alfa0)
        noEstRotForm.addRow('ω [rad/s]:', self.ne_rot_w)
        noEstRotForm.addRow('t final [s]:', self.ne_rot_tfinal)
        noEstRotForm.addRow('Δt [s]:', self.ne_rot_dt)
        noEstRotLayout.addLayout(noEstRotForm)
        self.btnSimRot = QPushButton('Simular')
        self.btnSimRot.clicked.connect(self._onSimular)
        noEstRotLayout.addWidget(self.btnSimRot)
        self.pbRot = QProgressBar(); self.pbRot.setTextVisible(True); self.pbRot.setValue(0)
        noEstRotLayout.addWidget(self.pbRot)
        self.adimRot = AdimInfoWidget()
        noEstRotLayout.addWidget(self.adimRot)
        noEstRotLayout.addStretch(1)

        # ---- Traslación armónica ----
        noEstTras = QWidget(); noEstTrasLayout = QVBoxLayout(noEstTras); noEstTrasForm = QFormLayout()
        self.ne_tras_V = QLineEdit(); _num(self.ne_tras_V,'35.76'); self.ne_tras_V.setText("35.76")
        self.ne_tras_h0 = QLineEdit(); _num(self.ne_tras_h0,'0.0254'); self.ne_tras_h0.setText("0.0254")
        self.ne_tras_w = QLineEdit(); _num(self.ne_tras_w,'200.0'); self.ne_tras_w.setText("200.0")
        self.ne_tras_tfinal = QLineEdit(); _num(self.ne_tras_tfinal,'0.01'); self.ne_tras_tfinal.setText("0.01")
        self.ne_tras_dt = QLineEdit(); self.ne_tras_dt.setText("0.0001")
        noEstTrasForm.addRow('V [m/s]:', self.ne_tras_V)
        noEstTrasForm.addRow('h0 [m]:', self.ne_tras_h0)
        noEstTrasForm.addRow('ω [rad/s]:', self.ne_tras_w)
        noEstTrasForm.addRow('t final [s]:', self.ne_tras_tfinal)
        noEstTrasForm.addRow('Δt [s]:', self.ne_tras_dt)
        noEstTrasLayout.addLayout(noEstTrasForm)
        self.btnSimTras = QPushButton('Simular')
        self.btnSimTras.clicked.connect(self._onSimular)
        noEstTrasLayout.addWidget(self.btnSimTras)
        self.pbTras = QProgressBar(); self.pbTras.setTextVisible(True); self.pbTras.setValue(0)
        noEstTrasLayout.addWidget(self.pbTras)
        self.adimTras = AdimInfoWidget()
        noEstTrasLayout.addWidget(self.adimTras)
        noEstTrasLayout.addStretch(1)

        # ---- Desde archivo
        noEstArchivo = QWidget(); noEstArchivoLayout = QVBoxLayout(noEstArchivo)
        rmRow = QHBoxLayout()
        rmRow.addWidget(QLabel("RM:"))
        self.btnCargarRM = QPushButton("Cargar archivo…")
        self.ne_rm_path = QLineEdit(); self.ne_rm_path.setReadOnly(True)
        self.ne_rm_path.setPlaceholderText("Ningún archivo seleccionado")
        rmRow.addWidget(self.btnCargarRM)
        rmRow.addWidget(self.ne_rm_path, 1)
        noEstArchivoLayout.addLayout(rmRow)

        self.ne_archivo_scroll = QScrollArea()
        self.ne_archivo_scroll.setWidgetResizable(True)
        self._ne_archivo_inner = QWidget()
        self._ne_archivo_form = QFormLayout(self._ne_archivo_inner)
        self._ne_archivo_form.setContentsMargins(0, 0, 0, 0)
        self._ne_archivo_form.setSpacing(6)
        self.ne_archivo_scroll.setWidget(self._ne_archivo_inner)
        noEstArchivoLayout.addWidget(self.ne_archivo_scroll)

        self.btnSimArchivo = QPushButton('Simular')
        self.btnSimArchivo.clicked.connect(self._onSimular)
        noEstArchivoLayout.addWidget(self.btnSimArchivo)
        self.pbArchivo = QProgressBar(); self.pbArchivo.setTextVisible(True); self.pbArchivo.setValue(0)
        noEstArchivoLayout.addWidget(self.pbArchivo)
        self.adimArchivo = AdimInfoWidget()
        noEstArchivoLayout.addWidget(self.adimArchivo)
        noEstArchivoLayout.addStretch(1)

        self.stackNoEst.addWidget(noEstMRU)
        self.stackNoEst.addWidget(noEstRot)
        self.stackNoEst.addWidget(noEstTras)
        self.stackNoEst.addWidget(noEstArchivo)

        self.stackCinematica.addWidget(est)
        self.stackCinematica.addWidget(noEst)

        self.cbModo.currentIndexChanged.connect(self._onModoChanged)
        self.cbCinematicaNoEst.currentIndexChanged.connect(self._onCinematicaNoEstChanged)
        self.btnCargarRM.clicked.connect(self._onCargarRMFile)

        self._ne_rm_file: Optional[str] = None
        self._ne_archivo_paths: Dict[str, str] = {}
        self._ne_archivo_lineedits: Dict[str, QLineEdit] = {}
        self._ne_archivo_buttons: Dict[str, QPushButton] = {}
        self._ne_archivo_relflags: Dict[str, QCheckBox] = {}
        self._ne_archivo_rel_values: Dict[str, bool] = {}

        self._onModoChanged(self.cbModo.currentIndex())
        self._onCinematicaNoEstChanged(self.cbCinematicaNoEst.currentIndex())

    # ---- API de progreso para MainWindow ----
    def _current_progress_bar(self) -> QProgressBar:
        if self.stackCinematica.currentIndex() == 0:
            return self.pbEst
        else:
            idx = self.stackNoEst.currentIndex()
            return [self.pbMRU, self.pbRot, self.pbTras, self.pbArchivo][idx]

    def start_progress(self, max_steps: int):
        pb = self._current_progress_bar()
        pb.setRange(0, max(0, int(max_steps)))
        pb.setValue(0)
        pb.setFormat("Progreso: %p%")

    def set_progress(self, value: int):
        pb = self._current_progress_bar()
        pb.setValue(int(value))

    def finish_progress(self):
        pb = self._current_progress_bar()
        pb.setValue(pb.maximum())

    # ---- Lectura de parámetros de UI -> SimulationConfig ----
    def _collect_config(self) -> SimulationConfig:
        if self.stackCinematica.currentIndex() == 0:
            return SimulationConfig(
                modo="estacionario",
                V=float(self.est_V.text() or "0"),
                alfa_i=float(self.est_alfa_i.text() or "0"),
                alfa_f=float(self.est_alfa_f.text() or "0"),
                delta_alfa=float(self.est_delta_alfa.text() or "0")
            )
        else:
            subidx = self.stackNoEst.currentIndex()
            if subidx == 0:  # MRU
                return SimulationConfig(
                    modo="no_estacionario", submodo="mru",
                    V=float(self.ne_mru_V.text() or "0"),
                    alfa=float(self.ne_mru_alfa.text() or "0"),
                    t_final=float(self.ne_mru_tfinal.text() or "0"),
                    dt=float(self.ne_mru_dt.text() or "0.01")
                )
            elif subidx == 1:  # Rotación armónica
                return SimulationConfig(
                    modo="no_estacionario", submodo="rotacion",
                    V=float(self.ne_rot_V.text() or "0"),
                    alfa=float(self.ne_rot_alfa0.text() or "0"),
                    w=float(self.ne_rot_w.text() or "0"),
                    t_final=float(self.ne_rot_tfinal.text() or "0"),
                    dt=float(self.ne_rot_dt.text() or "0.01")
                )
            elif subidx == 2:  # Traslación armónica
                return SimulationConfig(
                    modo="no_estacionario", submodo="traslacion",
                    V=float(self.ne_tras_V.text() or "0"),
                    h0=float(self.ne_tras_h0.text() or "0"),
                    w=float(self.ne_tras_w.text() or "0"),
                    t_final=float(self.ne_tras_tfinal.text() or "0"),
                    dt=float(self.ne_tras_dt.text() or "0.01")
                )
            else:  # Desde archivo
                rutas = { sid: ed.text().strip()
                          for sid, ed in self._ne_archivo_lineedits.items()
                          if ed.text().strip() }
                return SimulationConfig(
                    modo="no_estacionario",
                    submodo="archivo",
                    archivo_path=None,
                    rm_archivo_path=self._ne_rm_file,
                    archivos_por_solido=(rutas or None)
                )

    def _onModoChanged(self, idc: int):
        self.stackCinematica.setCurrentIndex(idc)
        estacionario = (idc == 0)
        instantes = 1
        self.modoChanged.emit(estacionario, instantes)

    def _onCinematicaNoEstChanged(self, idc: int):
        self.stackNoEst.setCurrentIndex(idc)
        if idc == 3:
            self.refresh_archivo_per_solid_rows()

    def _onCargarRMFile(self):
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Cargar archivo de RM", "", "Texto (*.txt *.csv *.dat);;Todos los archivos (*)"
        )
        if ruta:
            self._ne_rm_file = ruta
            self.ne_rm_path.setText(ruta)

    def _onSimular(self):
        cfg = self._collect_config()
        self.simulateRequested.emit(cfg)

    def refresh_archivo_per_solid_rows(self):
        self._ne_archivo_rel_values = {sid: chk.isChecked() for sid, chk in self._ne_archivo_relflags.items()}

        while self._ne_archivo_form.rowCount():
            self._ne_archivo_form.removeRow(0)
        self._ne_archivo_lineedits.clear()
        self._ne_archivo_buttons.clear()
        self._ne_archivo_relflags.clear()

        if not self.state.solidos:
            self._ne_archivo_form.addRow(QLabel("No hay sólidos cargados."))
            return

        for sid, s in self.state.solidos.items():
            fila = QWidget(); row = QHBoxLayout(fila); row.setContentsMargins(0,0,0,0); row.setSpacing(6)
            btn = QPushButton("Cargar archivo…")
            ed  = QLineEdit(); ed.setReadOnly(True)
            ed.setPlaceholderText("Ningún archivo seleccionado")

            chk = QCheckBox("Relativizar coordenadas")
            chk.setChecked(self._ne_archivo_rel_values.get(sid, True))

            if sid in self._ne_archivo_paths:
                ed.setText(self._ne_archivo_paths[sid])

            def _make_pick(sid=sid, ed=ed):
                def _pick():
                    ruta, _ = QFileDialog.getOpenFileName(
                        self, f"Curva no estacionaria para '{self.state.solidos[sid].nombre}'",
                        "", "Texto (*.txt *.csv *.dat);;Todos los archivos (*)"
                    )
                    if ruta:
                        ed.setText(ruta)
                        self._ne_archivo_paths[sid] = ruta
                return _pick
            btn.clicked.connect(_make_pick())

            row.addWidget(btn)
            row.addWidget(ed, 1)
            row.addWidget(chk)

            self._ne_archivo_lineedits[sid] = ed
            self._ne_archivo_buttons[sid] = btn
            self._ne_archivo_relflags[sid] = chk

            self._ne_archivo_form.addRow(QLabel(s.nombre), fila)

    def get_relativizar_flags(self) -> Dict[str, bool]:
        return {sid: chk.isChecked() for sid, chk in self._ne_archivo_relflags.items()}
    
    def apply_sim_ui(self, sim_dict: dict):
        # Referencias
        ref = sim_dict.get("referencias", {}) or {}
        if "l" in ref:   self.edL.setText(f"{ref.get('l')}")
        if "V" in ref:   self.edVref.setText(f"{ref.get('V')}")
        if "rho" in ref: self.edRho.setText(f"{ref.get('rho')}")
        if "nu" in ref:  self.edNu.setText(f"{ref.get('nu')}")
        if "a"  in ref:  self.edA.setText(f"{ref.get('a')}")
        if "g"  in ref:  self.edG.setText(f"{ref.get('g')}")

        # Modo
        modo_idx = int(sim_dict.get("modo_index", 0))
        self.cbModo.setCurrentIndex(modo_idx)
        self._onModoChanged(modo_idx)

        # Estacionario
        est = sim_dict.get("estacionario", {}) or {}
        if est:
            self.est_V.setText(est.get("V", self.est_V.text()))
            self.est_alfa_i.setText(est.get("alfa_i", self.est_alfa_i.text()))
            self.est_alfa_f.setText(est.get("alfa_f", self.est_alfa_f.text()))
            self.est_delta_alfa.setText(est.get("delta_alfa", self.est_delta_alfa.text()))

        # No estacionario
        nes_idx = int(sim_dict.get("no_est_submodo_index", 0))
        self.cbCinematicaNoEst.setCurrentIndex(nes_idx)
        self._onCinematicaNoEstChanged(nes_idx)

        mru = sim_dict.get("no_est_MRU", {}) or {}
        if mru:
            self.ne_mru_V.setText(mru.get("V", self.ne_mru_V.text()))
            self.ne_mru_alfa.setText(mru.get("alfa", self.ne_mru_alfa.text()))
            self.ne_mru_tfinal.setText(mru.get("t_final", self.ne_mru_tfinal.text()))
            self.ne_mru_dt.setText(mru.get("dt", self.ne_mru_dt.text()))

        rot = sim_dict.get("no_est_rotacion", {}) or {}
        if rot:
            self.ne_rot_V.setText(rot.get("V", self.ne_rot_V.text()))
            self.ne_rot_alfa0.setText(rot.get("alfa0", self.ne_rot_alfa0.text()))
            self.ne_rot_w.setText(rot.get("w", self.ne_rot_w.text()))
            self.ne_rot_tfinal.setText(rot.get("t_final", self.ne_rot_tfinal.text()))
            self.ne_rot_dt.setText(rot.get("dt", self.ne_rot_dt.text()))

        tras = sim_dict.get("no_est_traslacion", {}) or {}
        if tras:
            self.ne_tras_V.setText(tras.get("V", self.ne_tras_V.text()))
            self.ne_tras_h0.setText(tras.get("h0", self.ne_tras_h0.text()))
            self.ne_tras_w.setText(tras.get("w", self.ne_tras_w.text()))
            self.ne_tras_tfinal.setText(tras.get("t_final", self.ne_tras_tfinal.text()))
            self.ne_tras_dt.setText(tras.get("dt", self.ne_tras_dt.text()))

        # Desde archivo
        arch = sim_dict.get("no_est_archivo", {}) or {}
        self._ne_rm_file = arch.get("rm_archivo_path", None)
        if self._ne_rm_file:
            self.ne_rm_path.setText(self._ne_rm_file)
        else:
            self.ne_rm_path.clear()

        self.refresh_archivo_per_solid_rows()

        rutas_por_sid = arch.get("archivos_por_solido", {}) or {}
        rel_flags = arch.get("relativizar_por_solido", {}) or {}

        for sid, ed in self._ne_archivo_lineedits.items():
            ruta = rutas_por_sid.get(sid, "")
            ed.setText(ruta)
            if ruta:
                self._ne_archivo_paths[sid] = ruta

        for sid, chk in self._ne_archivo_relflags.items():
            chk.setChecked(bool(rel_flags.get(sid, True)))


    def update_adim_info_from_mp(self, mp: Optional[object]):
        data = None
        try:
            data = getattr(mp, "adimInfo", None)
            if not isinstance(data, dict):
                data = None
        except Exception:
            data = None

        for w in (getattr(self, "adimEst", None),
                getattr(self, "adimMRU", None),
                getattr(self, "adimRot", None),
                getattr(self, "adimTras", None),
                getattr(self, "adimArchivo", None)):
            if w is not None:
                w.set_data(data)

if __name__ == '__main__':
    QLocale.setDefault(QLocale.c())
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec())
