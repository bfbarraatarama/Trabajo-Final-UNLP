# Trabajo Final - Método de los Paneles Bidimensional, Multielemento y No Estacionario


- **Autor:** Bruno Francisco Barra Atarama  
- **Institución:** Departamento de Ingeniería Aeroespacial, Facultad de Ingeniería, Universidad Nacional de La Plata  
- **Año:** 2025  
- **Licencia:** PolyForm Noncommercial License 1.0.0 — ver [LICENSE](LICENSE)  
- **SPDX:** PolyForm-Noncommercial-1.0.0  

## Motivación
El estudio del flujo alrededor de cuerpos aerodinámicos en problemas no estacionarios es un aspecto fundamental en la ingeniería aeroespacial ya que permite analizar las fuerzas y momentos que actúan sobre vehículos y estructuras inmersos en un fluido con el que interactúan y evolucionan de forma conjunta con el pasar del tiempo.

Ejemplos de este tipo de problemas pueden ser, el estudio de los modos aeroelásticos de una aeronave, la interacción entre una aeronave y la estela de otra, factores de carga durante una maniobra, transitorios por el cambio en la geometría de un ala o elemento aerodinámico en general, caracterización de un rotor y su estela, entre otros.

Dentro de este campo los métodos de resolución numérica han cobrado gran relevancia al permitir el modelado y resolución de problemas complejos, previamente a la realización de ensayos experimentales costosos y sin recurrir a soluciones analíticas de extensa y laboriosa deducción, que además en la mayoría de los casos su aplicación queda limitada a representaciones de problemas físicos sencillos e ideales.

El método de los paneles, el cual es el tema de este trabajo, es una técnica ampliamente utilizada en la aerodinámica computacional para resolver problemas de flujo potencial sin la necesidad de una discretización del fluido en una malla, lo que la hace computacionalmente menos costosa que otros métodos. Además, esta técnica posee variaciones orientadas para flujo de bajas velocidades, transónicas y supersónicas. También, pueden contemplarse efectos viscosos con adaptaciones basadas en teorías de capa límite o consideraciones experimentales.

Esta herramienta propone un equilibrio entre fidelidad de los resultados y el costo computacional que resulta atractivo en muchas aplicaciones en las que pueda utilizarse. Por esto, se considera importante su estudio e implementación.

## Objetivos

En particular, en este trabajo se desarrollará el método de los paneles bidimensional, incompresible, no viscoso, estacionario y no estacionario, multielemento, de paneles rectos con distribuciones de dobletes constantes y vórtices puntuales y con la aplicación de la condición de Neumann.  

La bibliografía respecto al tema es vasta, pero muchas veces se encuentra en un formato el cual creo, personalmente, que no ofrece una conexión clara y completa entre la matemática resultante del método y una potencial implementación, lo que considero esencial para un primer acercamiento al método de los paneles.

Por primer acercamiento no quiero dar la impresión de "incompletitud", sino remarcar que la aplicación desarrollada abarca una pequeña fracción de la amplia familia de formulaciones del método de los paneles. Cada una de las variantes existentes presenta desafíos propios y está orientada a resolver problemas con características particulares, o similares pero con diferente desempeño numérico o diferentes capacidades algorítmicas. En este sentido, si emprendiera la tarea de implementar la mayoría de estas variantes con fines prácticos, considero que el punto de partida natural sería precisamente el que aborda este trabajo. Así, a partir de los conceptos adquiridos y lecciones aprendidas aquí, podrían desarrollarse las demás variantes con mayor solvencia.

Por lo tanto y concretamente, el presente trabajo tiene como objetivo el desarrollo de una aplicación computacional para la resolución numérica de flujos de la forma ya mencionada y, además, se busca asentar por escrito de manera clara y sintética los fundamentos físicos y matemáticos del método, de forma que el trabajo no conste solo de una herramienta de cálculo, sino también de una guía comprensible y estructurada del método, donde se disponga de forma completa y compatible con el código, lo que hizo falta resolver para producir una aplicación funcional, validada y con desempeño computacional aceptable; facilitando su reutilización y expansión en potenciales, variados e interesantes futuros desarrollos.

## Estructura del repositorio

En el presente repositorio se pone a disposición el documento escrito del trabajo final en cuestión y toda la implementación desarrollada, interfaz gráfica, ejemplos y demás, organizados como se muestra a continuación:
```
TrabajoFinal/
│
├── Trabajo Final - Bruno F. Barra Atarama.pdf   # Memoria escrita
│
├── src/                        # Código fuente del simulador
│   ├── MP2D.py                 # Clase principal para diseñar simulaciones, ejecutarlas y graficar
│   ├── Cinematicas.py          # Generadores de cinemáticas (prescritas, aeroelásticas y flap)
│   ├── Tipos.py                # Tipado y configuraciones auxiliares
│   ├── Importacion.py          # Importación y discretización de perfiles
│   ├── _TernasMoviles2D.py     # Módulos auxiliares
│   ├── _Vortices2D.py
│   ├── _ConjuntoSolidos2D.py
│   └── _Paneles2D.py
│
├── gui/
│   ├── GUI.py                  # Código fuente de la interfaz gráfica
│   ├── GUI.spec                # Archivo de especificaciones de PyInstaller
│   ├── logos.png               # Logos institucionales
│   ├── build/                  # Archivos temporales generados durante la compilación (no versionado)
│   └── dist/                   # (No versionado)
│       └── GUI.exe             # Ejecutable stand-alone de la interfaz gráfica (no versionado)
├── Ejemplos/                   # Notebooks de ejemplo de uso y validación
│   ├── ejemploBase.ipynb                   # Simulación de perfil fijo con ángulo de ataque variable
│   ├── ejemploAlaFlap.ipynb                # Caso ala-flap basado en NACA Report 614
│   ├── ejemploCargaYGuardado.ipynb         # Guardado/carga de simulaciones (.npz / .mat)
│   ├── ejemploCinematicaAeroelastica.ipynb # Cinemática aeroelástica
│   ├── ejemploWagner.ipynb                 # Respuesta impulsiva
│   ├── ejemploAnimacion.ipynb              # Generación de la animación del campo de velocidades
│   ├── animación.mp4                       # Animación del campo de velocidades
│   ├── testArmonico.ipynb                  # Oscilaciones armónicas, validación con NACA TN 2465
│   └── benchmarkTernasMoviles.ipynb        # Tests de performance
│
├── Ejemplos/rec/             # Datos auxiliares y resultados guardados
│   ├── NACATN2465/           # CSV de curvas experimentales del reporte
│   ├── perfiles/             # Coordenadas de perfiles en formato Selig/Lednicer
│   ├── *.npz / *.mat         # Ejemplos de simulaciones guardadas
│   └── *.png                 # Gráficos de Cp, polares, etc.
│
├── LICENSE
├── README.md
├── setup.py
└── requirements.txt
```

## Compatibilidad
Este proyecto fue desarrollado y probado exclusivamente en *Windows 11*.

No se garantiza el correcto funcionamiento ni la compatibilidad en otros sistemas operativos.

## Instalación
1. Clonar el repositorio.
2. Crear y activar el entorno virtual de *Python 3.11+*.
3. Instalar dependencias. Para ello en una terminal del entorno virtual ejecutar:
    - `pip install -r requirements.txt`. Esto instala las bibliotecas requeridas.
    - `pip install .`. Esto instala el programa del método de los paneles. Para realizar modificaciones en el código, en su lugar se recomienda utilizar una instalación en modo editable con `pip install -e .`. 

## Interfaz gráfica (*GUI*)
La interfaz gráfica se dispone como el ejecutable `GUI.exe` en los **Releases** en la barra lateral del repositorio, listo para ser descargado.

### Compilación
También, la *GUI* puede ejecutarse directamente desde el entorno *Python*, o compilarse como el ejecutable `GUI.exe` usando *PyInstaller*. 

Esto genera el archivo `GUI.exe` dentro de `gui/dist/`, con los recursos embebidos (logos, licencia, etc.).

La compilación puede realizarse de las siguientes formas, en una terminal del entorno virtual:

- Sin consola:
    ```cmd
    pyinstaller gui/GUI.py --onefile --noconsole ^
      --distpath gui/dist ^
      --workpath gui/build ^
      --specpath gui ^
      --add-data "%CD%\gui\logos.png;." ^
      --add-data "%CD%\LICENSE;."
    ```
- Con consola (útil para depurar):
    ```cmd
    pyinstaller gui/GUI.py --onefile --console ^
      --distpath gui/dist ^
      --workpath gui/build ^
      --specpath gui ^
      --add-data "%CD%\gui\logos.png;." ^
      --add-data "%CD%\LICENSE;."
    ```
## Cómo citar
Use el botón **Cite this repository** en la barra lateral del repositorio.

## Correcciones y contribuciones
¿Encontraste un error en la documentación, figuras o resultados?
Escribime a **bfbarraatarama@gmail.com** con asunto **MP2D: corrección**.