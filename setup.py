from setuptools import setup

setup(
    name="TrabajoFinal",      # nombre de la distribución pip
    version="0.1",
    packages=["src"],         # sólo publico el paquete "src"
    package_dir={             # mapeo: el paquete "src" vive en ./src
        "src": "src"
    },
    python_requires=">=3.11, <=3.13"
)