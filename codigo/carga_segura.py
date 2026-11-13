"""Carga de .npy de objeto (diccionarios de idtracker.ai) sin ejecutar código ajeno.

`np.load(..., allow_pickle=True)` deserializa con pickle, que puede ejecutar código. Aquí se
lee la cabecera .npy y se deserializa con un Unpickler que sólo admite las clases de numpy que
usan esos archivos (verificado con pickletools sobre los datos de ref. Romero-Ferrero et al.
2023); cualquier otra clase hace fallar la carga.
"""
from __future__ import annotations

import io
import pickle

import numpy as np
import numpy.lib.format as fmt

try:
    from numpy._core import multiarray as _ma
except ImportError:                                    # numpy < 2
    from numpy.core import multiarray as _ma

PERMITIDAS = {
    ("numpy.core.multiarray", "_reconstruct"): _ma._reconstruct,
    ("numpy._core.multiarray", "_reconstruct"): _ma._reconstruct,
    ("numpy.core.multiarray", "scalar"): _ma.scalar,
    ("numpy._core.multiarray", "scalar"): _ma.scalar,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
}


class _Restringido(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return PERMITIDAS[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"clase no permitida: {module}.{name}") from None


def cargar(ruta):
    with open(ruta, "rb") as fh:
        fmt.read_magic(fh)
        fmt.read_array_header_1_0(fh)
        obj = _Restringido(io.BytesIO(fh.read()), encoding="latin1").load()
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        obj = obj.item()
    return obj
