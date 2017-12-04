# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-locals, too-many-arguments
from __future__ import absolute_import

import ctypes
import copy
import numpy as np
from .base import _LIB
from .base import check_call
from .base import ctypes2buffer

def read_profile():
    '''
    Parameters
    ----------
    None

    Returns
    -------
    buffer : bytearray
        The raw byte memory buffer
    '''
    length = ctypes.c_size_t()
    cptr = ctypes.POINTER(ctypes.c_char)()
    check_call(_LIB.MXReadDFGEProfile(ctypes.byref(length), ctypes.byref(cptr)))
    return ctypes2buffer(cptr, length.value)
