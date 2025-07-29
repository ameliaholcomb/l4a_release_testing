import pandas as pd
import numpy as np

def is_all_null_numeric(data):
    kind = data.dtype.kind
    if kind in "iu":  # integer or unsigned integer
        return (data == np.iinfo(data.dtype).max).all()
    if kind in "f":  # float
        isnull = (data == np.finfo(data.dtype).max).all()
        isnull |= (data == -9999.0).all()
        return isnull

def is_all_null_str(data):
    if (data == "").all():
        return True
    if (data == "None").all():
        return True
    if (data == "NaN").all():
        return True
    else:
        return False

def check_null(data: pd.Series):
    kind = data.dtype.kind
    if kind in "b":  # boolean
        return False  # Booleans are not considered null
    elif kind in "iuf": # integer, unsigned integer, float
        return is_all_null_numeric(data)
    elif kind in "O":  # object
        s = data.values[0]
        if type(s) is np.ndarray:
            if s.dtype.kind in "iuf":
                return is_all_null_numeric(np.array(data.to_list()))
            else:
                raise TypeError(f"Unknown type for {data.name}: {s.dtype.kind}")
        elif type(s) is bytes:
            return is_all_null_str(data.str.decode("utf-8"))
        elif type(s) is str:
            return is_all_null_str(data)
        else:
            raise TypeError(f"Unknown type for {data.name}: {type(s)}")
    else:
        raise TypeError(f"Unknown type for {data.name}: {kind}")
    

def check_zero(data: pd.Series):
    kind = data.dtype.kind
    if kind in "iuf":
        if (data == 0).all():
            return True
    if kind in "O":  # object
        if type(data.values[0]) is np.ndarray:
            if (np.array(data.to_list()) == 0).all():
                return True
    if kind in "b":  # boolean
        if not data.any():
            return True
    return False