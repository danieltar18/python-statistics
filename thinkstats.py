import bisect
import contextlib
import re
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy

from empiricaldist import Hist, Pmf, Cdf, Hazard

from scipy.stats import norm
from scipy.stats import gaussian_kde
from scipy.integrate import simpson

from IPython.display import display
from statsmodels.iolib.table import SimpleTable

# The following are magic commands from thinkpython.py

from IPython.core.magic import register_cell_magic
from IPython.core.magic_arguments import (
    argument,
    magic_arguments,
    parse_argstring,
)

def read_brfss(
    filename="datasets/brfss/CDBRFS08.ASC.gz", compression="gzip", nrows=None
):
    """Reads the BRFSS data.

    filename: string
    compression: string
    nrows: int number of rows to read, or None for all

    returns: DataFrame
    """
    # column names and column specs from
    # https://www.cdc.gov/brfss/annual_data/2008/varLayout_table_08.htm
    var_info = [
        ("age", 100, 102, int),
        ("sex", 142, 143, int),
        ("wtyrago", 126, 130, int),
        ("finalwt", 798, 808, int),
        ("wtkg2", 1253, 1258, int),
        ("htm3", 1250, 1253, int),
    ]
    columns = ["name", "start", "end", "type"]
    variables = pd.DataFrame(var_info, columns=columns)

    colspecs = variables[["start", "end"]].values.tolist()
    names = variables["name"].tolist()

    df = pd.read_fwf(
        filename,
        colspecs=colspecs,
        names=names,
        compression=compression,
        nrows=nrows,
    )

    clean_brfss(df)
    return df


def clean_brfss(df):
    """Recodes BRFSS variables.

    df: DataFrame
    """
    df["age"] = df["age"].replace([7, 9], np.nan)
    df["htm3"] = df["htm3"].replace([999], np.nan)
    df["wtkg2"] = df["wtkg2"].replace([99999], np.nan) / 100
    df["wtyrago"] = df.wtyrago.replace([7777, 9999], np.nan)
    df["wtyrago"] = df.wtyrago.apply(
        lambda x: x / 2.2 if x < 9000 else x - 9000
    )