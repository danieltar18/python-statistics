import pandas as pd
import numpy as np
import requests
from statadict import parse_stata_dict

def download(url, path="datasets"):
    content = requests.get(url).content

    with open(f"{path + url.split('/')[-1]}", mode="wb") as file:
        file.write(content)

def read_nsfg_data(dct_file="datasets/nsfg/2002FemPreg.dct", dat_file="datasets/nsfg/2002FemPreg.dat.gz"):

    stata_dict = parse_stata_dict(dct_file)
    resp = pd.read_fwf(
        dat_file,
        names=stata_dict.names,
        colspecs=stata_dict.colspecs,
        compression="gzip",
    )
    return resp

def clean_fem_preg(df):
    """Recodes variables from the pregnancy frame.

    df: DataFrame
    """
    df.agepreg /= 100.0
    df.loc[df.birthwgt_lb > 20, "birthwgt_lb"] = np.nan
    na_vals = [97, 98, 99]
    df["birthwgt_lb"] = df.birthwgt_lb.replace(na_vals, np.nan)
    df["birthwgt_oz"] = df.birthwgt_oz.replace(na_vals, np.nan)
    df["totalwgt_lb"] = df.birthwgt_lb + df.birthwgt_oz / 16.0

    df["hpagelb"] = df.hpagelb.replace(na_vals, np.nan)
    df["babysex"] = df.babysex.replace([7, 9], np.nan)
    df["nbrnaliv"] = df.nbrnaliv.replace([9], np.nan)


def prep_nsfg_data():
    """Reads the NSFG pregnancy data.

    dct_file: string file name
    dat_file: string file name

    returns: DataFrame
    """
    preg = read_nsfg_data()
    clean_fem_preg(preg)
    return preg


## Female Resp

def read_fem_resp(dct_file="datasets/nsfg/2002FemResp.dct", dat_file="datasets/nsfg/2002FemResp.dat.gz"):
    """Read the 2002 NSFG respondent file.

    dct_file: string file name
    dat_file: string file name

    returns: DataFrame
    """
    resp =  read_nsfg_data(dct_file, dat_file)
    clean_fem_resp(resp)
    return resp


def clean_fem_resp(resp):
    """Cleans a respondent DataFrame.

    resp: DataFrame of respondents

    Adds columns: agemarry, age, decade, fives
    """
    resp["cmmarrhx"] = resp.cmmarrhx.replace([9997, 9998, 9999], np.nan)
    resp["agemarry"] = (resp.cmmarrhx - resp.cmbirth) / 12.0
    resp["age"] = (resp.cmintvw - resp.cmbirth) / 12.0
    month0 = pd.to_datetime("1899-12-15")
    dates = [(month0 + pd.DateOffset(months=cm)) for cm in resp.cmbirth]
    resp["year"] = pd.DatetimeIndex(dates).year - 1900
    resp["decade"] = resp.year // 10
    resp["fives"] = resp.year // 5


def read_fem_preg(dct_file="datasets/nsfg/2002FemPreg.dct", dat_file="datasets/nsfg/2002FemPreg.dat.gz"):
    """Reads the NSFG pregnancy data.

    dct_file: string file name
    dat_file: string file name

    returns: DataFrame
    """
    preg = read_nsfg_data(dct_file, dat_file)
    clean_fem_preg(preg)
    return preg

def get_nsfg_groups():
    """Read the NSFG pregnancy file and split into groups.
    
    returns: all live births, first babies, other babies
    """
    preg = read_fem_preg()
    live = preg.query("outcome == 1")
    firsts = live.query("birthord == 1")
    others = live.query("birthord != 1")
    return live, firsts, others