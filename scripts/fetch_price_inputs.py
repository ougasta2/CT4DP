"""Retrieve the publicly distributed EnergyZero/Gasradar gas-price proxy."""
from pathlib import Path
from fetch_forecast_inputs import get


if __name__=='__main__':
    get('https://gasradar.nl/data/gasprijs-per-dag.csv',{},Path('data/external/gas/gasradar_daily.csv.gz'))
