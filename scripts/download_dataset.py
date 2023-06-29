#!/usr/bin/env python3
"""It allows downloading the data set to be used.
"""

import os
import zipfile


def download_covid19_dataset():
    """Download the data set to use and extract the data into the data folder.
    """
    
    os.system("wget --no-check-certificate " + \
              "https://datosabiertos.salud.gob.mx/gobmx/salud/datos_abiertos/datos_abiertos_covid19.zip " + \
              "-O ../data/datos_abiertos_covid19.zip")
    
    zip_file = "../data/datos_abiertos_covid19.zip"
    zip_ref = zipfile.ZipFile(zip_file, "r")
    zip_ref.extractall("../data/")
    zip_ref.close()
    os.remove(zip_file)

    print("Done! Dataset saved in ../data/")


if __name__ == "__main__":
    download_covid19_dataset()
