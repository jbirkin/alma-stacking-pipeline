import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM, Planck18

import warnings
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning, append=True)

# ------------------------------------------------------------------------------------------------------------

nu_cii = 1900.53690000 # frequency of [CII] in GHz
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

cristal_dir = "/Users/jbirkin/Documents/research/CRISTAL/"  # cristal main directory
cube_dir = cristal_dir+"data/cubes/natural/"               # cube directory
ctab_file = cristal_dir+"tables/cristal_table.fits"         # file in which main table is stored
ctab = Table.read(ctab_file)                                # load the main table as an astropy Table
ctab = ctab[ctab["Use?"]]                                   # remove CRISTAL-18 from table

alpine_dir = "/Users/jbirkin/Documents/research/ALPINE/"
alptab = Table.read(alpine_dir+"ALPINE_merged_catalogs.fits")
# alptab = alptab[alptab["g20_stack?"]]
alp_cube_dir =  "/Users/jbirkin/Data/alma/alpine/raw_cubes/share.lam.fr/files/ALMA_DATA/DR1/raw_cubes/"

# ------------------------------------------------------------------------------------------------------------