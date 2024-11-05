work_dir = "../"
cube_dir = work_dir+"data/cubes/natural/"

cristal_tab_file = work_dir+"data/cristal_table.fits"         # file in which main table is stored
cristal_tab = Table.read(cristal_tab_file)                    # load the main table as an astropy Table

nu_cii = 1900.53690000              # frequency of [CII] in GHz