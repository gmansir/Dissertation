import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from scipy.optimize import curve_fit

class FunctionsDiskIntegrate:

    def __init__(self, target):
        # Store the object name (e.g., 'titan', 'neptune1') for later use
        self.object = target

        # Dictionary containing file paths for all science targets and calibrators
        # Each entry includes:
        #  - UVB, VIS, NIR: paths to input spectral data cubes for each arm
        #  - save_UVB, save_VIS, save_NIR: output paths for the disk-integrated spectra
        #  - range_UVB, range_VIS, range_NIR: tuple indices indicating wavelength slices to use
        all_paths = {
                'titan1': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_1\\MOV_Titan_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_1\\MOV_Titan_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_1\\MOV_Titan_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_1\\MOV_Titan_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_1\\MOV_Titan_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_1\\MOV_Titan_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'titan2': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_2\\MOV_Titan_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_2\\MOV_Titan_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_2\\MOV_Titan_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_2\\MOV_Titan_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_2\\MOV_Titan_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_2\\MOV_Titan_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'titan3': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_3\\MOV_Titan_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_3\\MOV_Titan_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_3\\MOV_Titan_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_3\\MOV_Titan_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_3\\MOV_Titan_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_3\\MOV_Titan_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'neptune1': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_1\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_1\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_1\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_1\\MOV_Neptune_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_1\\MOV_Neptune_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_1\\MOV_Neptune_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'neptune2': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_2\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_2\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_2\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_2\\MOV_Neptune_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_2\\MOV_Neptune_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_2\\MOV_Neptune_DiskIntegrated_NIR.fits",
                    'range_UVB': (350, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'neptune3': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_3\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_3\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_3\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_3\\MOV_Neptune_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_3\\MOV_Neptune_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_3\\MOV_Neptune_DiskIntegrated_NIR.fits",
                    'range_UVB': (350, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'neptune4': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_4\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_4\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_4\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_4\\MOV_Neptune_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_4\\MOV_Neptune_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_4\\MOV_Neptune_DiskIntegrated_NIR.fits",
                    'range_UVB': (350, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'feige-110': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\FEIGE-110\\FEIGE-110_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\FEIGE-110\\FEIGE-110_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\FEIGE-110\\FEIGE-110_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\FEIGE-110\\MOV_FEIGE-110_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\FEIGE-110\\MOV_FEIGE-110_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\FEIGE-110\\MOV_FEIGE-110_DiskIntegrated_NIR.fits",
                    'range_UVB': (350, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'uranus1': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_1\\MOV_Uranus_2_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_1\\MOV_Uranus_2_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_1\\MOV_Uranus_2_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_1\\MOV_Uranus_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_1\\MOV_Uranus_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_1\\MOV_Uranus_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'uranus2': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_2\\MOV_Uranus_2_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_2\\MOV_Uranus_2_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_2\\MOV_Uranus_2_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_2\\MOV_Uranus_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_2\\MOV_Uranus_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_2\\MOV_Uranus_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'uranus3': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_3\\MOV_Uranus_2_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_3\\MOV_Uranus_2_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_3\\MOV_Uranus_2_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_3\\MOV_Uranus_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_3\\MOV_Uranus_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_3\\MOV_Uranus_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'uranus4': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_4\\MOV_Uranus_2_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_4\\MOV_Uranus_2_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_4\\MOV_Uranus_2_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_4\\MOV_Uranus_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_4\\MOV_Uranus_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_4\\MOV_Uranus_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'gd71': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\GD71\\GD71_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\GD71\\GD71_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\GD71\\GD71_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\GD71\\MOV_GD71_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\GD71\\MOV_GD71_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\GD71\\MOV_GD71_DiskIntegrated_NIR.fits",
                    'range_UVB': (350, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'saturn1': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_1\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_1\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_1\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_1\\MOV_Saturn_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_1\\MOV_Saturn_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_1\\MOV_Saturn_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'saturn3': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_3\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_3\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_3\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_3\\MOV_Saturn_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_3\\MOV_Saturn_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_3\\MOV_Saturn_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'saturn4': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_4\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_4\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_4\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_4\\MOV_Saturn_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_4\\MOV_Saturn_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_4\\MOV_Saturn_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'saturn5': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_5\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_5\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_5\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_5\\MOV_Saturn_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_5\\MOV_Saturn_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_5\\MOV_Saturn_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'saturn6': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_6\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_6\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_6\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_6\\MOV_Saturn_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_6\\MOV_Saturn_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_6\\MOV_Saturn_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'saturn7': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_7\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_7\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_7\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_7\\MOV_Saturn_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_7\\MOV_Saturn_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_7\\MOV_Saturn_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'saturn8': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_8\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_8\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_8\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_8\\MOV_Saturn_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_8\\MOV_Saturn_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_8\\MOV_Saturn_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'saturn9': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_9\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_9\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_9\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_9\\MOV_Saturn_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_9\\MOV_Saturn_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_9\\MOV_Saturn_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'saturn10': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_10\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_10\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_10\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_10\\MOV_Saturn_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_10\\MOV_Saturn_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_10\\MOV_Saturn_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'saturn11': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_11\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_11\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_11\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_11\\MOV_Saturn_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_11\\MOV_Saturn_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_11\\MOV_Saturn_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'saturn12': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_12\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_12\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_12\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_12\\MOV_Saturn_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_12\\MOV_Saturn_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_12\\MOV_Saturn_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'saturn13': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_13\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_13\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_13\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_13\\MOV_Saturn_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_13\\MOV_Saturn_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_13\\MOV_Saturn_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'saturn14': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_14\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_14\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_14\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_14\\MOV_Saturn_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_14\\MOV_Saturn_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_14\\MOV_Saturn_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'saturn16': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_16\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_16\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_16\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_16\\MOV_Saturn_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_16\\MOV_Saturn_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_16\\MOV_Saturn_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'saturn17': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_17\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_17\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_17\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_17\\MOV_Saturn_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_17\\MOV_Saturn_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_17\\MOV_Saturn_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'saturn18': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_18\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_18\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_18\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_18\\MOV_Saturn_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_18\\MOV_Saturn_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_18\\MOV_Saturn_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'saturn19': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_19\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_19\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_19\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_19\\MOV_Saturn_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_19\\MOV_Saturn_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_19\\MOV_Saturn_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'saturn20': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_20\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_20\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_20\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_20\\MOV_Saturn_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_20\\MOV_Saturn_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_20\\MOV_Saturn_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'saturn21': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_21\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_21\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_21\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_21\\MOV_Saturn_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_21\\MOV_Saturn_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_21\\MOV_Saturn_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'saturn22': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_22\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_22\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_22\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_22\\MOV_Saturn_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_22\\MOV_Saturn_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_22\\MOV_Saturn_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'saturn23': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_23\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_23\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_23\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_23\\MOV_Saturn_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_23\\MOV_Saturn_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_23\\MOV_Saturn_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'saturn24': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_24\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_24\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_24\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_24\\MOV_Saturn_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_24\\MOV_Saturn_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_24\\MOV_Saturn_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'saturn25': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_25\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_25\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_25\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_25\\MOV_Saturn_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_25\\MOV_Saturn_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_25\\MOV_Saturn_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'saturn26': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_26\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_26\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_26\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_26\\MOV_Saturn_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_26\\MOV_Saturn_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_26\\MOV_Saturn_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'ltt7987': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\LTT7987\\LTT7987_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\LTT7987\\LTT7987_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\LTT7987\\LTT7987_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\LTT7987\\MOV_LTT7987_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\LTT7987\\MOV_LTT7987_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\LTT7987\\MOV_LTT7987_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'hip09': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\LTT7987\\LTT7987_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Standard2\\Hip095318_TELL_IFU_TELL_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Standard2\\Hip095318_TELL_IFU_TELL_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Standard2\\MOV_Hip095318_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Standard2\\MOV_Hip095318_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Standard2\\MOV_Hip095318_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'pluto1': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_1\\MOV_Pluto_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_1\\MOV_Pluto_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_1\\MOV_Pluto_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_1\\MOV_Pluto_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_1\\MOV_Pluto_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_1\\MOV_Pluto_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'pluto2': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_2\\MOV_Pluto_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_2\\MOV_Pluto_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_2\\MOV_Pluto_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_2\\MOV_Pluto_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_2\\MOV_Pluto_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_2\\MOV_Pluto_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'pluto3': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_3\\MOV_Pluto_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_3\\MOV_Pluto_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_3\\MOV_Pluto_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_3\\MOV_Pluto_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_3\\MOV_Pluto_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_3\\MOV_Pluto_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'pluto4': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_4\\MOV_Pluto_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_4\\MOV_Pluto_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_4\\MOV_Pluto_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_4\\MOV_Pluto_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_4\\MOV_Pluto_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_4\\MOV_Pluto_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'pluto5': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_5\\MOV_Pluto_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_5\\MOV_Pluto_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_5\\MOV_Pluto_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_5\\MOV_Pluto_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_5\\MOV_Pluto_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_5\\MOV_Pluto_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'pluto6': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_6\\MOV_Pluto_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_6\\MOV_Pluto_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_6\\MOV_Pluto_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_6\\MOV_Pluto_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_6\\MOV_Pluto_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_6\\MOV_Pluto_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'pluto7': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_7\\MOV_Pluto_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_7\\MOV_Pluto_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_7\\MOV_Pluto_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_7\\MOV_Pluto_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_7\\MOV_Pluto_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_7\\MOV_Pluto_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'pluto8': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_8\\MOV_Pluto_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_8\\MOV_Pluto_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_8\\MOV_Pluto_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_8\\MOV_Pluto_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_8\\MOV_Pluto_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\Offset_8\\MOV_Pluto_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
                'feige-110_p1': {
                    'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\FEIGE-110\\FEIGE-110_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                    'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\FEIGE-110\\FEIGE-110_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                    'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\FEIGE-110\\FEIGE-110_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                    'save_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\FEIGE-110\\MOV_Feige-110_DiskIntegrated_UVB.fits",
                    'save_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\FEIGE-110\\MOV_Feige-110_DiskIntegrated_VIS.fits",
                    'save_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Pluto\\Night_1\\FEIGE-110\\MOV_Feige-110_DiskIntegrated_NIR.fits",
                    'range_UVB': (320, 550),
                    'range_VIS': (600, 950),
                    'range_NIR': (1100, 2300),
                },
        }

        # Store only the paths corresponding to the selected object
        self.file_paths = all_paths[target]

        self.header = None
        self.wave = np.array([])
        self.data = np.array([])
        self.errs = np.array([])
        self.qual = np.array([])
        self.medians = np.array([])
        self.adc_data = np.array([])
        self.sig_clip_data = np.array([])
        self.naxis = int()
        self.pix_x = int()
        self.pix_y = int()
        self.dx = float()
        self.dy = float()

    @staticmethod
    def closest_index(x: float, arr: np.ndarray) -> int:
        """
        Returns the index of the element in `arr` that is closest to the target value `x`.

        Args:
            x (float): Target value to match.
            arr (array): Input array in which to find the closest value.

        Returns:
            index (int): Index of the element in `arr` that is closest to `x`.
        """

        # Assert type and check emptiness
        arr = np.asarray(arr)
        if len(arr) == 0:
            raise ValueError("Input array is empty.")

        # Compute the absolute difference between each element in arr and the target value x
        difference_array = np.absolute(arr - x)

        # Find the index of the minimum difference, i.e., the closest value to x
        index = difference_array.argmin()

        return index

    def extract_info(self, cube_path, plot_save_loc=None, plots=True):
        """
        Provides a quick visualization of the data, allowing you to check the dimensions
        of the slit, data quality, and object center. Also returns important parameters.

        Args:
            cube_path (str): Path to the data cube.
            plot_save_loc (str or None): Directory where plots will be saved, if specified.
            plots (bool): Whether to display and/or save a median-collapsed image of the cube.

        Returns:
            data (3D array): The raw data of the data cube.
            wave (array): An array representing the wavelength range of the data.
            pix_x (int): Number of pixels in the x-direction of the slit.
            pix_y (int): Number of pixels in the y-direction of the slit.
            dx (float): Arcseconds between pixels in the x-direction of the slit.
            dy (float): Arcseconds between pixels in the y-direction of the slit.
        """

        # Resolve the full path to the FITS cube and open it
        # Load the 3D data cube (shape: [wavelength, y, x])
        obs = os.path.abspath(cube_path)
        with fits.open(obs) as hdul:
            self.header = hdul[0].header
            self.data = fits.getdata(obs, ext=0)
            self.errs = fits.getdata(obs, ext=1)
            self.qual = fits.getdata(obs, ext=2)

        if self.data.ndim != 3:
            raise ValueError("Provided data must be a 3D array.")

        # Extract spatial and spectral dimensions from the header
        self.naxis = self.header["NAXIS3"]  # Number of spectral layers (wavelength dimension)
        self.pix_x = int(self.header["NAXIS1"])  # Spatial x-dimension (e.g., along-slit direction)
        self.pix_y = int(self.header["NAXIS2"])  # Spatial y-dimension (e.g., across-slit direction)

        # Convert pixel scale from degrees to arcseconds (assumes square pixels)
        self.dx = np.abs(self.header["CDELT1"]) * 3600  # arcsec/pixel along x
        self.dy = np.abs(self.header["CDELT2"]) * 3600  # arcsec/pixel along y

        # Construct wavelength array using standard WCS linear solution
        self.wave = np.zeros(self.naxis)
        for i in range(self.naxis):
            self.wave[i] = (i + self.header["CRPIX3"]) * self.header["CDELT3"] + self.header["CRVAL3"]

        # Compute the median spectrum value at each (x, y) spaxel
        self.medians = np.zeros((self.pix_y, self.pix_x))
        for i in range(self.pix_x):
            for j in range(self.pix_y):
                self.medians[j, i] = np.nanpercentile(self.data[:, j, i], 50)

        # Optional: visualize the median-collapsed image
        if plots:
            fig, axes = plt.subplots(1, 1, figsize=(6, 10))
            im = axes.imshow(self.medians, origin="lower", aspect=self.dy / self.dx)  # correct aspect ratio
            axes.set_title("Visualization of the data cube")
            axes.set_ylabel("y-spaxels")
            axes.set_xlabel("x-spaxels")

            # Add colorbar with label
            cbar = plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
            cbar.set_label("Median Flux", rotation=270, labelpad=15)

            # Save plot if output path is given
            if plot_save_loc is not None:
                plt.savefig(plot_save_loc + "/datacube_visualization.png")

        # Print cube size to console for quick diagnostics
        print(" ")
        print("Size of the data: (", self.naxis, ", ", self.pix_y, ", ", self.pix_x, ")")

    def atmospheric_dispersion_correction(
            self,
            center_x=True,
            center_y=True,
            range_x=None,
            range_y=None,
            plots=True,
            plot_save_loc=None):
        """
        Applies atmospheric dispersion correction to a 3D data cube by tracking
        the apparent movement of the object's center with wavelength.

        Also propagates the variance cube (self.errs) through the same resampling:
        - Integer shifts: copy variance from source pixel
        - Subpixel interpolation: var' = w0^2*var0 + w1^2*var1
        - Optional QUAL handling: drop flagged neighbors if present

        Returns (assigned to attributes):
            self.adc_data : flux cube after ADC
            self.adc_var  : variance cube after ADC
            (plus the usual center + normalized slices for QC)
        """

        # --- Basic checks on variance + optional QUAL ---
        if not hasattr(self, "errs"):
            raise AttributeError("Expected self.errs to be a variance cube of the same shape as self.data.")
        if self.errs.shape != self.data.shape:
            raise ValueError("self.errs (variance cube) must match self.data shape.")

        have_qual = hasattr(self, "qual") and (self.qual is not None) and (self.qual.shape == self.data.shape)

        # Convenience: local views
        flux_cube = self.data
        var_cube = self.errs  # assumed to be variance

        # --- Compute x-centroid of flux at each wavelength slice ---
        x_coords_center = np.zeros(self.naxis)
        for l in range(self.naxis):
            weights = np.sum(flux_cube[l, :, :], axis=0)  # sum over y → (pix_x,)
            total_flux = np.sum(weights)
            x_coords_center[l] = np.sum(np.arange(self.pix_x) * weights) / total_flux if total_flux != 0 else np.nan

        if not center_x:
            x_coords_center[:] = np.nanmedian(x_coords_center)

        # --- Compute y-centroid of flux at each wavelength slice ---
        y_coords_center = np.zeros(self.naxis)
        for l in range(self.naxis):
            weights = np.sum(flux_cube[l, :, :], axis=1)  # sum over x → (pix_y,)
            total_flux = np.sum(weights)
            y_coords_center[l] = np.sum(np.arange(self.pix_y) * weights) / total_flux if total_flux != 0 else np.nan

        if not center_y:
            y_coords_center[:] = np.nanmedian(y_coords_center)

        # --- Parabola model + fit helpers ---
        def parabola(x, a, b, c):
            return a * x ** 2 + b * x + c

        def prepare_fit_data(coords_center, fit_range, wave):
            if fit_range is None:
                fit_x = wave
                fit_y = coords_center - coords_center[0]
            else:
                lo = FunctionsDiskIntegrate.closest_index(fit_range[0], wave)
                hi = FunctionsDiskIntegrate.closest_index(fit_range[1], wave)
                fit_x = wave[lo:hi]
                fit_y = coords_center[lo:hi] - coords_center[0]
            return fit_x, fit_y

        # Fits
        y_fit_x, y_fit_y = prepare_fit_data(y_coords_center, range_y, self.wave)
        popt_y, _ = curve_fit(parabola, y_fit_x, y_fit_y)

        x_fit_x, x_fit_y = prepare_fit_data(x_coords_center, range_x, self.wave)
        popt_x, _ = curve_fit(parabola, x_fit_x, x_fit_y)

        # Plot the spatial center movement over wavelength in both y and x directions
        if plots:
            # Set the figure aspect ratio (wider than tall) and initialize subplots
            w, h = plt.figaspect(0.8)
            fig, axes = plt.subplots(2, 1, figsize=(w, h), sharex=True)

            # Reduce vertical spacing between plots
            plt.subplots_adjust(hspace=0)

            # --- Y-direction plot ---
            # Plot the measured center position in y-spaxels
            axes[0].plot(self.wave / 1000, y_coords_center, linewidth=0.3, color="k", alpha=1,
                         label='center in y-spaxels')

            # Plot the best-fit parabola (smooth, continuous curve)
            axes[0].plot(self.wave / 1000, y_coords_center[0] + parabola(self.wave, *popt_y),
                         color="cornflowerblue", label='parabola')

            # Plot the rounded version of the parabola (discrete pixel positions)
            axes[0].plot(self.wave / 1000, y_coords_center[0] + np.round(parabola(self.wave, *popt_y)),
                         linestyle=":", color="rebeccapurple", label='rounded parabola', zorder=10)

            axes[0].set_ylim(5, 15)  # Fixed y-range for clearer view
            axes[0].set_ylabel("y-spaxels", fontsize=14)
            axes[0].legend(fontsize=8, loc='lower left')
            axes[0].grid(False)

            # --- X-direction plot ---
            axes[1].plot(self.wave / 1000, x_coords_center, linewidth=0.3, color="k", alpha=1,
                         label='center in x-spaxels')
            axes[1].plot(self.wave / 1000, x_coords_center[0] + parabola(self.wave, *popt_x),
                         color="cornflowerblue", label='parabola')
            axes[1].plot(self.wave / 1000, x_coords_center[0] + np.round(parabola(self.wave, *popt_x)),
                         linestyle=":", color="rebeccapurple", label='rounded parabola', zorder=10)

            axes[1].set_ylim(0.75, 2)  # Fixed x-range for clearer view
            axes[1].set_xlabel("Wavelength (µm)", fontsize=14)
            axes[1].set_ylabel("x-spaxels", fontsize=14)
            axes[1].legend(fontsize=8, loc='upper left')
            axes[1].grid(False)

            # Save the plot if an output location is provided
            if plot_save_loc is not None:
                plt.savefig(plot_save_loc + "/CenterMovement.png")

            plt.show()

        def compute_centering(center_flag, coords_center, popt, wave, pix_limit, axis_name):
            if center_flag:
                fit = coords_center[0] + parabola(wave, *popt)
                fit_rounded = np.round(fit)
                oob = (fit_rounded > pix_limit - 1) | (fit_rounded < 0)
                if np.any(oob):
                    print(f"WARNING: Object displacement exceeds slit in {axis_name}; clipping.")
                    mov = np.clip(fit_rounded, 0, pix_limit - 1)
                    mov_float = np.clip(fit, 0, pix_limit - 1)
                else:
                    mov = fit_rounded
                    mov_float = fit
            else:
                mov = np.round(coords_center)
                mov_float = coords_center.copy()

            variation = mov_float - mov  # fractional offset in [-0.5, 0.5] typical
            dif_center = int(float(np.max(mov)) - float(np.min(mov)))
            long_pix = int(pix_limit + dif_center)
            offset = mov - mov[0]
            return mov, mov_float, variation, dif_center, long_pix, offset

        # --- CENTERING OFFSETS ---
        y_mov, y_mov_float, y_variation, y_dif_center, long_y, y_offset = compute_centering(
            center_y, y_coords_center, popt_y, self.wave, self.pix_y, "Y"
        )
        x_mov, x_mov_float, x_variation, x_dif_center, long_x, x_offset = compute_centering(
            center_x, x_coords_center, popt_x, self.wave, self.pix_x, "X"
        )

        # --- Helper for linear interpolation of flux + variance with optional QUAL ---
        def mix_flux_var(f0, v0, f1, v1, w1, good0=0, good1=0):
            """
            Combine two neighbors with weight w1 for the 'right/upper' neighbor.
            Flux:  f = (1-w1)*f0 + w1*f1
            Var:   v = (1-w1)^2*v0 + w1^2*v1
            Qual:  bitwise OR of contributing QUAL flags
            """

            w0 = 1.0 - w1
            f = w0 * f0 + w1 * f1
            v = (w0 * w0) * v0 + (w1 * w1) * v1
            q = good0 | good1  # combine flags
            return f, v, q

        if have_qual:
            qual_cube = self.qual
        else:
            qual_cube = np.zeros_like(self.data, dtype=np.uint16)  # no flags initially

        # ========================
        #  Y re-sampling (flux + var)
        # ========================
        if center_y:
            new_long_y = int(long_y - 2 * y_dif_center)
            corrected_ydata = np.zeros((self.naxis, new_long_y, self.pix_x), dtype=flux_cube.dtype)
            corrected_yvar = np.zeros((self.naxis, new_long_y, self.pix_x), dtype=var_cube.dtype)
            corrected_yqual = np.zeros((self.naxis, new_long_y, self.pix_x), dtype=qual_cube.dtype)

            for i in range(self.pix_x):
                for j in range(new_long_y):
                    for l in range(self.naxis):
                        movement = int(y_dif_center + y_offset[l])
                        base_idx = j + movement
                        frac = float(y_variation[l])

                        # Clip base_idx to valid range
                        base_idx = np.clip(base_idx, 0, self.pix_y - 1)

                        if frac > 0:
                            neighbor_idx = base_idx + 1
                            if neighbor_idx >= self.pix_y:
                                # edge: use base only
                                f0 = flux_cube[l, base_idx, i]
                                v0 = var_cube[l, base_idx, i]
                                q0 = qual_cube[l, base_idx, i]
                            else:
                                f0 = flux_cube[l, base_idx, i]
                                v0 = var_cube[l, base_idx, i]
                                f1 = flux_cube[l, neighbor_idx, i]
                                v1 = var_cube[l, neighbor_idx, i]
                                if have_qual:
                                    good0 = (self.qual[l, base_idx, i] == 0)
                                    good1 = (self.qual[l, neighbor_idx, i] == 0)
                                else:
                                    good0 = good1 = True
                                f0, v0, q0 = mix_flux_var(f0, v0, f1, v1, w1=frac, good0=good0, good1=good1)
                            corrected_ydata[l, j, i] = f0
                            corrected_yvar[l, j, i] = v0
                            corrected_yqual[l,j,i] = q0

                        elif frac < 0:
                            neighbor_idx = base_idx - 1
                            if neighbor_idx < 0:
                                f0 = flux_cube[l, base_idx, i]
                                v0 = var_cube[l, base_idx, i]
                                q0 = qual_cube[l, base_idx, i]
                            else:
                                w1 = -frac
                                f0 = flux_cube[l, base_idx, i]
                                v0 = var_cube[l, base_idx, i]
                                f1 = flux_cube[l, neighbor_idx, i]
                                v1 = var_cube[l, neighbor_idx, i]
                                if have_qual:
                                    good0 = (self.qual[l, base_idx, i] == 0)
                                    good1 = (self.qual[l, neighbor_idx, i] == 0)
                                else:
                                    good0 = good1 = True
                                f0, v0, q0 = mix_flux_var(f0, v0, f1, v1, w1=w1, good0=good0, good1=good1)
                            corrected_ydata[l, j, i] = f0
                            corrected_yvar[l, j, i] = v0
                            corrected_yqual[l,j,i] = q0

                        else:
                            # exact integer shift
                            corrected_ydata[l, j, i] = flux_cube[l, base_idx, i]
                            corrected_yvar[l, j, i] = var_cube[l, base_idx, i]
                            corrected_yqual[l,j,i] = qual_cube[l, base_idx, i]

            corrected_yvar = np.clip(corrected_yvar, 0, None)
        else:
            corrected_ydata = flux_cube.copy()
            corrected_yvar = var_cube.copy()
            corrected_yqual = qual_cube.copy()
            new_long_y = self.pix_y

        # ========================
        #  X re-sampling (flux + var)
        # ========================
        if center_x:
            new_long_x = int(long_x - 2 * x_dif_center)
            corrected_data = np.zeros((self.naxis, new_long_y, new_long_x), dtype=flux_cube.dtype)
            corrected_var = np.zeros((self.naxis, new_long_y, new_long_x), dtype=var_cube.dtype)
            corrected_qual = np.zeros((self.naxis, new_long_y, new_long_x), dtype=qual_cube.dtype)

            for j in range(new_long_y):
                for i in range(new_long_x):
                    for l in range(self.naxis):
                        movement = int(x_dif_center + x_offset[l])
                        base_idx = i + movement
                        frac = float(x_variation[l])

                        # Clip base_idx to valid range
                        base_idx = np.clip(base_idx, 0, self.pix_x - 1)

                        if frac > 0:
                            neighbor_idx = base_idx + 1
                            if neighbor_idx >= self.pix_x:
                                f0 = corrected_ydata[l, j, base_idx]
                                v0 = corrected_yvar[l, j, base_idx]
                                q0 = corrected_yqual[l, j, base_idx]
                            else:
                                f0 = corrected_ydata[l, j, base_idx]
                                v0 = corrected_yvar[l, j, base_idx]
                                f1 = corrected_ydata[l, j, neighbor_idx]
                                v1 = corrected_yvar[l, j, neighbor_idx]
                                if have_qual:
                                    good0 = (self.qual[l, j, base_idx] == 0)
                                    good1 = (self.qual[l, j, neighbor_idx] == 0)
                                else:
                                    good0 = good1 = True
                                f0, v0, q0 = mix_flux_var(f0, v0, f1, v1, w1=frac, good0=good0, good1=good1)
                            corrected_data[l, j, i] = f0
                            corrected_var[l, j, i] = v0
                            corrected_qual[l,j,i] = q0

                        elif frac < 0:
                            neighbor_idx = base_idx - 1
                            if neighbor_idx < 0:
                                f0 = corrected_ydata[l, j, base_idx]
                                v0 = corrected_yvar[l, j, base_idx]
                                q0 = corrected_yqual[l, j, base_idx]
                            else:
                                w1 = -frac
                                f0 = corrected_ydata[l, j, base_idx]
                                v0 = corrected_yvar[l, j, base_idx]
                                f1 = corrected_ydata[l, j, neighbor_idx]
                                v1 = corrected_yvar[l, j, neighbor_idx]
                                if have_qual:
                                    good0 = (self.qual[l, j, base_idx] == 0)
                                    good1 = (self.qual[l, j, neighbor_idx] == 0)
                                else:
                                    good0 = good1 = True
                                f0, v0, q0 = mix_flux_var(f0, v0, f1, v1, w1=w1, good0=good0, good1=good1)
                            corrected_data[l, j, i] = f0
                            corrected_var[l, j, i] = v0
                            corrected_qual[l, j,i] = q0

                        else:
                            corrected_data[l, j, i] = corrected_ydata[l, j, base_idx]
                            corrected_var[l, j, i] = corrected_yvar[l, j, base_idx]
                            corrected_qual[l,j,i] = corrected_yqual[l, j, base_idx]

            corrected_var = np.clip(corrected_var, 0, None)
        else:
            corrected_data = corrected_ydata
            corrected_var = corrected_yvar
            corrected_qual = corrected_yqual
            new_long_x = self.pix_x

        # --- Compute final center coordinates (unchanged) ---
        x_center = int(np.round(x_coords_center[0] + parabola(self.wave, *popt_x)[0] - x_dif_center))
        y_center = int(np.round(y_coords_center[0] + parabola(self.wave, *popt_y)[0] - y_dif_center))
        x_center = max(0, min(x_center, self.pix_x - 1))
        y_center = max(0, min(y_center, self.pix_y - 1))

        print("\nPrevious center (before correction): (y, x) = (",
              y_center + y_dif_center, ",", x_center + x_dif_center, ")")
        print("Center of the object: (y, x) = (", y_center, ",", x_center, ")")
        print(f'Flux shape after ADC: {corrected_data.shape}')
        print(f'Var  shape after ADC: {corrected_var.shape}')

        # --- QC 1D slices (same normalization as before; note: variance is not returned normalized) ---
        try:
            center_slice = flux_cube[:, y_center + y_dif_center, x_center + x_dif_center]
        except IndexError:
            center_slice = flux_cube[:, corrected_data.shape[1]-1, corrected_data.shape[2]-1]
        norm_const = np.nanmax(center_slice) if np.isfinite(center_slice).any() else 1.0
        normalized_center_slice = center_slice / norm_const
        try:
            corrected_slice = corrected_data[:, y_center, x_center]
        except IndexError:
            corrected_slice = corrected_data[:, corrected_data.shape[1]-1, corrected_data.shape[2]-1]
        normalized_corrected_slice = corrected_slice / norm_const

        # Save results to object
        self.adc_data = corrected_data
        self.adc_var = corrected_var
        self.adc_qual = corrected_qual

        return (y_center, x_center), normalized_center_slice, normalized_corrected_slice, norm_const

    def sigma_clipping_adapted_for_ifu(
            self,
            outlier_threshold=5,
            window=100):
        """
        Performs an adapted sigma-clipping on an IFU data cube to identify and replace
        spectral outliers (hot/cold pixels or cosmic rays) along the wavelength axis,
        recalculating variance for replaced pixels and handling NaN slices.

        Args:
            outlier_threshold (float): Outlier threshold in units of IQR (interquartile range).
            window (int): Size of the wavelength window for local median calculation.

        Returns:
            None: Sets self.sig_clip_data and self.sig_clip_var.
        """

        if self.adc_data.ndim != 3 or self.adc_var.ndim != 3:
            raise ValueError("Both 'adc_data' and 'adc_var' must be 3D arrays [wavelength, y, x].")

        n_lambda, pix_y, pix_x = self.adc_data.shape

        clean_data = self.adc_data.copy()
        clean_var = self.adc_var.copy()

        median_map_flux = np.zeros((pix_y, pix_x))
        iqr = np.zeros(n_lambda)

        half_window = window // 2
        for lam_idx in range(half_window, n_lambda - half_window):

            # Extract window for this lambda
            slice_window_flux = clean_data[lam_idx - half_window: lam_idx + half_window]

            # If entire slice is NaN, skip
            if np.all(np.isnan(slice_window_flux)):
                continue

            # Median flux per pixel
            median_map_flux[:, :] = np.nanpercentile(slice_window_flux, 50, axis=0)

            # Normalize slice for IQR calculation
            normalized_slice = clean_data[lam_idx] / median_map_flux
            flat_norm = normalized_slice.flatten()
            median_val = np.nanpercentile(flat_norm, 50)
            deviation = flat_norm - median_val
            q3 = float(np.nanpercentile(deviation, 75))
            q1 = float(np.nanpercentile(deviation, 25))
            iqr[lam_idx] = q3 - q1

            deviation_2d = deviation.reshape((pix_y, pix_x))

            # Outlier masks
            mask_upper = deviation_2d > outlier_threshold * q3
            mask_lower = deviation_2d < outlier_threshold * q1
            mask_outlier = mask_upper | mask_lower

            if not np.any(mask_outlier):
                continue

            # Replace flux and recalculate variance for outliers
            for y_idx, x_idx in zip(*np.where(mask_outlier)):
                local_flux_window = slice_window_flux[:, y_idx, x_idx]

                # Skip if all NaN
                if np.all(np.isnan(local_flux_window)):
                    clean_data[lam_idx, y_idx, x_idx] = np.nan
                    clean_var[lam_idx, y_idx, x_idx] = np.nan
                    continue

                # Replace flux with median of window
                median_flux = np.nanmedian(local_flux_window)
                clean_data[lam_idx, y_idx, x_idx] = median_flux

                # Replace variance with variance of window flux values
                var_flux = np.nanvar(local_flux_window, ddof=1)  # sample variance
                clean_var[lam_idx, y_idx, x_idx] = var_flux if not np.isnan(var_flux) else np.nan

        self.sig_clip_data = clean_data
        self.sig_clip_var = clean_var

    def sigma_clipping_1d(self, outlier_threshold=5, window=100):
        """
        Sigma-clipping for a 1D spectrum (flux, variance) using a sliding window.

        Args:
            outlier_threshold (float): Outlier threshold in units of IQR.
            window (int): Size of the wavelength window for local statistics.

        Returns:
            None: Updates self.sig_clip_data (flux) and self.sig_clip_var (variance).
        """

        if self.final_data.ndim != 1 or self.final_var.ndim != 1:
            raise ValueError("Both 'adc_data' and 'adc_var' must be 1D arrays [wavelength].")

        n_lambda = self.final_data.shape[0]
        clean_data = self.final_data.copy()
        clean_var = self.final_var.copy()

        half_window = window // 2

        for lam_idx in range(half_window, n_lambda - half_window):

            # extract local window
            flux_window = clean_data[lam_idx - half_window: lam_idx + half_window]

            # skip if all NaN
            if np.all(np.isnan(flux_window)):
                continue

            # compute median + IQR in window
            median_val = np.nanmedian(flux_window)
            q1 = np.nanpercentile(flux_window, 25)
            q3 = np.nanpercentile(flux_window, 75)
            iqr = q3 - q1

            if iqr == 0 or np.isnan(iqr):
                continue

            # deviation from median
            deviation = clean_data[lam_idx] - median_val

            # outlier check
            if deviation > outlier_threshold * iqr or deviation < -outlier_threshold * iqr:
                # replace with median
                clean_data[lam_idx] = median_val

                # recalc variance from window
                var_flux = np.nanvar(flux_window, ddof=1)
                clean_var[lam_idx] = var_flux if not np.isnan(var_flux) else np.nan

        self.final_data = clean_data
        self.final_var = clean_var

    def optimal_radius_selection_ifu(self,
                                     center,
                                     lower_lam,
                                     upper_lam,
                                     error=33,
                                     plots=True,
                                     titles=False,
                                     percentage=20,
                                     save_loc=None,
                                     n_min=None,
                                     n_max=None,
                                     debug_plots=False):
        """
        Determines the optimal radius for disk integration of spectra by analyzing a small, flat range
        of the spectra and observing its behavior as the radius of the integrated area increases.

        Args:
            center (tuple): Central pixel coordinates of the object in the format (y-center, x-center).
            lower_lam (float): Lower limit in wavelength for the spectra to study.
            upper_lam (float): Upper limit in wavelength for the spectra to study.
            error (float): Percentage of error to consider as a deviation from the theoretical
                           signal-to-noise increase.
            plots (bool): True if you want to visualize plots.
            titles (bool): True if you want titles on the plots.
            percentage (float): Percentage of the initial data to consider for fitting.
            save_loc (str): If a path is provided, the images are save in this directory.
            n_min (int): If a value is provided, the algorithm uses this amount of spaxels as the lower limit
            n_max (int): If a value is provided, the algorithm uses this amount of spaxels as the upper limit
            debug_plots (bool): True if you want additional plots for debugging purposes.
            
        Returns:
            radius (float): Optimal radius for disk integration in arcseconds.
            radius_spaxel (int): Number of pixels within the optimal radius.
        """

        sc_pix_y, sc_pix_x = self.sig_clip_data.shape[1:3]

        # Compute the distance matrix from the center in arcseconds
        distance_matrix = np.zeros((sc_pix_y, sc_pix_x))
        for i in range(sc_pix_x):
            for j in range(sc_pix_y):
                dx_arcsec = self.dx * (i - center[1])
                dy_arcsec = self.dy * (j - center[0])
                distance_matrix[j, i] = np.sqrt(dx_arcsec ** 2 + dy_arcsec ** 2)

        # Create radius steps (in arcseconds)
        max_distance = np.max(distance_matrix)
        r_n = 500  # Number of steps
        radius = np.linspace(0, max_distance, r_n)

        # Determine wavelength indices
        low_idx = self.closest_index(lower_lam, self.wave)
        high_idx = self.closest_index(upper_lam, self.wave)
        # Catch to check user input ranges
        if high_idx == low_idx:
            raise ValueError("Check range limits (should be in nm).")

        if debug_plots:
            print(f"low_idx: {low_idx}, high_idx: {high_idx}")

            fig1, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(self.wave[low_idx:high_idx], self.sig_clip_data[low_idx:high_idx, center[0], center[1]], label='Sig clipped data')
            ax1.set_title('Wavelength region selected')
            ax1.set_xlabel('Wavelength (nm)')
            ax1.set_ylabel('sig clipped data')
            ax1.legend()
            fig1.show()

        lambda_values = self.wave[low_idx:high_idx]

        # model and bounded fit to avoid negative c
        def snr_mod(x, alph, const):
            return alph * np.sqrt(x + const)

        # --- helper functions -----------------------------------------------------
        def compute_snr_arrays(method="original", n_min=n_min):
            snr_arr = np.full(r_n, np.nan)
            sig_arr = np.full(r_n, np.nan)
            noise_arr = np.full(r_n, np.nan)
            r_spax = np.full(r_n, np.nan)
            spec_arr = np.full((r_n, len(lambda_values)), np.nan)

            for rr in range(1, r_n):
                summed_flux = np.zeros(len(lambda_values))
                summed_var = np.zeros(len(lambda_values))
                pixel_count = 0

                for ii in range(sc_pix_x):
                    for jj in range(sc_pix_y):
                        if distance_matrix[jj, ii] < radius[rr]:
                            flux_pix = self.sig_clip_data[low_idx:high_idx, jj, ii]
                            var_pix = self.sig_clip_var[low_idx:high_idx, jj, ii]

                            valid_mask = np.isfinite(flux_pix) & np.isfinite(var_pix)
                            if not np.any(valid_mask):
                                continue

                            summed_flux[valid_mask] += flux_pix[valid_mask]
                            summed_var[valid_mask] += var_pix[valid_mask]
                            pixel_count += 1

                r_spax[rr] = pixel_count
                # Skip radii that don’t meet the minimum spaxel requirement
                if pixel_count == 0 or (n_min is not None and pixel_count < n_min):
                    snr_arr[rr] = np.nan
                    sig_arr[rr] = np.nan
                    noise_arr[rr] = np.nan

                if method == "original":
                    sig_val = np.nansum(summed_flux)
                    sig_arr[rr] = sig_val
                    noise_val = np.sqrt(np.nansum(summed_var)) if np.any(np.isfinite(summed_var)) else np.nan
                    noise_arr[rr] = noise_val
                    spec_arr[rr] = summed_flux
                    if np.isfinite(noise_val) and noise_val > 0:
                        snr_arr[rr] = sig_val / noise_val

                else:  # fallback = mean spectrum
                    mean_spec = np.where(pixel_count > 0, summed_flux / pixel_count, np.nan)
                    mean_var = np.where(pixel_count > 0, summed_var / (pixel_count ** 2), np.nan)
                    sig_val = np.nansum(mean_spec)
                    noise_val = np.sqrt(np.nansum(mean_var)) if np.any(np.isfinite(mean_var)) else np.nan
                    sig_arr[rr] = sig_val
                    noise_arr[rr] = noise_val
                    spec_arr[rr] = mean_spec
                    if np.isfinite(noise_val) and noise_val > 0:
                        snr_arr[rr] = sig_val / noise_val

            return sig_arr, noise_arr, snr_arr, r_spax, spec_arr

        def choose_optimal_index(signal_arr, snr_arr, n_min=None):
            fit_range = max(3, int(percentage / 100 * r_n))  # at least 3 for stability

            xdata = signal_arr[1:fit_range]
            ydata = snr_arr[1:fit_range]
            valid = np.isfinite(xdata) & np.isfinite(ydata) & (xdata > 0) & (ydata > 0)

            if np.sum(valid) < 3:
                good_idxs = np.where(np.isfinite(snr_arr))[0]
                if len(good_idxs) == 0:
                    return 0, np.nan, np.nan
                # enforce minimum radius here
                if n_min is not None:
                    good_idxs = good_idxs[good_idxs >= n_min]
                    if len(good_idxs) == 0:
                        return n_min, np.nan, np.nan
                return int(good_idxs[np.nanargmax(snr_arr[good_idxs])]), np.nan, np.nan

            x_fit = xdata[valid]
            y_fit = ydata[valid]

            def snr_model(x, al, const):
                return al * np.sqrt(x + const)

            try:
                popt, _ = curve_fit(snr_model, x_fit, y_fit, bounds=([0, 0], [np.inf, np.inf]), maxfev=10000)
                alpha_fit, c_fit = popt
            except Exception:
                good_idxs = np.where(np.isfinite(snr_arr))[0]
                if len(good_idxs) == 0:
                    return 0, np.nan, np.nan
                if n_min is not None:
                    good_idxs = good_idxs[good_idxs >= n_min]
                    if len(good_idxs) == 0:
                        return n_min, np.nan, np.nan
                return int(good_idxs[np.nanargmax(snr_arr[good_idxs])]), np.nan, np.nan

            # main deviation search
            start_idx = max(1, n_min if n_min is not None else 1)
            for ii in range(start_idx, r_n):
                if not np.isfinite(signal_arr[ii]) or not np.isfinite(snr_arr[ii]):
                    continue
                predicted = snr_model(signal_arr[ii], alpha_fit, c_fit)
                if predicted <= 0:
                    continue
                actual = snr_arr[ii]
                if np.abs(predicted - actual) > predicted * (error / 100):
                    return max(start_idx, ii - 1), alpha_fit, c_fit

            # fallback to last valid index, but enforce n_min
            good_idxs = np.where(np.isfinite(snr_arr))[0]
            if len(good_idxs) == 0:
                return 0, alpha_fit, c_fit
            if n_min is not None:
                good_idxs = good_idxs[good_idxs >= n_min]
                if len(good_idxs) == 0:
                    return n_min, alpha_fit, c_fit
            return int(good_idxs[-1]), alpha_fit, c_fit

        # --- main: run original then fallback if optimal == 0 --------------------
        # first attempt: original algorithm
        signal_r_orig, noise_r_orig, snr_orig, radius_spaxel_orig, spec_r_orig = compute_snr_arrays(method="original", n_min=n_min)
        opt_idx_orig, alpha_orig, c_orig = choose_optimal_index(signal_r_orig, snr_orig, n_min=n_min)

        # decide whether to accept original result
        use_fallback = False
        if opt_idx_orig == 0 or not np.isfinite(opt_idx_orig):
            use_fallback = True
        elif np.isfinite(radius_spaxel_orig[opt_idx_orig]) and radius_spaxel_orig[opt_idx_orig] == 0:
            use_fallback = True

        if not use_fallback:
            # accept original
            signal_r = signal_r_orig
            noise_r = noise_r_orig
            snr_radius = snr_orig
            radius_spaxel = radius_spaxel_orig
            optimal_idx = int(opt_idx_orig)
            alpha, c = alpha_orig, c_orig
            method_used = "original"
        else:
            # run fallback and re-evaluate
            signal_r_fb, noise_r_fb, snr_fb, radius_spaxel_fb, spec_r_fb = compute_snr_arrays(method="fallback", n_min=n_min)
            opt_idx_fb, alpha_fb, c_fb = choose_optimal_index(signal_r_fb, snr_fb, n_min=n_min)

            # if fallback produced a useful index (>0), use it; else keep original but warn
            if opt_idx_fb > 0 and np.isfinite(opt_idx_fb):
                signal_r = signal_r_fb
                noise_r = noise_r_fb
                snr_radius = snr_fb
                radius_spaxel = radius_spaxel_fb
                optimal_idx = int(opt_idx_fb)
                alpha, c = alpha_fb, c_fb
                method_used = "fallback"
            else:
                # fallback failed: keep original result but warn
                signal_r = signal_r_orig
                noise_r = noise_r_orig
                snr_radius = snr_orig
                radius_spaxel = radius_spaxel_orig
                optimal_idx = int(opt_idx_orig) if np.isfinite(opt_idx_orig) else 0
                alpha, c = alpha_orig, c_orig
                method_used = "original (fallback failed)"

        # Informative print for debugging
        print(
            f"[optimal_radius_selection_ifu] method used: {method_used}, optimal_idx = {optimal_idx}, radius_spaxel = {radius_spaxel[optimal_idx] if np.isfinite(radius_spaxel[optimal_idx]) else 'NaN'}")

        if n_min is not None and radius_spaxel[optimal_idx] < n_min:
            valid_idxs = np.where(radius_spaxel >= n_min)[0]
            if len(valid_idxs) > 0:
                optimal_idx = valid_idxs[0]  # first radius that satisfies minimum spaxels
            else:
                optimal_idx = 0  # fallback if nothing meets minimum

        # Enforce max spaxel count if requested
        if n_max is not None and radius_spaxel[optimal_idx] > n_max:
            closest_n = min(radius_spaxel, key=lambda x: abs(n_max - x))
            optimal_idx = np.where(radius_spaxel == closest_n)[0][0]

        if plots:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(signal_r[1:], snr_radius[1:], color="black", label="Measured SNR")
            ax.plot(signal_r[1:], snr_mod(signal_r[1:], alpha, c),
                    color="cornflowerblue", label="Theoretical SNR")

            ax.fill_between(
                signal_r[1:],
                (1 - error / 100) * snr_mod(signal_r[1:], alpha, c),
                (1 + error / 100) * snr_mod(signal_r[1:], alpha, c),
                color="red", alpha=0.1
            )

            # Check if optimal_idx is valid
            if optimal_idx is not None and 0 <= optimal_idx < len(snr_radius):
                ax.plot(signal_r[optimal_idx], snr_radius[optimal_idx], ".",
                        color="blue", markersize=10,
                        label=f"Optimal Radius: {int(radius_spaxel[optimal_idx])} spaxels")
                if titles:
                    plt.suptitle(f"Optimal Signal-to-Noise Radius ({snr_radius[optimal_idx]:.2f})")
            else:
                # Fallback labeling if optimal_idx is missing/invalid
                if titles:
                    plt.suptitle("Optimal Signal-to-Noise Radius (fallback method)")

            ax.set_xlabel("Signal", fontsize=14)
            ax.set_ylabel("SNR", fontsize=14)
            ax.legend()
            ax.grid(False)

            if save_loc:
                plt.savefig(f"{save_loc}/SignalToNoiseIncrease.png", dpi=300, bbox_inches='tight')
            plt.show()

        if debug_plots:

            x = np.linspace(1, 78, num=500)

            # Plot 1: Signal vs Radius Index
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(x,signal_r, label='Signal')
            #ax1.set_title('Signal vs Radius Index')
            ax1.set_xlabel('Spaxels')
            ax1.set_ylabel('Signal')
            ax1.legend()
            plt.savefig("C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\SignalvsSpaxels.png")
            fig1.show()

            # Plot 2: Noise vs Radius Index
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(noise_r, label='Noise', color='orange')
            ax2.set_title('Noise vs Radius Index')
            ax2.set_xlabel('Radius Index')
            ax2.set_ylabel('Noise')
            ax2.legend()
            fig2.show()

            # Plot 3: Spaxel Count vs Radius Index
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(radius_spaxel, label='Spaxel Count', color='green')
            ax3.set_title('Spaxel Count vs Radius Index')
            ax3.set_xlabel('Radius Index')
            ax3.set_ylabel('Number of Spaxels')
            ax3.legend()
            fig3.show()

            # Plot 4: Measured SNR vs Radius Index
            fig4, ax4 = plt.subplots(figsize=(10, 4))
            ax4.plot(snr_radius[1:], label='Measured SNR', color='black')
            ax4.set_title('Measured SNR vs Radius Index')
            ax4.set_xlabel('Radius Index')
            ax4.set_ylabel('SNR')
            ax4.legend()
            fig4.show()

        print("\nOptimal radius (arcsec):", radius[optimal_idx])
        print("Number of spaxels inside the optimal radius:", int(radius_spaxel[optimal_idx]))

        return radius[optimal_idx], int(radius_spaxel[optimal_idx])

    def disk_integrate(self, center, rad):
        """
        Performs disk integration on IFU observations using the provided radius.

        Args:
            center (tuple): Central pixel coordinates of the object in the format (y-center, x-center).
            rad (float): Limit for the disk integration.

        Returns:
            flux_final (array): The final disk-integrated spectra.
            var_final  (array): The variance of the integrated spectra.
            qual_final (array): Boolean or int mask per wavelength (0=good, 1=flagged).
        """

        nx = len(self.sig_clip_data[0, 0, :])
        ny = len(self.sig_clip_data[0, :, 0])

        # distance from the center for each spaxel
        distance_matrix = np.zeros((ny, nx))
        for i in range(nx):
            for j in range(ny):
                distance_matrix[j, i] = np.sqrt(
                    (self.dy * (j - center[0])) ** 2 +
                    (self.dx * (i - center[1])) ** 2
                )

        n_lambda = len(self.wave)
        flux_final = np.zeros(n_lambda)
        var_final = np.zeros(n_lambda)
        qual_final = np.zeros(n_lambda, dtype=bool)  # False=good, True=flagged

        for i in range(nx):
            for j in range(ny):
                if distance_matrix[j, i] < rad:
                    flux_final += self.sig_clip_data[:, j, i]
                    var_final += self.sig_clip_var[:, j, i]

                    if hasattr(self, "adc_qual") and self.adc_qual is not None:
                        qual_final |= (self.adc_qual[:, j, i] != 1)

        return flux_final, var_final, qual_final

    def integrate_extended(self, cube_path,
                       data=np.array([]),
                       wave=np.array([]),
                       mode="all",
                       discard=[],
                       A=1.5,
                       lower=None,
                       upper=None,
                       save_plots=None,
                       aspect=0.2564):
        """
        Integrates the pixels of the data cube to create a single final spectrum.
        This function operates in two modes: "all", which integrates all pixels except those specified
        in the discard parameter, and "drop," which analyzes if there are pixels too different from the others.
        In the "drop" mode, any pixel detected as an outlier is not considered in the integration.
        "drop fitting is a variation of "drop" mode but uses a line fitting to get rid of the linear tendency of the data.
        For this last one is important to choose a linear section of the spectra with the parameters upper ad lower.


        Args:
            cube_path (str): Path to the data cube.
            data (3D array): 3D array containing the data cube.
            wave (array): Array containing the wavelengths of the data.
            mode (str): Mode of integration to be used, which can be "all" or "drop."
            discard (array): Array of tuples indicating the pixels that should not be considered in the integration.
            A (float): Parameter to determine if a pixel is an outlier (recommended: 1.5).
            lower (float): Lower limit in wavelength for studying the dispersion in the pixels.
            upper (float): Upper limit in wavelength for studying the dispersion in the pixels.
            save_plots (str): Path where the generated images should be saved.
            aspect (float): Value to control the aspect ratio of the displayed images.

        Returns:
            flux_final (array): The final integrated spectrum.
            wave (array): The wavelength array for the final spectrum.
        """

        if len(data) == 0:
            obs = get_pkg_data_filename(cube_path)
            hdul = fits.open(cube_path)
            header = hdul[0].header
            N = header["NAXIS3"]
            wave = np.zeros(N)
            pix_x = header["NAXIS1"]
            pix_y = header["NAXIS2"]
            #obtain the data and wavelength
            data = fits.getdata(obs, ext=0)[:, :, :]

            for i in range(N):
                wave[i] = (i+header["CRPIX3"])*header["CDELT3"] + header["CRVAL3"]

        if lower != None:
            lower = FunctionsDiskIntegrate.closest_index(lower, wave)
        if upper != None:
            upper = FunctionsDiskIntegrate.closest_index(upper, wave)

        if (len(data) > 0) and (len(wave) == 0):
            print(" ")
            print("wave= ")
            print("Error: If you provide the data, also should provide the wavelength")

        if len(data) > 0:
            N = len(data)
            pix_x = len(data[0, 0, :])
            pix_y = len(data[0, :, 0])


        pix_x = len(data[0, 0, :])
        pix_y = len(data[0, :, 0])

        if mode =="all":
            flux_final= np.zeros(N)
            for i in range(pix_x):
                for j in range(pix_y):
                    if (j, i) not in discard:
                        flux_final = flux_final + data[:, j, i]

            return flux_final, wave

        if mode == "drop":

            medianas = np.zeros((pix_y, pix_x))
            for i in range(pix_x):
                for j in range(pix_y):
                    medianas[j, i] = np.nanpercentile(data[lower:upper, j, i], 50)
            medianas = medianas / np.max(medianas)
            medianas_raw = medianas.copy()

            leveled_data = data / medianas

            median_flux = np.zeros(N)
            for l in range(N):
                median_flux[l] = np.nanpercentile(leveled_data[l, :, :], 50)
            flux_final= np.zeros(N)
            deviations = np.zeros((pix_y, pix_x))
            for i in range(pix_x):
                for j in range(pix_y):
                    flux = leveled_data[:, j, i]
                    dif = flux[lower:upper] - median_flux[lower:upper]
                    deviations[j, i] = np.std(dif)

            iqr = np.nanpercentile(deviations, 75) - np.nanpercentile(deviations, 25)
            limit = np.nanpercentile(deviations, 75) + A*iqr
            #deviations = deviations / iqr

            for i in range(pix_x):
                for j in range(pix_y):
                    if (j, i) not in discard:
                        if deviations[j, i] > limit:
                            print("pixel (y="+str(j)+", x="+str(i)+") considered noisy")
                            medianas[j, i] = None
                            pass
                        elif deviations[j, i] < limit:
                            flux_final = flux_final + data[:, j, i]
                    if (j, i) in discard:
                        print("pixel (y="+str(j)+", x="+str(i)+") discarded")
                        medianas[j, i] = None



            fig, axes = plt.subplots(1, 3, figsize=(6, 21))
            axes[0].set_title("Visualization")
            im0 = axes[0].imshow(medianas_raw, origin="lower", aspect=aspect)
            #bar0 = plt.colorbar(im0)
            axes[0].set_ylabel("y-spaxels")
            axes[0].set_xlabel("x-spaxels")

            axes[2].set_title("Integrated pixels")
            im2 = axes[2].imshow(medianas, origin="lower", aspect=aspect)
            #bar2 = plt.colorbar(im2)
            axes[2].set_ylabel("y-spaxels")
            axes[2].set_xlabel("x-spaxels")

            axes[1].set_title("Noise per pixel")
            im1 = axes[1].imshow(deviations, origin="lower", aspect=aspect)
            #bar1 = plt.colorbar(im1)
            axes[1].set_ylabel("y-spaxels")
            axes[1].set_xlabel("x-spaxels")
            if save_plots != None:
                plt.savefig(save_plots + "/Visualization_RawData.png")

            return flux_final, wave


        if mode == "drop fitting":

            def linear(x, m, c):
                return x*m+c

            n = int(upper-lower)
            centered_data = np.zeros((n, pix_y, pix_x))


            for j in range(pix_y):
                for i in range(pix_x):
                    popt, pcov = curve_fit(linear, wave[lower:upper], data[lower:upper, j, i])
                    m, c = popt
                    centered_data[:, j, i] = data[lower:upper, j, i] - linear(wave[lower:upper], m, c)

            medianas = np.zeros((pix_y, pix_x))
            for i in range(pix_x):
                for j in range(pix_y):
                    medianas[j, i] = np.nanpercentile(data[lower:upper, j, i], 50)
            medianas = medianas / np.max(medianas)
            medianas_raw = medianas.copy()

            median_centered = np.zeros(n)
            leveled_data = centered_data / medianas
            for l in range(n):
                median_centered[l] = np.nanpercentile(leveled_data[l], 50)

            flux_final= np.zeros(N)
            deviations = np.zeros((pix_y, pix_x))
            for i in range(pix_x):
                for j in range(pix_y):
                    flux = leveled_data[:, j, i]
                    dif = flux - median_centered
                    deviations[j, i] = np.std(dif)

            iqr = np.nanpercentile(deviations, 75) - np.nanpercentile(deviations, 25)
            limit = np.nanpercentile(deviations, 75) + A*iqr

            for i in range(pix_x):
                for j in range(pix_y):
                    if (j, i) not in discard:
                        if deviations[j, i] > limit:
                            print("pixel (y="+str(j)+", x="+str(i)+") considered noisy")
                            medianas[j, i] = None
                            pass
                        elif deviations[j, i] < limit:
                            flux_final = flux_final + data[:, j, i]
                    if (j, i) in discard:
                        print("pixel (y="+str(j)+", x="+str(i)+") discarded")
                        medianas[j, i] = None

            fig, axes = plt.subplots(1, 3, figsize=(6, 21))
            axes[0].set_title("Visualization")
            im0 = axes[0].imshow(medianas_raw, origin="lower", aspect=aspect)
            #bar0 = plt.colorbar(im0)
            axes[0].set_ylabel("y-spaxels")
            axes[0].set_xlabel("x-spaxels")

            axes[2].set_title("Integrated pixels")
            im2 = axes[2].imshow(medianas, origin="lower", aspect=aspect)
            #bar2 = plt.colorbar(im2)
            axes[2].set_ylabel("y-spaxels")
            axes[2].set_xlabel("x-spaxels")

            axes[1].set_title("Noise per pixel")
            im1 = axes[1].imshow(deviations, origin="lower", aspect=aspect)
            #bar1 = plt.colorbar(im1)
            axes[1].set_ylabel("y-spaxels")
            axes[1].set_xlabel("x-spaxels")
            if save_plots != None:
                plt.savefig(save_plots + "/Visualization_RawData.png")

            return flux_final, wave


    def save_file(self, path_to_save,
                  header,
                  data,
                  variance,
                  qual,
                  radius = None,
                  radius_spaxel = None,
                  center = None,
                  comment = None,
                  lower_limit=None,
                  upper_limit=None,
                  correct_center_x=None,
                  correct_center_y=None,
                  look_center_x=None,
                  look_center_y=None,
                  sc_thresh=None,
                  window_sc=None,
                  percentage=None,
                  error=None):
        """
        Save the final data into a FITS file. Also writes in the header all the important information about the final data.
        This is dessign for disk-integrated spectra.

        Args:
            path_to_save (str): Path where the data will be saved.
            header (str): Header of the raw data.
            data (float): Final data that needs to be saved.
            radius (float): Optimal radius for disk integration in arcseconds.
            radius_spaxel (int): Number of pixels within the optimal radius.
            center (tuple): Calculated center of the new data cube.
            lower_limit (float): Lower limit in wavelength for optimal radius selection.
            upper_limit (float): Upper limit in wavelength for optimal radius selection.
            corrected_center_x (bool): True if atmospheric correction is needed in the x-direction.
            corrected_center_y (bool): True if atmospheric correction is needed in the y-direction.
            look_center_x (tuple): Higher and lower values for parabolic fit in the x-direction.
            look_center_y (tuple): Higher and lower values for parabolic fit in the y-direction.
            plots (bool): True to visualize plots.
            max_plots (float): Factor to set vertical plot limits. ylim = max_plots * data_median.
            sc_thresh (float): Amount of sigma away from the median to consider an outlier.
            window_sc (float): Width of the window for comparison and data leveling.
            percentage (float): Percentage of initial data to consider for fitting.
            error (float): Percentage of error to consider as a deviation from the theoretical signal-to-noise increase.

        Returns:
            None
        """

        hdr = header

        if radius_spaxel != None:
            hdr["SH SPAXELS R"] = radius_spaxel
        if radius_spaxel == None:
            hdr["SH SPAXELS R"] = "No information"

        if radius != None:
            hdr["SH ANGULAR R"] = radius
        if radius == None:
            hdr["SH ANGULAR R"] = "No information"

        if sc_thresh != None:
            hdr["SH A SIGMA CLIPPING"] = sc_thresh
            hdr["SH WINDOW SIGMA CLIPPING"] = window_sc
        if sc_thresh == None:
            hdr["SH A SIGMA CLIPPING"] = "No information"
            hdr["SH WINDOW SIGMA CLIPPING"] = "No information"

        if center != None:
            hdr["SH OBJECT CENTER Y"] = center[0]
            hdr["SH OBJECT CENTER X"] = center[1]
        if center == None:
            hdr["SH OBJECT CENTER Y"] = "No information"
            hdr["SH OBJECT CENTER X"] = "No information"

        if comment != None:
            hdr["SH COMMENT"] = comment
        if comment == None:
            hdr["SH COMMENT"] = "No comments"

        if lower_limit != None:
            hdr["SH LOWER LIMIT"] = lower_limit
            hdr["SH UPPER LIMIT"] = upper_limit
        if lower_limit == None:
            hdr["SH LOWER LIMIT"] = "No information"
            hdr["SH UPPER LIMIT"] = "No information"

        if correct_center_x != None:
            hdr["SH X CENTER CORRECTION"] = correct_center_x
        if correct_center_y != None:
            hdr["SH Y CENTER CORRECTION"] = correct_center_y
        if correct_center_x == None:
            hdr["SH X CENTER CORRECTION"] = "No information"
        if correct_center_y == None:
            hdr["SH Y CENTER CORRECTION"] = "No information"

        if look_center_x != None:
            hdr["SH X CENTER CORRECTION RANGE"] = look_center_x
        if look_center_y != None:
            hdr["SH Y CENTER CORRECTION RANGE"] = look_center_y
        if look_center_x == None:
            hdr["SH X CENTER CORRECTION RANGE"] = "No information"
        if look_center_y == None:
            hdr["SH Y CENTER CORRECTION RANGE"] = "No information"

        if percentage != None:
            hdr["SH PERCENTAGE FOR FITTING"] = look_center_x
        if percentage == None:
            hdr["SH PERCENTAGE FOR FITTING"] = "No information"

        if error != None:
            hdr["SH ERROR ACEPTED"] = error
        if percentage == None:
            hdr["SH ERROR ACEPTED"] = "No information"

        # Write to fits file
        empty_primary = fits.PrimaryHDU(data, header=hdr)
        variance_hdu = fits.ImageHDU(variance, name='VAR')
        qual_hdu = fits.ImageHDU(qual.astype(np.uint8), name='QUAL')

        hdul = fits.HDUList([empty_primary, variance_hdu, qual_hdu])
        hdul.writeto(path_to_save, overwrite=True, output_verify='ignore')

    def save_file_extended(self, path_to_save,
                       header,
                       data,
                       mode_used=None,
                       discard_pixels=None,
                       comment=None,
                       lower=None,
                       upper=None):
        """
        Save the final data into a FITS file and write important information into the header.
        This function is designed for cases where the observation is entirely within the studied object.

        Args:
            path_to_save (str): Path where the data will be saved.
            header (str): Header of the raw data.
            data (float): Final data to be saved.
            mode_used (str): Mode of integration to be used, which can be "all" or "drop."
            discard_pixels (list): List of tuples indicating the pixels that should not be considered in the integration.
            comment (str): Special comment to be saved in the header of the final FITS file.
            lower (float): Lower limit in wavelength for studying the dispersion in the pixels.
            upper (float): Upper limit in wavelength for studying the dispersion in the pixels.

        Returns:
            None
        """

        hdr = header

        if mode_used != None:
            hdr["SH DI MODE USED"] = mode_used
        if mode_used == None:
            hdr["SH DI MODE USED"] = "No information"
        if discard_pixels != None:
            hdr["SH DISCARD PIXELS FOR DI"] = str(discard_pixels)
        if discard_pixels == None:
            hdr["SH DISCARD PIXELS FOR DI"] = "No information"
        if comment != None:
            hdr["SH COMMENT"] = comment
        if comment == None:
            hdr["SH COMMENT"] = "No comments"
        if lower != None:
            hdr["SH LOWER LIMIT"] = lower
        if lower == None:
            hdr["SH LOWER LIMIT"] = "No comments"
        if upper != None:
            hdr["SH UPPER LIMIT"] = upper
        if upper == None:
            hdr["SH UPPER LIMIT"] = "No comments"

        empty_primary = fits.PrimaryHDU(data, header=hdr)

        hdul = fits.HDUList([empty_primary])
        hdul.writeto(path_to_save, overwrite=True)


    def process_my_ifu_obs(self, fits_path,
                           lower_limit,
                           upper_limit,
                           correct_center_x=True,
                           correct_center_y=True,
                           look_center_x=None,
                           look_center_y=None,
                           plots=True,
                           max_plots=3,
                           sig_clip_thresh=5,
                           window_sc=None,
                           percentage=25,
                           error=33,
                           path_to_save = None,
                           comment = None,
                           save_plots = None,
                           n_min=None,
                           N_max=None,
                           titles=False):
        """
        Computes a single disk-integrated spectrum from observations with IFUs. The algorithm involves three steps:
        1. Corrects atmospheric dispersion (optional for x and y directions).
        2. Identifies and replaces outliers using an adapted sigma clipping algorithm.
        3. Selects the optimal radius for disk integration based on central pixel analysis.

        Args:
            fits_path (str): Path to the data cube.
            lower_limit (float): Lower limit in wavelength for optimal radius selection.
            upper_limit (float): Upper limit in wavelength for optimal radius selection.
            corrected_center_x (bool): True if atmospheric correction is needed in the x-direction.
            corrected_center_y (bool): True if atmospheric correction is needed in the y-direction.
            look_center_x (tuple): Higher and lower values for parabolic fit in the x-direction.
            look_center_y (tuple): Higher and lower values for parabolic fit in the y-direction.
            plots (bool): True to visualize plots.
            max_plots (float): Factor to set vertical plot limits. ylim = max_plots * data_median.
            sig_clip_thresh (float): Amount of sigma away from the median to consider an outlier.
            window_sc (float): Width of the window for comparison and data leveling.
            percentage (float): Percentage of initial data to consider for fitting.
            error (float): Percentage of error to consider as a deviation from the theoretical signal-to-noise increase.
            path_to_save (str): Path where the final data will be saved.
            comment (str): Special comment that will be saved in the header of the final FITS file.
            save_plots (str): If a path is provided, the images are save in this directory.
            N_max (int): If a value is provided, the algorithm uses this amount of spaxels as the upper limit

        Returns:
            final_data (array): Disk-integrated spectra of the data-cube
            wave (array): Wavelength of the final spectra
        """

        # Extracts important information from the datacube and optionally plots a spatial
        # heatmap for vizualiztion and inspection
        self.extract_info(fits_path, save_plots)

        # Applies atmospheric dispersion correction to a 3D data cube by tracking
        # the apparent movement of the object's center with wavelength.
        center, norm_center_slice, norm_adc_slice, norm_const = self.atmospheric_dispersion_correction(center_x=correct_center_x,
                                                                                                           center_y=correct_center_y,
                                                                                                           range_x=look_center_x,
                                                                                                           range_y=look_center_y,
                                                                                                           plots=plots,
                                                                                                           plot_save_loc=save_plots)
        # Grab 1D QUAL mask for the extracted spaxel
        qual_slice = self.adc_qual[:, center[0], center[1]]  # shape (naxis,)
        adc_mask = (qual_slice == 1)  # True = good pixels

        # ------- Atmospheric Dispersion Correction Plot -------
        fig, (ax1, ax2) = plt.subplots(2, 1,
                                       gridspec_kw={'height_ratios': [2, 1], 'hspace': 0}, sharex=True)

        # Plot original and corrected normalized flux on top subplot
        ax1.plot(self.wave[adc_mask], norm_center_slice[adc_mask],
                 c="k", label="Original data", linewidth=0.5)
        ax1.plot(self.wave[adc_mask], norm_adc_slice[adc_mask],
                 c="cornflowerblue", label="Atmospheric Dispersion Corrected data", linewidth=0.5)
        ax1.set_ylim(top=(1.5 * np.nanpercentile(norm_center_slice[adc_mask], 98)), bottom=0.0)
        ax1.set_ylabel("Normalized Flux")
        ax1.legend()
        ax1.grid(False)

        # Calculate and plot residuals (difference between original and corrected)
        residuals_adc = norm_center_slice - norm_adc_slice
        ax2.plot(self.wave[adc_mask], residuals_adc[adc_mask],
                 c='cornflowerblue', linewidth=0.5)
        ax2.set_ylim(bottom=-1.5 * np.nanpercentile(-residuals_adc[adc_mask], 98),
                     top=1.5 * np.nanpercentile(residuals_adc[adc_mask], 98))
        ax2.set_xlabel("Wavelength (µm)")
        ax2.set_ylabel(r"$\Delta$", fontsize=14)
        ax2.grid(False)

        if titles:
            fig.suptitle("Atmospheric Dispersion Correction", fontsize=14)

        if save_plots is not None:
            plt.savefig(save_plots + "/AtmosphericDispersion_Correction.png")
        plt.show()

        # ---------------- Sigma Clipping ----------------
        if not window_sc:
            window_sc = max(21, len(self.wave) // 20)
        self.sigma_clipping_adapted_for_ifu(outlier_threshold=sig_clip_thresh, window=int(window_sc))

        # Create plot
        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            gridspec_kw={'height_ratios': [2, 1], 'hspace': 0},
            sharex=True
        )

        # Extract data slice at the specified spatial center
        try:
            norm_clipped_slice = self.sig_clip_data[:, center[0], center[1]]
        except IndexError:
            norm_clipped_slice = self.sig_clip_data[:, -1, -1]

        # Normalize to show relative scaling
        ax1.plot(self.wave[adc_mask] / 1000, norm_adc_slice[adc_mask], color="black", linewidth=0.5, label="Before Clipping")
        ax1.plot(self.wave[adc_mask] / 1000, norm_clipped_slice[adc_mask] / norm_const, color="cornflowerblue", linewidth=0.5,
                 label="After Clipping")
        ax1.set_ylim(bottom=0.0, top=1.5*np.nanpercentile(norm_clipped_slice[adc_mask] / norm_const, 98))
        ax1.set_ylabel("Normalized Flux", fontsize=12)
        ax1.legend(loc="upper right", fontsize=10)

        # Calculate and plot residuals
        residuals = norm_adc_slice - norm_clipped_slice
        ax2.plot(self.wave[adc_mask] / 1000, residuals[adc_mask], color="cornflowerblue", linewidth=0.5, label="Residuals")
        ax2.set_ylim(bottom=-1.5*np.nanpercentile(-residuals[adc_mask], 98), top=1.5*np.nanpercentile(residuals[adc_mask], 98))
        ax2.set_xlabel("Wavelength (µm)", fontsize=14)
        ax2.set_ylabel(r"$\Delta$", fontsize=14)

        # Optional title, save, and show
        if titles:
            fig.suptitle("Sigma Clipping", fontsize=14)
        plt.savefig(save_plots + "/SigmaClipping.png", dpi=300, bbox_inches='tight')
        plt.show()


        # ---------------- Optimal Radius Selection ----------------
        radius, radius_spaxels = self.optimal_radius_selection_ifu(center,
                                                                   lower_limit,
                                                                   upper_limit,
                                                                   percentage=percentage,
                                                                   error=error,
                                                                   save_loc=save_plots,
                                                                   titles=titles,
                                                                   n_min=n_min,
                                                                   n_max=N_max,
                                                                   debug_plots=True)


        # ---------------- Final Integration ----------------
        self.final_data, self.final_var, self.final_qual = self.disk_integrate(center, radius)
        self.sigma_clipping_1d(outlier_threshold=sig_clip_thresh, window=int(window_sc))


        # ---------------- Plotting ----------------
        collapsed_data = np.nanmedian(self.data, axis=(1,2))
        scale_factor = np.nanmax(self.final_data) / np.nanmax(collapsed_data)
        norm_center_slice = collapsed_data*scale_factor / (np.nanmax(collapsed_data*scale_factor))
        normalized_final = self.final_data / (np.nanmax(collapsed_data*scale_factor))

        # Grab 1D QUAL mask for the extracted spaxel
        qual_mask = (self.final_qual == 0)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1,
                                       gridspec_kw={'height_ratios': [2, 1, 1], 'hspace': 0}, sharex=True)
        ax1.plot(self.wave[qual_mask]/1000, norm_center_slice[qual_mask], c="k", label="Original Center Pixel", linewidth=0.5)
        ax1.plot(self.wave[qual_mask]/1000, norm_adc_slice[qual_mask], c="cornflowerblue", label="Dispersion Corrected Center", linewidth=0.5)
        ax1.plot(self.wave[qual_mask]/1000, normalized_final[qual_mask], c="orchid", label="Final data", linewidth=0.5 )
        ax1.set_ylabel("Relative Flux")
        ax1.set_ylim(bottom=0.0, top=1.1*np.nanpercentile(norm_adc_slice[qual_mask], 98))
        ax1.legend()
        residuals_adc = norm_center_slice - norm_adc_slice
        residuals_final = norm_adc_slice - normalized_final
        ax2.plot(self.wave[qual_mask]/1000, residuals_adc[qual_mask], c='cornflowerblue', linewidth=0.5)
        ax2.set_ylim(bottom=-1.5*np.nanpercentile(-residuals_adc, 98), top=1.5*np.nanpercentile(residuals_adc, 98))
        ax3.plot(self.wave/1000, residuals_final, c="orchid", linewidth=0.5)
        ax3.set_ylim(bottom=-1.5*np.nanpercentile(-residuals_final, 98), top=1.5*np.nanpercentile(residuals_final, 98))
        # fig.suptitle("Atmospheric Dispersion Correction") #, fontsize=21)
        ax3.set_xlabel("Wavelength (µm)")  # , fontsize=18)
        ax2.set_ylabel(r"$\Delta$", fontsize=14)
        ax3.set_ylabel(r"$\Delta$", fontsize=14)
        if save_plots != None and comment == 'VIS':
            plt.savefig(save_plots + "/Final_DiskIntegrated_Spectra_VIS.png")
        elif save_plots != None:
            plt.savefig(save_plots + "/Final_DiskIntegrated_Spectra.png")
        plt.show()

        fig, (ax1, ax2) = plt.subplots(2, 1,
                                       gridspec_kw={'height_ratios': [2, 1], 'hspace': 0}, sharex=True)
        ax1.plot(self.wave[qual_mask]/1000, norm_center_slice[qual_mask], c="k", label="Original center slice", linewidth=0.5)
        ax1.plot(self.wave[qual_mask]/1000, normalized_final[qual_mask], c="cornflowerblue", label="Spacially Integrated", linewidth=0.5 )
        ax1.set_ylabel("Relative Flux")
        ax1.set_ylim(bottom=0.0, top=1.5*np.nanpercentile(normalized_final[qual_mask], 98))
        ax1.legend()
        residuals_final = norm_center_slice - normalized_final
        ax2.plot(self.wave[qual_mask]/1000, residuals_final[qual_mask], c="cornflowerblue", linewidth=0.5)
        ax2.set_ylim(bottom=-1.5*np.nanpercentile(-residuals_final[qual_mask], 98), top=1.5*np.nanpercentile(residuals_final[qual_mask], 98))
        ax2.set_xlabel("Wavelength (µm)")  # , fontsize=18)
        ax2.set_ylabel(r"$\Delta$", fontsize=14)
        if save_plots != None and comment == 'VIS':
            plt.savefig(save_plots + "/Final_DiskIntegrated_Spectra_VIS_Mini.png")
        elif save_plots != None:
            plt.savefig(save_plots + "/Final_DiskIntegrated_Spectra.png")
        plt.show()

        if path_to_save != None:
            hdul = fits.open(fits_path)
            self.save_file(path_to_save,
                           hdul[0].header,
                           self.final_data,
                           self.final_var,
                           self.final_qual,
                           radius=radius,
                           radius_spaxel=radius_spaxels,
                           center=center,
                           comment=comment,
                           lower_limit=lower_limit,
                           upper_limit=upper_limit,
                           correct_center_x=correct_center_x,
                           correct_center_y=correct_center_y,
                           look_center_x=look_center_x,
                           look_center_y=look_center_y,
                           sc_thresh=sig_clip_thresh,
                           window_sc=window_sc,
                           percentage=percentage,
                           error=error)

        return self.final_data

    def process_ifu_extended(self, fits_path,
                         plots=True,
                         max_plots=3,
                         A_sc=3,
                         window_sc=100,
                         discard=np.array([]),
                         mode_di = "drop",
                         A_di=3,
                         path_to_save = None,
                         comment = None,
                         save_plots = None,
                         lower=None,
                         upper=None):
        """
        Compute a single integrated spectrum from observations with IFUs. The algorithm involves two steps:
        1. Identifies and replaces outliers using an adapted sigma-clipping algorithm.
        2. Integrates the data cube with the integration mode provided.

        Args:
            fits_path (str): Path to the data cube.
            plots (bool): True to visualize plots.
            max_plots (float): Factor to set vertical plot limits. ylim = max_plots * data_median.
            A_sc (float): The number of standard deviations away from the median to consider as an outlier.
            window_sc (float): Width of the window for comparison and data leveling.
            mode_di (str): Mode of integration to be used, which can be "all" or "drop."
            discard (list): List of tuples indicating the pixels that should not be considered in the integration.
            A_di (float): Parameter to consider a pixel an outlier (recommended: 1.5).
            path_to_save (str): Path where the final data will be saved.
            comment (str): Special comment to be saved in the header of the final FITS file.
            save_plots (str): If a path is provided, the images will be saved in this directory.
            lower (float): Lower limit in wavelength for studying the dispersion in the pixels.
            upper (float): Upper limit in wavelength for studying the dispersion in the pixels.

        Returns:
            final_data (array): Integrated spectra of the data cube.
            wave (array): Wavelength of the final spectra.
        """

        data, wave, pix_x, pix_y, dx, dy = visualize(fits_path, save_plots, plots=False)

        clean_data = Sigma_clipping_adapted_for_IFU("",
                                                    data=data,
                                                    wave=wave,
                                                    A=A_sc,
                                                    window=window_sc)

        X = pix_x//2
        Y = pix_y//2
        fig, axes = plt.subplots(1, 1, figsize=(18, 10))
        median = np.median(data[:, Y, X])
        axes.plot(wave, data[:, Y, X], c="red", linewidth=0.5, label="Raw data")
        axes.plot(wave, clean_data[:, Y, X], c="k", linewidth=0.5, label="Data with Sigma-Clipping")
        axes.set_title("Data with and without Sigma clipping", fontsize=22)
        axes.set_xlabel("Wavelength", fontsize=18)
        axes.set_ylabel("Count", fontsize=18)
        axes.legend()
        axes.set_ylim(0, median*max_plots)
        if save_plots != None:
            plt.savefig(save_plots + "/SigmaClipping.png")

        final_data, wave = integrate_extended("",
                                              data=clean_data,
                                              wave=wave,
                                              mode=mode_di,
                                              discard=discard,
                                              A=A_di,
                                              lower=lower,
                                              upper=upper)

        fig, (axes, axes2) = plt.subplots(2, 1, figsize=(18, 10))
        median = np.median(final_data)
        axes.plot(wave, final_data, c="k", linewidth=0.5)
        axes.set_title("Final data after the Integration", fontsize=22)
        axes.set_xlabel("Wavelength", fontsize=18)
        axes.set_ylabel("Count", fontsize=18)
        axes.legend()
        axes.set_ylim(0, median*max_plots)
        if save_plots != None:
            plt.savefig(save_plots + "/Final_DiskIntegrated_Spectra.png")

        if path_to_save != None:
            hdul = fits.open(fits_path)
            save_file_extended(path_to_save,
                                        hdul[0].header,
                                        final_data,
                                        mode_used=mode_di,
                                        discard_pixels=discard,
                                        comment=comment,
                                        lower=lower,
                                        upper=upper)
        return final_data, wave

if __name__ == "__main__":

    #objects = ['neptune1', 'neptune2', 'neptune3', 'neptune4',]
    #objects = ['neptune1']
    #objects = ['feige-110']
    #objects = ['gd71']
    #objects = ['ltt7987']
    #objects = ['uranus1', 'uranus2', 'uranus3', 'uranus4']
    #objects = ['titan1', 'titan2', 'titan3']
    #objects = ['saturn7', 'saturn12', 'saturn14',
    #               'saturn17', 'saturn18', 'saturn19', 'saturn20', 'saturn21', 'saturn22',
    #               'saturn23', 'saturn24']#, 'saturn25', 'saturn26']
    #objects = ['saturn1', 'saturn3', 'saturn4', 'saturn5']
    #objects = ['saturn6', 'saturn7', 'saturn8', 'saturn9', 'saturn10']
    #objects = ['saturn11', 'saturn12', 'saturn13', 'saturn14', 'saturn16']
    #objects = ['saturn17', 'saturn18', 'saturn19', 'saturn20', 'saturn21']
    #objects = ['saturn22', 'saturn23', 'saturn24', 'saturn25', 'saturn26']
    #objects = ['hip09']
    #objects = ['pluto1', 'pluto2', 'pluto3', 'pluto4', 'pluto5', 'pluto6', 'pluto7', 'pluto8', 'feige-110_p1',]

    for obj in objects:
        inst = FunctionsDiskIntegrate(obj)
        save_plots, _ = os.path.split(inst.file_paths['save_UVB'])
        print(f'Processing UVB band of {obj}')
        uvb_dat = inst.process_my_ifu_obs(inst.file_paths['UVB'], inst.file_paths['range_UVB'][0], inst.file_paths['range_UVB'][1], path_to_save=inst.file_paths['save_UVB'], save_plots=save_plots, look_center_x=(350, 550), look_center_y=(350, 550), sig_clip_thresh=8, n_min=5)
        print(f'Processing VIS band of {obj}')
        vis_dat = inst.process_my_ifu_obs(inst.file_paths['VIS'], inst.file_paths['range_VIS'][0], inst.file_paths['range_VIS'][1], comment='VIS', path_to_save=inst.file_paths['save_VIS'], save_plots=save_plots, look_center_x=(600, 950), look_center_y=(600, 950), sig_clip_thresh=5, error=40, n_min=5)
        print(f'Processing NIR band of {obj}')
        nir_dat = inst.process_my_ifu_obs(inst.file_paths['NIR'], inst.file_paths['range_NIR'][0], inst.file_paths['range_NIR'][1], path_to_save=inst.file_paths['save_NIR'], save_plots=save_plots, sig_clip_thresh=2, n_min=5)

