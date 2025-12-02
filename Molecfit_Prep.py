import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import copy


class MolecfitPrep:

    def __init__(self, object):
        self.object = object
        all_paths = {
            'neptune1': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_1\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_1\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_1\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_1\\MOV_Neptune_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_1\\MOV_Neptune_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_1\\MOV_Neptune_DiskIntegrated_NIR.fits",
            },
            'neptune2': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_2\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_2\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_2\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_2\\MOV_Neptune_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_2\\MOV_Neptune_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_2\\MOV_Neptune_DiskIntegrated_NIR.fits",
            },
            'neptune3': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_3\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_3\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_3\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_3\\MOV_Neptune_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_3\\MOV_Neptune_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_3\\MOV_Neptune_DiskIntegrated_NIR.fits",
            },
            'neptune4': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_4\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_4\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_4\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_4\\MOV_Neptune_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_4\\MOV_Neptune_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_4\\MOV_Neptune_DiskIntegrated_NIR.fits",
            },
            'feige-110': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\FEIGE-110\\FEIGE-110_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\FEIGE-110\\FEIGE-110_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\FEIGE-110\\FEIGE-110_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\FEIGE-110\\MOV_FEIGE-110_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\FEIGE-110\\MOV_FEIGE-110_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\FEIGE-110\\MOV_FEIGE-110_DiskIntegrated_NIR.fits",
            },
            'uranus1': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_1\\MOV_Uranus_2_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_1\\MOV_Uranus_2_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_1\\MOV_Uranus_2_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_1\\MOV_Uranus_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_1\\MOV_Uranus_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_1\\MOV_Uranus_DiskIntegrated_NIR.fits",
            },
            'uranus2': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_2\\MOV_Uranus_2_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_2\\MOV_Uranus_2_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_2\\MOV_Uranus_2_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_2\\MOV_Uranus_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_2\\MOV_Uranus_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_2\\MOV_Uranus_DiskIntegrated_NIR.fits",
            },
            'uranus3': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_3\\MOV_Uranus_2_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_3\\MOV_Uranus_2_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_3\\MOV_Uranus_2_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_3\\MOV_Uranus_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_3\\MOV_Uranus_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_3\\MOV_Uranus_DiskIntegrated_NIR.fits",
            },
            'uranus4': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_4\\MOV_Uranus_2_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_4\\MOV_Uranus_2_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_4\\MOV_Uranus_2_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_4\\MOV_Uranus_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_4\\MOV_Uranus_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_4\\MOV_Uranus_DiskIntegrated_NIR.fits",
            },
            'gd71': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\GD71\\GD71_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\GD71\\GD71_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\GD71\\GD71_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\GD71\\MOV_GD71_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\GD71\\MOV_GD71_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\GD71\\MOV_GD71_DiskIntegrated_NIR.fits",
            },
            'titan1': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_1\\MOV_Titan_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_1\\MOV_Titan_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_1\\MOV_Titan_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_1\\MOV_Titan_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_1\\MOV_Titan_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_1\\MOV_Titan_DiskIntegrated_NIR.fits",
            },
            'titan2': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_2\\MOV_Titan_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_2\\MOV_Titan_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_2\\MOV_Titan_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_2\\MOV_Titan_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_2\\MOV_Titan_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_2\\MOV_Titan_DiskIntegrated_NIR.fits",
            },
            'titan3': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_3\\MOV_Titan_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_3\\MOV_Titan_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_3\\MOV_Titan_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_3\\MOV_Titan_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_3\\MOV_Titan_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_3\\MOV_Titan_DiskIntegrated_NIR.fits",
            },
            'saturn1': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_1\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_1\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_1\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_1\\MOV_Saturn_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_1\\MOV_Saturn_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_1\\MOV_Saturn_DiskIntegrated_NIR.fits",
            },
            'saturn3': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_3\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_3\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_3\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_3\\MOV_Saturn_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_3\\MOV_Saturn_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_3\\MOV_Saturn_DiskIntegrated_NIR.fits",
            },
            'saturn4': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_4\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_4\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_4\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_4\\MOV_Saturn_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_4\\MOV_Saturn_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_4\\MOV_Saturn_DiskIntegrated_NIR.fits",
            },
            'saturn5': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_5\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_5\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_5\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_5\\MOV_Saturn_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_5\\MOV_Saturn_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_5\\MOV_Saturn_DiskIntegrated_NIR.fits",
            },
            'saturn6': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_6\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_6\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_6\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_6\\MOV_Saturn_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_6\\MOV_Saturn_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_6\\MOV_Saturn_DiskIntegrated_NIR.fits",
            },
            'saturn7': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_7\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_7\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_7\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_7\\MOV_Saturn_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_7\\MOV_Saturn_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_7\\MOV_Saturn_DiskIntegrated_NIR.fits",
            },
            'saturn8': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_8\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_8\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_8\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_8\\MOV_Saturn_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_8\\MOV_Saturn_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_8\\MOV_Saturn_DiskIntegrated_NIR.fits",
            },
            'saturn9': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_9\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_9\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_9\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_9\\MOV_Saturn_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_9\\MOV_Saturn_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_9\\MOV_Saturn_DiskIntegrated_NIR.fits",
            },
            'saturn10': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_10\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_10\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_10\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_10\\MOV_Saturn_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_10\\MOV_Saturn_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_10\\MOV_Saturn_DiskIntegrated_NIR.fits",
            },
            'saturn11': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_11\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_11\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_11\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_11\\MOV_Saturn_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_11\\MOV_Saturn_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_11\\MOV_Saturn_DiskIntegrated_NIR.fits",
            },
            'saturn12': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_12\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_12\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_12\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_12\\MOV_Saturn_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_12\\MOV_Saturn_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_12\\MOV_Saturn_DiskIntegrated_NIR.fits",
            },
            'saturn13': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_13\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_13\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_13\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_13\\MOV_Saturn_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_13\\MOV_Saturn_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_13\\MOV_Saturn_DiskIntegrated_NIR.fits",
            },
            'saturn14': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_14\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_14\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_14\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_14\\MOV_Saturn_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_14\\MOV_Saturn_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_14\\MOV_Saturn_DiskIntegrated_NIR.fits",
            },
            'saturn16': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_16\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_16\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_16\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_16\\MOV_Saturn_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_16\\MOV_Saturn_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_16\\MOV_Saturn_DiskIntegrated_NIR.fits",
            },
            'saturn17': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_17\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_17\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_17\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_17\\MOV_Saturn_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_17\\MOV_Saturn_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_17\\MOV_Saturn_DiskIntegrated_NIR.fits",
            },
            'saturn18': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_18\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_18\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_18\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_18\\MOV_Saturn_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_18\\MOV_Saturn_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_18\\MOV_Saturn_DiskIntegrated_NIR.fits",
            },
            'saturn19': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_19\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_19\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_19\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_19\\MOV_Saturn_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_19\\MOV_Saturn_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_19\\MOV_Saturn_DiskIntegrated_NIR.fits",
            },
            'saturn20': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_20\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_20\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_20\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_20\\MOV_Saturn_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_20\\MOV_Saturn_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_20\\MOV_Saturn_DiskIntegrated_NIR.fits",
            },
            'saturn21': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_21\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_21\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_21\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_21\\MOV_Saturn_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_21\\MOV_Saturn_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_21\\MOV_Saturn_DiskIntegrated_NIR.fits",
            },
            'saturn22': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_22\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_22\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_22\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_22\\MOV_Saturn_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_22\\MOV_Saturn_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_22\\MOV_Saturn_DiskIntegrated_NIR.fits",
            },
            'saturn23': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_23\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_23\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_23\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_23\\MOV_Saturn_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_23\\MOV_Saturn_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_23\\MOV_Saturn_DiskIntegrated_NIR.fits",
            },
            'saturn24': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_24\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_24\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_24\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_24\\MOV_Saturn_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_24\\MOV_Saturn_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_24\\MOV_Saturn_DiskIntegrated_NIR.fits",
            },
            'saturn25': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_25\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_25\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_25\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_25\\MOV_Saturn_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_25\\MOV_Saturn_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_25\\MOV_Saturn_DiskIntegrated_NIR.fits",
            },
            'saturn26': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_26\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_26\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_26\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_26\\MOV_Saturn_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_26\\MOV_Saturn_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_26\\MOV_Saturn_DiskIntegrated_NIR.fits",
            },
            'ltt7987': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\LTT7987\\LTT7987_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\LTT7987\\LTT7987_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\LTT7987\\LTT7987_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\LTT7987\\MOV_LTT7987_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\LTT7987\\MOV_LTT7987_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\LTT7987\\MOV_LTT7987_DiskIntegrated_NIR.fits",
            },
            'hip09': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\LTT7987\\LTT7987_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Standard2\\Hip095318_TELL_IFU_TELL_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Standard2\\Hip095318_TELL_IFU_TELL_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Standard2\\MOV_Hip095318_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Standard2\\MOV_Hip095318_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Standard2\\MOV_Hip095318_DiskIntegrated_NIR.fits",
            },

        }
        self.file_paths = all_paths[object]

    def combine_disk_integrated_spectra(self, object_list, arm, clip_sigma=3.0):
        """
        Combine multiple disk-integrated spectra into a single, high-quality spectrum.
        Performs weighted combination using variance, robust clipping of outliers, and
        returns a properly propagated variance array.

        Parameters
        ----------
        object_list : list
            List of object identifiers.
        arm : str
            XSHOOTER arm ('UVB', 'VIS', 'NIR').
        clip_sigma : float, optional
            Sigma threshold for robust outlier rejection (default 3.0).

        Returns
        -------
        out_flux : np.ndarray
            Combined flux spectrum (1D, length Nlam).
        out_var : np.ndarray
            Combined variance spectrum (1D, length Nlam).
        N_used : np.ndarray
            Number of spectra contributing at each wavelength.
        """
        spectra = []
        errs_list = []
        quals_list = []
        airm_starts = []
        airm_ends = []
        exptimes = []

        # --- Load DI spectra if possible ---
        for obj in object_list:
            try:
                path = MolecfitPrep(obj).file_paths.get(f'DI_{arm}')
                if path is None or not os.path.exists(path):
                    continue
                with fits.open(path) as hdul:
                    data = hdul[0].data
                    errs = hdul[1].data
                    qual = hdul[2].data
                    header = hdul[0].header
                    n_lambda = header["NAXIS1"]
                    airm_start = header["ESO TEL AIRM START"]
                    airm_end = header["ESO TEL AIRM END"]
                    exptime = header["EXPTIME"]
                    wave = np.zeros(n_lambda)
                    for i in range(n_lambda):
                        wave[i] = (i + 1 - header["CRPIX3"]) * header["CDELT3"] + header["CRVAL3"]
                    if data is None or not np.any(data):
                        continue
                    spectra.append(data)
                    errs_list.append(errs)
                    quals_list.append(qual)
                    airm_starts.append(airm_start)
                    airm_ends.append(airm_end)
                    exptimes.append(exptime)
            except Exception as e:
                print(f"Skipping {obj} DI_{arm} due to error: {e}")
                continue

        # --- Fallback to post-xpipe summed spectra ---
        if len(spectra) == 0:
            print(f"No valid DI spectra found for arm {arm}. Falling back to post-xpipe cubes.")
            for obj in object_list:
                try:
                    path = MolecfitPrep(obj).file_paths.get(arm)
                    if path is None or not os.path.exists(path):
                        continue
                    with fits.open(path) as hdul:
                        cube = hdul[0].data
                        errs = hdul[1].data
                        qual = hdul[2].data
                        n_lambda = header["NAXIS3"]
                        wave = np.zeros(n_lambda)
                        for i in range(n_lambda):
                            wave[i] = (i + 1 - header["CRPIX3"]) * header["CDELT3"] + header["CRVAL3"]
                        if cube is None or not np.any(cube):
                            continue
                        summed = np.nansum(cube, axis=(1, 2))
                        summed_errs = np.sqrt(np.nansum(errs ** 2, axis=(1, 2)))
                        spectra.append(summed)
                        errs_list.append(summed_errs)
                        quals_list.append(qual)
                except Exception as e:
                    print(f"Skipping post-xpipe {obj} for {arm} due to error: {e}")
                    continue

        if len(spectra) == 0:
            raise ValueError(f"No valid spectra found for arm {arm} in either DI or post-xpipe.")

        # --- Stack spectra, variances, and quality flags ---
        flux = np.vstack(spectra)  # (Nexp, Nlam)
        errors = np.vstack(errs_list)  # (Nexp, Nlam)
        quals = np.vstack(quals_list)  # (Nexp, Nlam), integer mask

        # A pixel is usable if:
        #   - finite flux and error
        #   - error > 0
        #   - qual == 0
        mask = np.isfinite(flux) & np.isfinite(errors) & (errors > 0) & (quals == 0)

        Nexp, Nlam = flux.shape
        out_flux = np.full(Nlam, np.nan)
        out_var = np.full(Nlam, np.nan)
        N_used = np.zeros(Nlam, dtype=int)
        out_qual = np.zeros(Nlam, dtype=int)  # new quality mask for combined spectrum

        for i in range(Nlam):
            m = mask[:, i]

            if m.sum() == 0:
                # Nothing usable: carry forward flag
                out_qual[i] = 1
                continue

            f = flux[m, i]
            s = errors[m, i]

            # Robust median + MAD
            med = np.nanmedian(f)
            mad = 1.4826 * np.nanmedian(np.abs(f - med))
            scat = np.sqrt(s ** 2 + mad ** 2)

            # Outlier rejection
            r = (f - med) / scat
            keep = np.abs(r) <= clip_sigma

            if np.sum(keep) == 0:
                # Fallback: use median, still mark as "usable"
                out_flux[i] = med
                out_var[i] = mad ** 2 if mad > 0 else np.nanmedian(s ** 2)
                N_used[i] = 1
                continue

            # Weighted mean
            w = 1.0 / (s[keep] ** 2)
            out_flux[i] = np.nansum(w * f[keep]) / np.nansum(w)
            out_var[i] = max(1.0 / np.nansum(w), mad ** 2)
            N_used[i] = np.sum(keep)


        airm_start = np.nanmedian(airm_starts)
        airm_end = np.nanmedian(airm_ends)
        exptime = np.nanmedian(exptimes)

        # Return combined quality flag alongside everything else
        return wave, out_flux, out_var, N_used, out_qual, airm_start, airm_end, exptime

    def plot_check(self, olist, combined_flux, wave, combined_qual, arm='ARM', clip_sigma=3.0):
        """
        Visualize the combination of spectra.

        Parameters
        ----------
        olist : list of np.ndarray
            List of 1D spectra (spatially integrated) used for combination.
        combined_flux : np.ndarray
            Final combined flux spectrum.
        wave : np.ndarray
            Wavelength array corresponding to the spectra.
        arm : str, optional
            XSHOOTER arm for labeling.
        clip_sigma : float, optional
            Clipping sigma for display purposes.
        """

        nspec = len(olist)
        if nspec == 0:
            print("No spectra to visualize.")
            return

        # --- Setup figure ---
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        # --- Plot each spectrum as it is added ---
        for i, offset in enumerate(olist):
            offset_path = MolecfitPrep(offset).file_paths.get(f'DI_{arm}')
            with fits.open(offset_path) as hdul:
                spec = hdul[0].data
                qual = hdul[2].data
            qual_mask = (qual == 0)
            axes[0].plot(wave[qual_mask], spec[qual_mask], alpha=0.5, label=f"Offset {i+1}")

        axes[0].plot(wave, combined_flux, color='rebeccapurple', lw=1.5, label='Combined spectrum')
        axes[0].set_ylabel("Flux")
        axes[0].set_ylim(bottom=0.0, top=(2.0*np.nanpercentile(combined_flux, 98)))
        axes[0].set_title(f"{arm} arm: Individual spectra and combined")
        axes[0].legend(fontsize=8)
        axes[0].grid(True)

        # --- Residuals ---
        qual_mask = (combined_qual == 0)
        for i, offset in enumerate(olist):
            offset_path = MolecfitPrep(offset).file_paths.get(f'DI_{arm}')
            with fits.open(offset_path) as hdul:
                spec = hdul[0].data
            resid = spec - combined_flux
            axes[1].plot(wave[qual_mask], resid[qual_mask], alpha=0.5)

        axes[1].axhline(0, color='rebeccapurple', lw=1.0, linestyle='--')
        axes[1].set_xlabel("Wavelength")
        axes[1].set_ylabel("Residuals")
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    def make_molecfit_ready(self, post_xpipe_path, combined_flux, combined_var, combined_qual, airm_start, airm_end, exptime):
        """
        Prepare a MOLECFIT-ready FITS file by copying the post-xshooter pipeline output,
        then replacing the FLUX and ERR/VAR HDUs with repeated combined spectra.

        Parameters
        ----------
        post_xpipe_path : str
            Path to the post-XSHOOTER pipeline FITS file.
        combined_flux : np.ndarray
            1D combined spectrum from `combine_disk_integrated_spectra`.
        combined_var : np.ndarray
            1D variance array corresponding to `combined_flux`.
        """
        # --- parameters you can tune ---
        INFLATION_FACTOR = 4.0  # inflate variance for replaced pixels to be conservative
        REPLACED_QUAL_FLAG = 8  # bit flag to mark pixels we replaced (choose unused bit)
        REPLACED_FAR_FLAG = 16  # additional bit for "far from nearest good pixel"
        GAP_LIMIT = 20  # pixels; if replaced pixel is farther from any good pixel than this, treat more conservatively

        # --- prepare ---
        Nlam = combined_flux.shape[0]
        x = np.arange(Nlam)

        # robust mask of "base-good" pixels used to build the interpolant:
        # choose pixels that are finite, non-negative and have finite non-negative variance
        base_good = np.isfinite(combined_flux) & np.isfinite(combined_var) & (combined_flux >= 0) & (combined_var >= 0)

        # if no base-good pixels exist, we can't interpolate sensibly -> fallback
        if not base_good.any():
            # mark everything bad and set safe defaults (caller should handle this as fatal)
            combined_qual[:] |= 2  # keep previous flags but set a fatal bit (tunable)
            combined_flux[:] = 0.0
            fallback_var = np.nanmedian(errors) ** 2 if (
                        'errors' in globals() and np.isfinite(np.nanmedian(errors))) else 1.0
            combined_var[:] = fallback_var * 100.0
            print(
                "WARNING: no non-negative finite pixels available to interpolate from; filled with zeros and large variance.")
        else:
            # interpolate flux and var from base_good pixels
            interp_flux = np.interp(x, x[base_good], combined_flux[base_good])

            # prepare variance array for interpolation: ensure finite positive values at base_good positions
            var_good = np.array(combined_var, copy=True)
            # replace non-finite or negative variances at base_good positions with median
            valid_var_vals = var_good[base_good]
            median_var = np.nanmedian(valid_var_vals[np.isfinite(valid_var_vals)]) if np.any(
                np.isfinite(valid_var_vals)) else 1.0
            var_good[~np.isfinite(var_good) | (var_good <= 0)] = median_var
            interp_var = np.interp(x, x[base_good], var_good[base_good])

            # build replace mask: any finite negative OR non-finite value
            replace_mask = (np.isfinite(combined_flux) & (combined_flux < 0)) | (~np.isfinite(combined_flux))

            # If you *only* want to replace negatives and not NaNs, change above accordingly.

            # Apply replacements for all matched pixels (regardless of existing qual)
            if np.any(replace_mask):
                # assign interpolated flux, but force non-negative (clip to 0)
                new_vals = interp_flux[replace_mask]
                new_vals = np.where(new_vals < 0.0, 0.0, new_vals)
                combined_flux[replace_mask] = new_vals

                # assign variance: use interpolated variance * inflation factor; ensure positive finite
                new_vars = interp_var[replace_mask]
                # replace any non-finite or <=0 new_vars with median_var
                new_vars[~np.isfinite(new_vars) | (new_vars <= 0)] = median_var
                combined_var[replace_mask] = new_vars * INFLATION_FACTOR

                # flag replacements (OR preserves existing flags)
                combined_qual[replace_mask] |= REPLACED_QUAL_FLAG

            # handle "far from nearest good pixel" conservative inflation
            good_idx = np.nonzero(base_good)[0]
            pos = np.searchsorted(good_idx, x)
            left_idx = good_idx[np.clip(pos - 1, 0, good_idx.size - 1)]
            right_idx = good_idx[np.clip(pos, 0, good_idx.size - 1)]
            dist = np.minimum(np.abs(x - left_idx), np.abs(right_idx - x))
            far_mask = (dist > GAP_LIMIT) & replace_mask
            if np.any(far_mask):
                # more aggressive inflation and flagging for extrapolated replacements
                combined_var[far_mask] *= 10.0
                combined_qual[far_mask] |= REPLACED_FAR_FLAG

            # final safety: force any remaining negatives (should be none) to zero and mark them
            still_neg = combined_flux < 0
            if np.any(still_neg):
                combined_flux[still_neg] = 0.0
                combined_var[still_neg] = np.where(np.isfinite(combined_var[still_neg]) & (combined_var[still_neg] > 0),
                                                   combined_var[still_neg] * INFLATION_FACTOR,
                                                   median_var * INFLATION_FACTOR)
                combined_qual[still_neg] |= REPLACED_QUAL_FLAG

        # diagnostic output
        n_replaced = np.sum((combined_qual & REPLACED_QUAL_FLAG) != 0)
        n_replaced_far = np.sum((combined_qual & REPLACED_FAR_FLAG) != 0)
        n_neg_after = np.sum(combined_flux < 0)
        print(f"patched {int(n_replaced)} pixels (including {int(n_replaced_far)} far/extrapolated); negatives remaining: {int(n_neg_after)}")

        # --- Open post-xpipe file and copy HDUList ---
        hdulist = fits.open(post_xpipe_path)  # don't use 'with' here
        new_hdulist = fits.HDUList([hdu.copy() for hdu in hdulist])

        # --- Identify flux and variance HDUs ---
        if 'FLUX' not in new_hdulist or 'ERRS' not in new_hdulist:
            raise ValueError("Post-xpipe file must contain 'FLUX' and 'ERRS' HDUs.")

        flux_hdu = new_hdulist['FLUX']
        err_hdu = new_hdulist['ERRS']
        qual_hdu = new_hdulist['QUAL']

        # --- Check length consistency ---
        nlam = flux_hdu.data.shape[0]
        if combined_flux.shape[0] != nlam or combined_var.shape[0] != nlam:
            raise ValueError(
                f"Combined spectrum length {combined_flux.shape[0]} or variance length {combined_var.shape[0]} "
                f"does not match the number of wavelength slices in post-xpipe ({nlam})."
            )

        # --- Get spatial shape ---
        ny, nx = flux_hdu.data.shape[1], flux_hdu.data.shape[2]

        # --- Create 3D cubes by repeating the combined spectrum along spatial dimensions ---
        flux_cube = np.tile(combined_flux[:, np.newaxis, np.newaxis], (1, ny, nx))
        var_cube = np.tile(combined_var[:, np.newaxis, np.newaxis], (1, ny, nx))
        qual_cube = np.tile(combined_qual[:, np.newaxis, np.newaxis], (1, ny, nx))

        # --- Replace HDU data ---
        flux_hdu.data = flux_cube
        err_hdu.data = var_cube
        qual_hdu.data = qual_cube

        # --- Optional: update header OBJECT card ---
        hdr0 = new_hdulist[0].header
        if hdr0.get('OBJECT', '').strip().upper() == 'STD, FLUX':
            hdr0['OBJECT'] = getattr(self, 'object', 'UNKNOWN').upper()
        hdr0['ESO TEL AIRM START'] = airm_start
        hdr0['ESO TEL AIRM END'] = airm_end
        hdr0['EXPTIME'] = exptime

        # --- Write MOLECFIT_READY output ---
        out_fname = post_xpipe_path.replace('.fits', '_MOLECFIT_READY.fits').replace("MERGE3D", "MERGE1D")
        new_hdulist.writeto(out_fname, output_verify="fix+warn", overwrite=True, checksum=True)
        print(f"MOLECFIT-ready file written to {out_fname}")

    def make_molecfit_ready_old(self, post_xpipe, DI_file):
        self.post_xpipe = post_xpipe
        self.DI_file = DI_file
        self.expected_EXTNAMEs = {'FLUX': -1, 'ERRS': -1, 'QUAL': -1}

        # Load original 3D file and 1D disk-integrated spectrum
        HDUlist_xpipe = fits.open(self.post_xpipe)
        HDUlist_DI = fits.open(self.DI_file)

        # Identify HDU indices for the 3D data
        for i_hdu, HDU in enumerate(HDUlist_xpipe):
            if HDU.header.get('EXTNAME') in self.expected_EXTNAMEs:
                self.expected_EXTNAMEs[HDU.header['EXTNAME']] = i_hdu

        # Copy the 3D structure
        nHDUlist = copy.deepcopy(HDUlist_xpipe)
        _HDUlist = copy.deepcopy(HDUlist_xpipe)

        # Prepare new extensions with placeholder data
        for ext in ['FLUX', 'ERRS', 'QUAL']:
            nHDUlist.append(_HDUlist[self.expected_EXTNAMEs[ext]])
            nHDUlist[-1].header['EXTNAME'] = _HDUlist[self.expected_EXTNAMEs[ext]].header['EXTNAME'] + "_MOLECFIT_READY"
            nHDUlist[-1].data = np.zeros_like(_HDUlist[self.expected_EXTNAMEs[ext]].data, dtype='float32')

        # Load the 1D spectrum
        newFLUX_1D = HDUlist_DI[0].data.astype('float32')
        HDUlist_DI.close()

        # Get shape of original 3D data: (nz, ny, nx)
        template_data = _HDUlist[self.expected_EXTNAMEs['FLUX']].data
        nz, ny, nx = template_data.shape

        # Tile the 1D spectrum into a 3D cube
        if newFLUX_1D.shape[0] != nz:
            raise ValueError(f"Mismatch in spectral length: 1D={newFLUX_1D.shape[0]}, cube={nz}")

        tiled_cube = np.tile(newFLUX_1D[:, np.newaxis, np.newaxis], (1, ny, nx))
        zero_cube = np.zeros_like(tiled_cube, dtype='float32')

        # Assign tiled data to MOLECFIT_READY extensions
        nHDUlist[-3].data = tiled_cube  # FLUX_MOLECFIT_READY
        nHDUlist[-2].data = zero_cube  # ERRS_MOLECFIT_READY
        nHDUlist[-1].data = zero_cube  # QUAL_MOLECFIT_READY

        # Add SPIKES extension as empty
        #spike_HDU = fits.ImageHDU(data=zero_cube, name='SPIKES')
        #nHDUlist.append(spike_HDU)

        # Generate output filename
        fname = self.post_xpipe.replace('.fits', '_MOLECFIT_READY.fits').replace("MERGE3D", "MERGE1D")

        # Save to disk
        nHDUlist.writeto(fname, output_verify="fix+warn", overwrite=True, checksum=True)
        HDUlist_xpipe.close()
        print('File Written')


if __name__ == "__main__":

    #object_list = ['neptune1', 'neptune2', 'neptune3', 'neptune4']
    #object_list = ['feige-110']
    #object_list = ['gd71']
    object_list = ['uranus1', 'uranus2', 'uranus3', 'uranus4']
    #object_list = ['titan1', 'titan2', 'titan3']
    #object_list = ['saturn7', 'saturn12', 'saturn14',
    #               'saturn17', 'saturn18', 'saturn19', 'saturn20', 'saturn21', 'saturn22',
    #               'saturn23', 'saturn24']#, 'saturn25', 'saturn26']
    #object_list = ['ltt7987']
    #object_list = ['hip09']
    arm = ['UVB', 'VIS', 'NIR']
    #object_list = ['neptune4']
    #arm = ['NIR']

    for a in arm:
        if len(object_list) >= 2:
            prep = MolecfitPrep(object_list[0])
            print(f"Combining images for {a} arm")
            wave, combined_spectrum, combined_var, n_used, combined_qual, airm_start, airm_end, exptime = prep.combine_disk_integrated_spectra(object_list, a)
            prep.plot_check(object_list, combined_spectrum, wave, combined_qual, a)
            print(f"Writing molecfit ready fits file for {a} arm")
            prep.make_molecfit_ready(prep.file_paths[a], combined_spectrum, combined_var, combined_qual, airm_start, airm_end, exptime)
        else:
            prep = MolecfitPrep(object_list[0])
            for obj in object_list:
                try:
                    path = prep.file_paths.get(f'DI_{a}')
                    if path is None or not os.path.exists(path):
                        continue
                    with fits.open(path) as hdul:
                        data = hdul[0].data
                        errs = hdul[1].data
                        qual = hdul[2].data
                        header = hdul[0].header
                        n_lambda = header["NAXIS1"]
                        airm_start = header["ESO TEL AIRM START"]
                        airm_end = header["ESO TEL AIRM END"]
                        exptime = header["EXPTIME"]
                        wave = np.zeros(n_lambda)
                        for i in range(n_lambda):
                            wave[i] = (i + 1 - header["CRPIX3"]) * header["CDELT3"] + header["CRVAL3"]
                        if data is None or not np.any(data):
                            print('data not found')
                            continue
                except Exception as e:
                    print(f"Skipping {obj} DI_{a} due to error: {e}")
                    continue
                print(f"Writing molecfit ready fits file for {a} arm")
                prep.make_molecfit_ready(prep.file_paths[a], data, errs, qual, airm_start, airm_end, exptime)
