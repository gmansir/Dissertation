"""
Module Name: Planet_Analysis_v2
Module Author: Giovannina Mansir (nina.mansir@gmail.com)
Module Version: 2.0.1
Last Modified: 2023-12-13

Description:
This class is for manipulating IFU images from XSHOOTER and compliling it into a library for ease of understanding and
modeling.

Usage:
import matplotlib.pyplot as plt
import Planet_Analysis_v2 as PA
analysis = PA.DataAnalysis(body='')
work_dir = '/home/gmansir/Thesis/Titan_tests/reflex_end_products/2023-03-20T15:27:11/XSHOO.2019-09-26T02:17:55.664_tpl/'
analysis.spectral_cleanup(work_dir)
analysis.edge_matching()

Dependancies:
-numpy
-matplotlib
-astropy
-copy
-glob
"""

import os
import pdb
import warnings
import copy
import glob
import pandas as pd
import numpy as np
import numpy.ma as ma
from astropy import units as un
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.modeling.models import BlackBody
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import StrMethodFormatter
from scipy.interpolate import UnivariateSpline as uspline
from scipy.constants import h, c, k
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from specutils import Spectrum1D
from specutils.fitting import fit_generic_continuum
import re
from datetime import datetime
#import procastro
from astropy import units as u
from astropy.nddata import StdDevUncertainty, NDData
from astropy.visualization import quantity_support
from astropy.modeling import models
from synphot import SourceSpectrum, units
from synphot import SpectralElement
from synphot import Observation
from synphot.models import Empirical1D, BlackBodyNorm1D

#matplotlib.use('TkAgg')

class DataAnalysis():

    '''
    A collection of methods for flexible analysis of our planetary library data
    '''

    def __init__(self, **kwargs):

        """
        Initializes a new instance of the DataAnalysis class.

        Args:
            body (str): The name of the celestial body of focus.
        """
        plt.ion()
        self.body = None
        celestial_bodies = ['neptune', 'fiege-110']
        celestial_bodies = ', '.join(celestial_bodies)

        if 'body' in kwargs:
            body = kwargs['body'].lower()
            while body not in celestial_bodies:
                print("Invalid object entered.")
                body = input(f"Please enter a valid celestial body ({celestial_bodies}): ").lower()
            self.body = body
        else:
            self.body = input(f"Please enter a celestial body ({celestial_bodies}): ").lower()
            while self.body not in celestial_bodies:
                print("Invalid object entered.")
                self.body = input(f"Please enter a valid celestial body ({celestial_bodies}): ").lower()

        self.work_dir = f"C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\{self.body.title()}-Check\\Post_Molecfit\\"

        print(f"Instance initiated for {self.body.title()}.")

    def spectral_cleanup(self, directory=None, molecfit_ready=True):

        """
        Collects files from a post XSHOOTER pipeline reduction, separated out into individual spaxel spectra. Then
        calculates their moving maximums, sigma clips the spikes, and median combines them. It saves the normalized
        median combined and sigma clipped spectrum into a file to be processed with make_molecfit_ready()

        :param work_dir: The working directory containing the separated spaxel spectra for processing

        :return: A normalized and sigma clipped fits file
        """

        if directory == None:
            pre_dir = self.work_dir + 'PreMolecfit/reflex_end_products/'
        else:
            pre_dir = directory
        flist = glob.glob(pre_dir+'/*')
        target = f'IFU_MERGE3D_DATA_OBJ_'
        base_file = [f for f in flist if target in f]
        base_file.sort()
        if len(base_file) == 0:
            raise ValueError(f'Please check directory, no viable images found in {pre_dir}.')
        band = base_file[0][-8:-5].upper()
        spaxels = [f for f in flist if 'pixel' in f]

        w, h = plt.figaspect(0.25)
        fig1 = plt.figure(1, figsize=(w, h))
        ax = fig1.add_subplot(211)
        all_data = []
        norm_data = []
        orig_data = []
        max_norm = []
        poly_specs = []
        for sp in spaxels:
            with fits.open(sp) as hdul:
                data = copy.deepcopy(hdul[0].data)
                odata = data / data[8888]
                orig_data.append(odata)
                ndata = data / np.median(data[1843:1860])
                norm_data.append(ndata)
                window_size = 24
                moving_medians = list(np.zeros(window_size - 1) + 2)
                moving_maximum = list(np.zeros(window_size - 1) + 2)
                mm = 0
                while mm < len(ndata) - window_size + 1:
                    window = ndata[mm:mm + window_size]
                    win_median = np.median(window)
                    moving_medians.append(win_median)
                    sorted_idx = np.argsort(window)
                    forth_highest_idx = sorted_idx[-8]
                    forth_highest_val = window[forth_highest_idx]
                    moving_maximum.append(forth_highest_val)
                    mm += 1
                moving_medians = np.array(moving_medians, dtype='f')
                residuals = ndata - moving_medians
                with warnings.catch_warnings():  # Ignore warnings due to NaNs or Infs
                    warnings.simplefilter("ignore")
                    sigclip = sigma_clip(residuals, sigma=4, cenfunc='median')
                clipped_data = np.ma.masked_array(ndata, mask=sigclip.mask)
                # filled_data = clipped_data.filled(moving_medians)
                filled_data = clipped_data.filled(np.nan)
                fdata = filled_data.tolist()
                all_data.append(fdata)
                CRVAL1 = hdul[0].header['CRVAL1']
                CDELT1 = hdul[0].header['CDELT1']
                NAXIS1 = hdul[0].header['NAXIS1']
                wave = np.array([CRVAL1 + CDELT1 * i for i in range(NAXIS1)]) / 1000.
                header = hdul[0].header
                moving_maximum = np.array(moving_maximum, dtype='f')
                poly_coeffs = np.polyfit(wave, moving_maximum, deg=6)
                poly_spec = np.polyval(poly_coeffs, wave)
                poly_specs.append(poly_spec)
                poly_norm = data / poly_spec
                max_norm.append(poly_norm)
                ax.plot(wave, filled_data, linewidth=0.5, label=(sp[-9:-5]))
        med_combined = np.nanmedian(all_data, axis=0)
        norm_combined = np.nanmedian(norm_data, axis=0)
        orig_combined = np.nanmedian(orig_data, axis=0)
        max_combined = np.nanmedian(max_norm, axis=0)
        max_combined = max_combined / np.median(max_combined[18433:18600])
        polys_combined = np.nanmedian(poly_specs, axis=0)
        spikes = np.nonzero(np.isnan(med_combined))
        med_errs = np.std(all_data, axis=0)
        window_size = 24
        moving_medians = list(np.zeros(window_size - 1) + 2)
        mm = 0
        while mm < len(med_combined) - window_size + 1:
            window = med_combined[mm:mm + window_size]
            win_median = np.nanmedian(window)
            moving_medians.append(win_median)
            mm += 1
        moving_medians = np.array(moving_medians, dtype='f')
        if isinstance(spikes[0], np.ndarray):
            spikes = np.concatenate(spikes)
        if len(spikes) == 0:
            pass
        else:
            for i in spikes:
                if i <= 2 or i >= len(med_combined) - 3:
                    continue
                med_combined[i - 2] = moving_medians[i - 2]
                med_combined[i - 1] = moving_medians[i - 1]
                med_combined[i] = moving_medians[i]
                med_combined[i + 1] = moving_medians[i + 1]
                med_combined[i + 2] = moving_medians[i + 2]

        print('Data Calculated')
        ax2 = fig1.add_subplot(212)
        ax2.plot(wave, orig_combined, linewidth=0.5, label='Original/float')
        ax2.plot(wave, norm_combined, linewidth=0.5, label='Original/median(range)')
        ax2.plot(wave, med_combined, linewidth=0.5, label='Range Norm, Sig Clip, Med Filled')
        ax2.plot(wave, max_combined, linewidth=0.5, label='Original/moving maximum')
        ax2.plot(wave, polys_combined, linewidth=0.5, label='Polyfit Spectra median combined')
        ax2.legend()
        ax.set_ylim([-0.2, 5])

        flux = np.array(med_combined)
        errs = np.array(med_errs)

        norm_sig_file = base_file[0][:-5] +'_NORM_SIG.fits'
        if os.path.exists(norm_sig_file):
            os.remove(norm_sig_file)
        HDU = fits.ImageHDU(data=flux, name='FLUX')
        hdulist = fits.HDUList([fits.PrimaryHDU(), HDU])
        HDU = fits.ImageHDU(data=errs, name='ERRS')
        hdulist.append(HDU)
        HDU = fits.ImageHDU(data=flux, name='QUAL')
        hdulist.append(HDU)
        HDU = fits.ImageHDU(data=spikes, name='SPIKES')
        hdulist.append(HDU)
        HDU = fits.ImageHDU(data=np.array(orig_combined), name='NO MODS')
        hdulist.append(HDU)
        HDU = fits.ImageHDU(data=np.array(norm_combined), name='NORMALIZED ONLY')
        hdulist.append(HDU)

        hdulist.writeto(norm_sig_file)
        hdulist.close()
        print('Wrote normalized and sigma clipped file')
        plt.show()

        if molecfit_ready == True:
            self.make_molefit_ready(base_file[0],norm_sig_file)

    def closest_index(self, arr, val):
        """
        Finds the index of the value closest to that requested in a given array

        :param arr: array
        :param val: value you are searching for
        :return: index in array of number closest to val
        """
        if type(arr) != list:
            arr = list(arr)
        close_func = lambda x: abs(x - val)
        close_val = min(arr, key=close_func)

        return arr.index(close_val)

    def fits_get_wave_data(self, file_path):

        with fits.open(file_path) as hdul:
            data = hdul[0].data
            data = np.nanmedian(data, axis=(1,2))
            data = data.tolist()

            try:
                errs = hdul['ERRS'].data
                errs = np.nan_to_num(errs)
                errs = np.nanmedian(errs, axis=(1,2))
                errs = errs.tolist()
            except KeyError or IndexError:
                errs = np.zeros(len(data))

            CRVAL3 = hdul[0].header['CRVAL3']
            CDELT3 = hdul[0].header['CDELT3']
            NAXIS3 = hdul[0].header['NAXIS3']
            EXPTIME = hdul[0].header['EXPTIME']
            wave = np.array([CRVAL3 + CDELT3 * i for i in range(NAXIS3)]) / 1000.

        return wave, data, EXPTIME, errs

    def replace_errs_hdu(source_file, target_file):
        # Open the source FITS file
        source_hdul = fits.open(source_file, mode='readonly')
        # Extract the ERRS HDU data
        errs_data = source_hdul['ERRS'].data
        # Open the target FITS file in update mode
        target_hdul = fits.open(target_file, mode='update')
        # Replace the ERRS HDU with the new data
        target_hdul['ERRS'].data = errs_data
        # Save the changes
        target_hdul.flush()
        # Close the files
        source_hdul.close()
        target_hdul.close()

    def clip_data(self, wave, data, low_wavelength, high_wavelength):
        low_idx = (np.abs(wave - low_wavelength)).argmin()
        high_idx = (np.abs(wave - high_wavelength)).argmin()
        wave_clip = wave[low_idx:high_idx]
        data_clip = data[low_idx:high_idx]
        return wave_clip, data_clip

    def bin_data(self, wave_clip, data_clip, num_bins):
        bin_width = len(wave_clip) // num_bins
        binned_wave = []
        binned_data = []
        for bin_start in np.arange(0, len(wave_clip), bin_width):
            bin_end = bin_start + bin_width
            bin_wave = wave_clip[bin_start:bin_end]
            bin_data = data_clip[bin_start:bin_end]
            bin_mean = np.median(bin_data)
            binned_wave.append(bin_wave)
            binned_data.append(bin_mean)
        return binned_wave, binned_data

    def mask_and_compress(self, wave, data, errs, mask_ranges):
        masked_wave = wave.copy()
        for m in mask_ranges:
            masked_wave = np.ma.masked_inside(masked_wave, m[0], m[1])
        mask = masked_wave.mask
        masked_data = np.ma.masked_array(data, mask, fill_value=np.nan)
        masked_errs = np.ma.masked_array(errs, mask, fill_value=np.nan)
        compressed_wave = masked_wave.compressed()
        compressed_data = masked_data.compressed()
        compressed_errs = masked_errs.compressed()
        return compressed_wave, compressed_data, compressed_errs

    def edge_matching(self, bonus_plots=False, annotate=True, wave_include=False, dichroic=True,
                      normalize_only=True, **kwargs):

        # UV-VIS Overlap: 0.545284 - 0.555926
        # VIS-NIR Overlap: 0.994165 - 1.01988

        # Define a dictionary that maps body names to file paths
        file_paths = {
            'neptune': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Post_Molecfit\\MOV_Neptune_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Post_Molecfit\\MOV_Neptune_SCIENCE_TELLURIC_CORR_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Post_Molecfit\\MOV_Neptune_SCIENCE_TELLURIC_CORR_NIR.fits",
                'PUVB': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_1\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'PVIS': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_1\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'PNIR': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_1\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DIUVB': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_1\\MOV_Neptune_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits",
                'DIVIS': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_1\\MOV_Neptune_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits",
                'DINIR': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_1\\MOV_Neptune_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits"
        },
            'fiege-110': {
                'UVB': '/home/gmansir/Thesis/Telluric/Data/MOV_DiskIntegrated_FIEGE-110_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'VIS': '/home/gmansir/Thesis/Telluric/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-15T22:59:10.824/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_FIEGE-110_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'NIR': '/home/gmansir/Thesis/Telluric/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-15T22:34:09.646/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_FIEGE-110_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'PUVB': '/home/gmansir/Thesis/Stars/Data/PreMolecfit/PreMolecfit_FEIGE-110_UVB.fits',
                'PVIS': '/home/gmansir/Thesis/Stars/Data/PreMolecfit/PreMolecfit_FEIGE-110_VIS.fits',
                'PNIR': '/home/gmansir/Thesis/Stars/Data/PreMolecfit/PreMolecfit_FEIGE-110_NIR.fits',
                'DIUVB': '/home/gmansir/Thesis/Telluric/Data/MOV_DiskIntegrated_FIEGE-110_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'DIVIS': '/home/gmansir/Thesis/Telluric/Data/MOV_DiskIntegrated_FIEGE-110_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'DINIR': '/home/gmansir/Thesis/Telluric/Data/MOV_DiskIntegrated_FIEGE-110_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits'
            },
        }

        # Define the data types and their prefixes
        body_files = file_paths[self.body]
        data_keys = ['UVB', 'VIS', 'NIR', 'PUVB', 'PVIS', 'PNIR', 'DIUVB', 'DIVIS', 'DINIR']

        # Initialize dictionaries to store the wave and data for each data type
        wave_data = {}
        errs_data = {}
        for key in data_keys:
            wave, data, exp, errs = self.fits_get_wave_data(body_files[key])
            data = np.array(data)
            data = data / exp
            if key in ['UVB', 'VIS', 'NIR']:
                errs = np.array(errs)
                errs = np.abs(data) * np.sqrt(errs/data)**2
                errs = errs.tolist()
                ekey = 'E' + key
                errs_data[ekey] = errs
            data = data.tolist()
            wave_data[key] = [wave, data]

        # Get the file paths for the current body (and divide by telescope gain)
        self.DIUVBwave, self.DIUVdata = wave_data['DIUVB']
        self.DIUVdata = [x/2.7 for x in self.DIUVdata]
        self.DIVISwave, self.DIVISdata = wave_data['DIVIS']
        self.DIVISdata = [x/1.6 for x in self.DIVISdata]
        self.DINIRwave, self.DINIRdata = wave_data['DINIR']
        self.DINIRdata = [x/1.5 for x in self.DINIRdata]

        # Define the data types and their clip ranges as tuples (start, end)
        data_clip_ranges = {
            'UVB': (0.552, 0.555926),
            'VIS': (0.552, 0.555926),
            'VIS2': (0.994165, 1.01988),
            'NIR': (0.994165, 1.01988),
        }
        prefixes = ['', 'P', 'DI']

        # Initialize dictionaries to store the clipped wave and data for each data type
        if normalize_only == False:
            clipped_data = {}
            for p in prefixes:
                for range in data_clip_ranges.keys():
                    data_key = p+range[0:3]
                    clip_key = p+range
                    wave, data = self.clip_data(wave_data[data_key][0], wave_data[data_key][1],
                                            data_clip_ranges[range][0], data_clip_ranges[range][1])
                    clipped_data[clip_key] = (wave, data)

        if bonus_plots == True:
            fig1 = plt.figure(1)
            ax1 = fig1.add_subplot(111)
            ax1.plot(clipped_data['UVB'], linewidth=0.5)
            ax1.plot(clipped_data['VIS'], linewidth=0.5)
            ax1.set_title('UV (blue) - VIS (orange) Overlap')

            fig2 = plt.figure(2)
            ax2 = fig2.add_subplot(111)
            ax2.plot(clipped_data['NIR'], linewidth=0.5)
            ax2.plot(clipped_data['VIS2'], linewidth=0.5)
            ax2.set_title('VIS (orange) - NIR (blue) Overlap')

        if dichroic == True:

            # Read the CSV file into a DataFrame
            dc = pd.read_csv('/home/gmansir/Thesis/Dichroic/D1_and_D2_final.csv')
            dc.columns.values[0] = 'Column1'
            dc.Wave = dc.Column1*0.001
            UV_low = (np.abs(dc.Wave - wave_data['UV'][0][0])).argmin()
            UV_high = (np.abs(dc.Wave - wave_data['UV'][0][-1])).argmin()
            UV_weights= np.interp(wave_data['UV'][0], dc.Wave[UV_low:UV_high], dc.UVB[UV_low:UV_high])
            VIS_low = (np.abs(dc.Wave - wave_data['VIS'][0][0])).argmin()
            VIS_high = (np.abs(dc.Wave - wave_data['VIS'][0][-1])).argmin()
            VIS_weights= np.interp(wave_data['VIS'][0], dc.Wave[VIS_low:VIS_high], dc.VIS[VIS_low:VIS_high])
            NIR_low = (np.abs(dc.Wave - wave_data['NIR'][0][0])).argmin()
            NIR_high = (np.abs(dc.Wave - wave_data['NIR'][0][-1])).argmin()
            dc.NIR = dc.NIR.fillna(1).clip(upper=100)
            NIR_weights= np.interp(wave_data['NIR'][0], dc.Wave[NIR_low:NIR_high], dc.NIR[NIR_low:NIR_high])
            wave_data['UV'][1] = wave_data['UV'][1] / np.array(UV_weights*0.01)
            wave_data['VIS'][1] = wave_data['VIS'][1] / np.array(VIS_weights*0.01)
            wave_data['NIR'][1] = np.nan_to_num(wave_data['NIR'][1], nan=0)
            wave_data['NIR'][1] = wave_data['NIR'][1] / np.array(NIR_weights*0.01)

        # Bin data to lower resolution
        num_bins = 8

        # Initialize dictionaries to store the binned wave and data for each data type
        if normalize_only == False:
            binned_data = {}
            for key in clipped_data.keys():
                wave, data = self.bin_data(clipped_data[key][0], clipped_data[key][1], num_bins)
                binned_data[key] = (wave, data)

        if bonus_plots == True:
            ax1.plot(binned_data['UVB'], linewidth=0.5)
            ax1.plot(binned_data['VIS'], linewidth=0.5)
            ax2.plot(binned_data['NIR'], linewidth=0.5)
            ax2.plot(binned_data['VIS2'], linewidth=0.5)
            plt.show()


        # Use linear regression to find the scale factors between the two spectra and adjust accordingly
        if normalize_only == False:
            UVIS = np.polyfit(binned_data['UVB'][1][1:-1], binned_data['VIS'][1][1:-1], 1)
            VIR = np.polyfit(binned_data['NIR'][1][1:-1], binned_data['VIS2'][1][1:-1], 1)
            PUVIS = np.polyfit(binned_data['PUVB'][1][1:-1], binned_data['PVIS'][1][1:-1], 1)
            PVIR = np.polyfit(binned_data['PNIR'][1][1:-1], binned_data['PVIS2'][1][1:-1], 1)
            DIUVIS = np.polyfit(binned_data['DIUVB'][1][1:-1], binned_data['DIVIS'][1][1:-1], 1)
            DIVIR = np.polyfit(binned_data['DINIR'][1][1:-1], binned_data['DIVIS2'][1][1:-1], 1)

        if bonus_plots == True:
            y = np.polyval(UVIS, np.array(binned_data['UVB'][1]))
            fig4 = plt.figure(4)
            ax4 = fig4.add_subplot(111)
            ax4.plot(binned_data['UVB'][1][1:-1], binned_data['VIS'][1][1:-1], 'o')
            ax4.plot(binned_data['UVB'][1], y)
            ax4.set_title('UV - VIS Linear Regression')
            plt.show()

            y2 = np.polyval(VIR, np.array(binned_data['NIR'][1]))
            fig5 = plt.figure(5)
            ax5 = fig5.add_subplot(111)
            ax5.plot(binned_data['NIR'][1], binned_data['VIS2'][1], 'o')
            ax5.plot(binned_data['NIR'][1], y2)
            ax5.set_title('VIS - NIR Linear Regression')
            plt.show()

        if normalize_only == True:
            for key in data_keys:
                norm_fact = np.percentile(np.nan_to_num(wave_data[key][1], nan=0.0), 99)
                wave_data[key][1] = wave_data[key][1] / norm_fact
                if key in ['UVB', 'VIS', 'NIR']:
                    ekey = 'E'+key
                    errs_data[ekey] = np.abs(wave_data[key][1]) * np.sqrt((errs_data[ekey]/ wave_data[key][1])**2)
        else:
            wave_data['UVB'][1] = np.polyval(UVIS, np.array(wave_data['UVB'][1]))
            wave_data['NIR'][1] = np.polyval(VIR, np.array(wave_data['NIR'][1]))
            wave_data['PUVB'][1] = np.polyval(PUVIS, np.array(wave_data['PUVB'][1]))
            wave_data['PNIR'][1] = np.polyval(PVIR, np.array(wave_data['PNIR'][1]))
            wave_data['DIUVB'][1] = np.polyval(DIUVIS, np.array(wave_data['DIUVB'][1]))
            wave_data['DINIR'][1] = np.polyval(DIVIR, np.array(wave_data['DINIR'][1]))

        # Define the bands and corresponding mask ranges
        mask_ranges = {
            'UVB': [[wave_data['UVB'][0][0], 0.30501], [0.544649, wave_data['UVB'][0][-1]]],
            'VIS': [[wave_data['VIS'][0][0], 0.544649], [1.01633, wave_data['VIS'][0][-1]]],
            'NIR': [[wave_data['NIR'][0][0], 1.01633], [2.192, wave_data['NIR'][0][-1]]],
            'PUVB': [[wave_data['PUVB'][0][0], 0.30501], [0.544649, wave_data['PUVB'][0][-1]]],
            'PVIS': [[wave_data['PVIS'][0][0], 0.544649], [1.01633, wave_data['PVIS'][0][-1]]],
            'PNIR': [[wave_data['PNIR'][0][0], 1.01633], [2.192, wave_data['PNIR'][0][-1]]],
            'DIUVB': [[wave_data['DIUVB'][0][0], 0.30501], [0.544649, wave_data['DIUVB'][0][-1]]],
            'DIVIS': [[wave_data['DIVIS'][0][0], 0.544649], [1.01633, wave_data['DIVIS'][0][-1]]],
            'DINIR': [[wave_data['DINIR'][0][0], 1.01633], [2.192, wave_data['DINIR'][0][-1]]],
        }

        # Initialize lists for wave and spec
        wave = []
        spec = []
        pwave = []
        pspec = []
        diwave = []
        dispec = []
        errs = []
        self.errs_data = {}

        # Loop through the bands and mask data accordingly
        for band in data_keys:
            wave_data_band = wave_data[band]
            mask_range = mask_ranges[band]

            # Mask and compress data
            if band in ['UVB', 'VIS', 'NIR']:
                key = 'E'+band
                wave_band, spec_band, com_errs = self.mask_and_compress(wave_data_band[0], wave_data_band[1], errs_data[key], mask_range)
            else:
                key = ''
                wave_band, spec_band, com_errs = self.mask_and_compress(wave_data_band[0], wave_data_band[1], np.zeros(len(wave_data_band[0])), mask_range)

            setattr(self, band + 'wave', wave_band)
            setattr(self, band + 'data', spec_band)

            # Append data to the appropriate lists
            if band in ['UVB', 'VIS', 'NIR']:
                wave.append(wave_band)
                spec.append(spec_band)
                errs.append(com_errs)
                self.errs_data[key] = com_errs
            elif band in ['PUVB', 'PVIS', 'PNIR']:
                pwave.append(wave_band)
                pspec.append(spec_band)
            else:
                diwave.append(wave_band)
                dispec.append(spec_band)
        # The wave and spec lists now contain the masked and compressed data for each band

        wave = np.array([i for band in wave for i in band])
        spec = np.array([i for band in spec for i in band])
        errs = np.array([i for band in errs for i in band])
        self.pwave = np.array([i for band in pwave for i in band])
        self.pspec = np.array([i for band in pspec for i in band])
        self.diwave = np.array([i for band in diwave for i in band])
        self.dispec = np.array([i for band in dispec for i in band])

        # Uses the edges of the full spectrum to remove any artificial incline  or below zero value
        # due to the edge-matching (move to linear regression only)
        if normalize_only == False:
            wave_edges = [i for edge in [wave[0:1000], wave[-1000:-1]] for i in edge]
            spec_edges = [i for edge in [spec[0:1000], spec[-1000:-1]] for i in edge]
            pwave_edges = [i for edge in [self.pwave[0:1000], self.pwave[-1000:-1]] for i in edge]
            pspec_edges = [i for edge in [self.pspec[0:1000], self.pspec[-1000:-1]] for i in edge]
            diwave_edges = [i for edge in [self.diwave[0:1000], self.diwave[-1000:-1]] for i in edge]
            dispec_edges = [i for edge in [self.dispec[0:1000], self.dispec[-1000:-1]] for i in edge]
            flatten_coeff = np.polyfit(wave_edges, spec_edges, 1)
            flatten_fit = np.polyval(flatten_coeff, wave)
            spec = spec - flatten_fit
            spec += np.abs(np.min(spec)) + 0.0001
            pflatten_coeff = np.polyfit(pwave_edges, pspec_edges, 1)
            pflatten_fit = np.polyval(pflatten_coeff, self.pwave)
            self.pspec = self.pspec - pflatten_fit
            self.pspec += np.abs(np.min(self.pspec)) + 0.0001
            diflatten_coeff = np.polyfit(diwave_edges, dispec_edges, 1)
            diflatten_fit = np.polyval(diflatten_coeff, self.diwave)
            self.dispec = self.dispec - diflatten_fit
            self.dispec += np.abs(np.min(self.dispec)) + 0.0001

        # Define the output file
        clean_spec_file = os.path.join(self.work_dir, f'MOV_{self.body.title()}_SCI_IFU_FULL_SPECTRUM.fits')

        # Create HDUs
        phdu = fits.PrimaryHDU()
        flux_hdu = fits.ImageHDU(spec, name='FLUX')
        errs_hdu = fits.ImageHDU(errs, name='ERRS')
        phdu.data = wave

        hdulist = fits.HDUList([phdu, flux_hdu, errs_hdu])
        hdulist.writeto(clean_spec_file, overwrite=True)
        print(f'File Saved to: ' + self.work_dir + f'MOV_{self.body.title()}_SCI_IFU_FULL_SPECTRUM.fits')

    def compare_fit(self, annotate=False, **kwargs):

        pre = '/home/gmansir/Thesis/Titan_old/Data/PostMolecfit/MOV_Titan_SCI_IFU_FULL_SPECTRUM_PREMOLECFIT.fits'
        #default = '/home/gmansir/Thesis/Titan/Data/PostMolecfit/MOV_Titan_SCI_IFU_FULL_SPECTRUM_BETA_DEFAULTS.fits'
        #adjust = '/home/gmansir/Thesis/Titan/Data/PostMolecfit/MOV_Titan_SCI_IFU_FULL_SPECTRUM_BETA_ADJUSTED.fits'
        post = '/home/gmansir/Thesis/Titan/Data/PostMolecfit/MOV_Titan_SCI_IFU_FULL_SPECTRUM.fits'

        # Plot the results!!
        w, h = plt.figaspect(0.25)
        fig = plt.figure(1, figsize=(w, h))
        ax = fig.add_subplot(111)
        pre_hdu = fits.open(pre)
        pre_flux = pre_hdu[0].data
        pre_wav = pre_hdu[1].data
        pre_hdu.close()
        ax.plot(pre_wav, pre_flux, linewidth=0.5, color='black', label='PreMolecfit')
        for i in [[0.686294, 0.691341], [0.759164, 0.787264], [0.822648, 0.822888], [0.93059, 0.955106], [1.11, 1.16],
                  [1.33, 1.48912], [2.41, 2.48], [1.7846, 1.9654]]:
            lidx = (np.abs(pre_wav - i[0])).argmin()
            hidx = (np.abs(pre_wav - i[1])).argmin()
            ax.plot(pre_wav[lidx:hidx], pre_flux[lidx:hidx], linewidth=0.5, color='silver')
        def_hdu = fits.open(post)
        def_flux = def_hdu[0].data
        def_wav = def_hdu[1].data
        def_hdu.close()
        ax.plot(def_wav, def_flux-0.03, linewidth=0.5, label='PostMolecfit')
        for i in [[0.686294, 0.691341], [0.759164, 0.787264], [0.822648, 0.822888], [0.93059, 0.955106], [1.11, 1.16],
                  [1.33, 1.48912], [2.41, 2.48], [1.7846, 1.9654]]:
            lidx = (np.abs(def_wav - i[0])).argmin()
            hidx = (np.abs(def_wav - i[1])).argmin()
            ax.plot(def_wav[lidx:hidx], def_flux[lidx:hidx]-0.03, linewidth=0.5, color='silver')
        #adj_hdu = fits.open(adjust)
        #adj_flux = adj_hdu[0].data
        #adj_wav = adj_hdu[1].data
        #adj_hdu.close()
        #ax.plot(adj_wav, adj_flux, linewidth=0.5, label='Adjusted')
        #for i in [[0.686294, 0.691341], [0.759164, 0.787264], [0.822648, 0.822888], [0.93059, 0.955106], [1.11, 1.16],
        #          [1.33, 1.48912], [2.41, 2.48], [1.7846, 1.9654]]:
        #    lidx = (np.abs(adj_wav - i[0])).argmin()
        #    hidx = (np.abs(adj_wav - i[1])).argmin()
        #    ax.plot(adj_wav[lidx:hidx], adj_flux[lidx:hidx], linewidth=0.5, color='silver')

        ax.set_xscale('log')
        ax.set_ylim(top=1.75)
        ax.set_title(f'{self.body.title()} Fit Comparison')

        # Set x-axis tick labels to non-scientific notation
        if annotate == True:
            self.annotate_plot_clean(**kwargs)
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.3f}'))
        ax.xaxis.set_minor_formatter(StrMethodFormatter('{x:.3f}'))
        ax.legend()
        plt.show()

    def load_eso_data(self, eso_path, m, b):
        eso_data = []
        with open(eso_path, 'r') as file:
            for line in file:
                columns = line.strip().split()
                if len(columns) == 2:
                    eso_data.append((float(columns[0]), float(columns[1])))
        eso_x = np.array([x * 0.0001 for x, _ in eso_data])
        eso_y = np.array([y * m + b for _, y in eso_data])
        return eso_x, eso_y

    def telluric_standard_solutuion(self, bonus_plots=False, num_bins=20, *args, **kwargs):

        file_paths = {
            'neptune': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\FEIGE-110\\MOV_FEIGE-110_SCI_IFU_MERGE3D_DATA_OBJ_UVB_MOLECFIT_READY.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\FEIGE-110\\MOV_FEIGE-110_SCIENCE_TELLURIC_CORR_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\FEIGE-110\\MOV_FEIGE-110_SCIENCE_TELLURIC_CORR_NIR.fits"
            },
            'feige': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\FEIGE-110\\MOV_FEIGE-110_SCI_IFU_MERGE3D_DATA_OBJ_UVB_MOLECFIT_READY.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\FEIGE-110\\MOV_FEIGE-110_SCIENCE_TELLURIC_CORR_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\FEIGE-110\\MOV_FEIGE-110_SCIENCE_TELLURIC_CORR_NIR.fits"
            },
        }

        match = re.match(r"([a-z]+)", self.body)
        planet = match.group(1)

        #if self.body == 'enceladus' or self.body == 'titan' or 'saturn' or 'LTT7897' in self.body:
        #    eso_path = '/home/gmansir/Thesis/Telluric/Data/fLTT7987.dat'
        #    m = 7 * 10 ** 12
        #    b = 0.0
        #    uv_exptime = 140
        #    vis_exptime = 240
        #    nir_exptime = 240
        #    self.tell_star = 'LTT7897'
        if self.body.lower() == 'neptune' or self.body.lower() == 'feige':
            eso_path = "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\FEIGE-110\\fFeige110.dat"
            m = 3 * 10 ** 12
            b = 0.0
            uv_exptime = 120
            vis_exptime = 190
            nir_exptime = 190
            self.tell_star = 'FIEGE-110'
        #elif 'uranus' or 'GD71' in self.body:
        #    eso_path = '/home/gmansir/Thesis/Telluric/Data/fGD71.dat'
        #    m = 1 * 10 ** 13
        #    b = 0.0
        #    uv_exptime = 600
        #    vis_exptime = 600
        #    nir_exptime = 200
        #    self.tell_star = 'GD71'

        body_files = file_paths[planet]
        UVBwave, UVBdata, _, _  = self.fits_get_wave_data(body_files['UVB'])
        VISwave, VISdata, _ ,_ = self.fits_get_wave_data(body_files['VIS'])
        NIRwave, NIRdata, _, _ = self.fits_get_wave_data(body_files['NIR'])
        UVBdata = [(x/2.7) /uv_exptime for x in UVBdata]
        UVBdata = UVBdata / np.percentile(UVBdata, 99)
        VISdata = [(x/1.6) / vis_exptime for x in VISdata]
        VISdata = VISdata / np.percentile(VISdata, 99)
        NIRdata = [(x/1.5) / nir_exptime for x in NIRdata]
        NIRdata = NIRdata / np.percentile(NIRdata, 99)

        #***remove the edge matching here***
        # Clip out edge regions for scaling:
        #UVwave_clip1, UVdata_clip1 = self.clip_data(UVwave, UVdata, 0.552, 0.555926)
        #VISwave_clip1, VISdata_clip1 = self.clip_data(VISwave, VISdata, 0.552, 0.555926)
        #VISwave_clip2, VISdata_clip2 = self.clip_data(VISwave, VISdata, 0.994165, 1.01988)
        #NIRwave_clip1, NIRdata_clip1 = self.clip_data(NIRwave, NIRdata, 0.994165, 1.01988)
        # UVlow - 0.545284

        # Bin data to lower resolution
        #num_bins = 8
        #UVwave_binned, UVdata_binned = self.bin_data(UVwave_clip1, UVdata_clip1, num_bins)
        #VISwave_binned, VISdata_binned = self.bin_data(VISwave_clip1, VISdata_clip1, num_bins)
        #VISwave_binned2, VISdata_binned2 = self.bin_data(VISwave_clip2, VISdata_clip2, num_bins)
        #NIRwave_binned, NIRdata_binned = self.bin_data(NIRwave_clip1, NIRdata_clip1, num_bins)

        # Use linear regression to find the scale factors between the two spectra and adjust accordingly
        #UVIS = np.polyfit(UVdata_binned[1:-1], VISdata_binned[1:-1], 1)
        #VIR = np.polyfit(NIRdata_binned[1:-1], VISdata_binned2[1:-1], 1)
        #UVdata = np.polyval(UVIS, np.array(UVdata))
        #NIRdata = np.polyval(VIR, np.array(NIRdata))
        #******

        # Mask telluric noise
        UVBmask_ranges = [[UVBwave[0], 0.30501], [0.544649, UVBwave[-1]]]
        VISmask_ranges = [[VISwave[0], 0.544649], [1.01633, VISwave[-1]]]
        NIRmask_ranges = [[NIRwave[0], 1.01633], [2.192, NIRwave[-1]]]
        UVBwave, UVBdata, _ = self.mask_and_compress(UVBwave, UVBdata, np.zeros(len(UVBwave)), UVBmask_ranges)
        VISwave, VISdata, _ = self.mask_and_compress(VISwave, VISdata, np.zeros(len(VISwave)), VISmask_ranges)
        NIRwave, NIRdata, _ = self.mask_and_compress(NIRwave, NIRdata, np.zeros(len(NIRwave)), NIRmask_ranges)

        eso_x, eso_y = self.load_eso_data(eso_path, m, b)
        eso_x = list(eso_x)
        eso_y = list(eso_y)
        uvb_idxs = [self.closest_index(eso_x, UVBwave[0]), self.closest_index(eso_x, UVBwave[-1])]
        eso_x_uvb = eso_x[uvb_idxs[0]:uvb_idxs[-1]]
        eso_y_uvb = eso_y[uvb_idxs[0]:uvb_idxs[-1]]
        vis_idxs = [self.closest_index(eso_x, VISwave[0]), self.closest_index(eso_x, VISwave[-1])]
        eso_x_vis = eso_x[vis_idxs[0]:vis_idxs[-1]]
        eso_y_vis = eso_y[vis_idxs[0]:vis_idxs[-1]]
        nir_idxs = [self.closest_index(eso_x, NIRwave[0]), self.closest_index(eso_x, NIRwave[-1])]
        eso_x_nir = eso_x[nir_idxs[0]:nir_idxs[-1]]
        eso_y_nir = eso_y[nir_idxs[0]:nir_idxs[-1]]

        if bonus_plots == True:
            w, h = plt.figaspect(0.25)
            fig44 = plt.figure(44, figsize=(w, h))
            ax44 = fig44.add_subplot(111)
            ax44.plot(eso_x, eso_y, color='indianred', label='ESO Spectrum')

        UVBwave_binned = []
        VISwave_binned = []
        NIRwave_binned = []
        UVBdata_binned = []
        VISdata_binned = []
        NIRdata_binned = []
        UVB_coeffs = None
        VIS_coeffs = None
        NIR_coeffs = None
        UVB_yfit = None
        VIS_yfit = None
        NIR_yfit = None
        eso_xuvb_binned = []
        eso_xvis_binned = []
        eso_xnir_binned = []
        eso_yuvb_binned = []
        eso_yvis_binned = []
        eso_ynir_binned = []
        eso_uvb_coeffs = None
        eso_vis_coeffs = None
        eso_nir_coeffs = None
        eso_yuvb_fit = None
        eso_yvis_fit = None
        eso_ynir_fit = None
        UVB_results = []
        VIS_results = []
        NIR_results = []
        eso_UVB_results = []
        eso_VIS_results = []
        eso_NIR_results = []

        UVB_dict = {'wave':UVBwave, 'xbin':UVBwave_binned, 'data':UVBdata, 'ybin':UVBdata_binned,
                   'coeffs':UVB_coeffs, 'yfit':UVB_yfit, 'solution':UVB_results, 'exptime':uv_exptime}
        VIS_dict = {'wave':VISwave, 'xbin':VISwave_binned, 'data':VISdata, 'ybin':VISdata_binned,
                    'coeffs':VIS_coeffs, 'yfit':VIS_yfit, 'solution':VIS_results, 'exptime':vis_exptime}
        NIR_dict = {'wave':NIRwave, 'xbin':NIRwave_binned, 'data':NIRdata, 'ybin':NIRdata_binned,
                    'coeffs':NIR_coeffs, 'yfit':NIR_yfit, 'solution':NIR_results, 'exptime':nir_exptime}
        eso_UVB_dict = {'wave':eso_x_uvb, 'xbin':eso_xuvb_binned, 'ybin':eso_yuvb_binned, 'coeffs': eso_uvb_coeffs,
                       'yfit': eso_yuvb_fit, 'solution':eso_UVB_results, 'data':eso_y_uvb}
        eso_VIS_dict = {'wave': eso_x_vis, 'xbin': eso_xvis_binned, 'ybin': eso_yvis_binned, 'data':eso_y_vis,
                        'coeffs': eso_vis_coeffs, 'yfit': eso_yvis_fit, 'solution':eso_VIS_results}
        eso_NIR_dict = {'wave': eso_x_nir, 'xbin': eso_xnir_binned, 'ybin': eso_ynir_binned, 'data':eso_y_nir,
                        'coeffs': eso_nir_coeffs, 'yfit': eso_ynir_fit, 'solution':eso_NIR_results}
        dicts = {'Tell_UV':UVB_dict, 'Tell_VIS':VIS_dict, 'Tell_NIR':NIR_dict,
                 'eso_UV': eso_UVB_dict, 'eso_VIS': eso_VIS_dict, 'esoNIR': eso_NIR_dict}

        num_bins = num_bins
        uvb_bins = np.linspace(UVBwave[0], UVBwave[-1], num_bins + 1)
        vis_bins = np.linspace(VISwave[0], VISwave[-1], num_bins + 1)
        nir_bins = np.linspace(NIRwave[0], NIRwave[-1], num_bins + 1)

        for label, info in dicts.items():
            wav = info['wave']
            data = info['data']
            if 'UV' in label:
                my_wave = UVBwave
                bin_edges = uvb_bins
            elif 'VIS' in label:
                my_wave = VISwave
                bin_edges = vis_bins
            else:
                my_wave = NIRwave
                bin_edges = nir_bins
            xbin = info['xbin']
            ybin = info['ybin']
            coeffs = info['coeffs']
            yfit = info['yfit']
            solution = info['solution']

            #xbin = bin_edges[:-1] + np.diff(bin_edges) / 2
            bin_indices = np.digitize(wav, bins=bin_edges[:-1])
            xbin = [np.median(np.array(wav)[bin_indices == i]) for i in np.arange(1,num_bins+1)]
            ybin = [np.median(np.array(data)[bin_indices == i]) for i in np.arange(1,num_bins+1)]

            coeffs = np.polyfit(xbin, ybin, 5)
            yfit = np.polyval(coeffs, wav)
            spl = uspline(xbin, ybin, k=3, s=0.0)
            wave = wav * un.um
            flux = data * (un.erg / (un.s * un.cm ** 2 * un.um))
            spectrum = Spectrum1D(spectral_axis=wave, flux=flux)
            continuum_fit = fit_generic_continuum(spectrum)
            if 'Tell' in label:
                solution.append(spl(wav))
                if bonus_plots == True:
                    if 'UVB' in label:
                        ax44.plot(wav, data, color='grey', label='Disk Integrated Star Spectrum')
                    else:
                        ax44.plot(wav, data, color='grey')
                    if 'UVB' in label:
                        ax44.plot(xbin, ybin, 'o', color='steelblue', label='Binned Data')
                    else:
                        ax44.plot(xbin, ybin, 'o', color='steelblue')
            else:
                spline_resampled = np.interp(my_wave, wav, spl(wav))
                solution.append(spline_resampled)
                #if bonus_plots == True:
                    #ax4.plot(xbin, ybin, 'o', color='firebrick', label='np.interp')
            if bonus_plots == True:
                #ax4.plot(spectrum.spectral_axis, continuum_fit(spectrum.spectral_axis), label='Spectutils',
                #         color='rebeccapurple')
                if 'UVB' in label:
                    ax44.plot(wav, spl(wav), color='green', label='Spline')
                else:
                    ax44.plot(wav, spl(wav), color='green')
                #ax4.plot(wav, yfit, color='black', label='Polyfit')
                ax44.set_xscale('log')

        self.UVB_solution = np.array(eso_UVB_results)/np.array(UVB_results)
        self.VIS_solution = np.array(eso_VIS_results)/np.array(VIS_results)
        self.NIR_solution = np.array(eso_NIR_results)/np.array(NIR_results)

        if bonus_plots == True:
            #plt.legend()
            plt.ylim(-0.5, 1.5)
            plt.title('Response Curve Creation')
            plt.show()

            w, h = plt.figaspect(0.25)
            fig2 = plt.figure(2, figsize=(w, h))
            ax2 = fig2.add_subplot(111)
            ax2.plot(UVBwave, self.UVB_solution[0], color='green')
            ax2.plot(VISwave, self.VIS_solution[0], color='green')
            ax2.plot(NIRwave, self.NIR_solution[0], color='green')
            ax2.set_xscale('log')
            ax2.set_ylim(0.0,2.0)
            plt.title('Response Curve')
            plt.show()

            w, h = plt.figaspect(0.25)
            fig7 = plt.figure(7, figsize=(w, h))
            ax7 = fig7.add_subplot(111)
            ax7.plot(UVBwave, UVBdata * self.UVB_solution[0])
            ax7.plot(VISwave, VISdata * self.VIS_solution[0])
            ax7.plot(NIRwave, NIRdata * self.NIR_solution[0])
            ax7.set_xscale('log')
            plt.title('Telluric Solution applied to telluric star data')
            plt.show()

    def blackbody(self, wavelength, temperature):
        # Function to calculate the blackbody spectrum
        wavelength_m = wavelength * 1e-6 # Convert wavelength from um to meters
        exponent = h * c / (wavelength_m * k * temperature)
        intensity = (2 * h * c ** 2) / (wavelength_m ** 5 * (np.exp(exponent) - 1))
        return intensity

    def apply_telluric_solution(self, num_bins=40, bonus_plots=False, more_plots=False, all_plots=False):

        self.edge_matching(bonus_plots=all_plots, annotate=False, threshold=0.3, dichroic=False)
        self.telluric_standard_solutuion(bonus_plots=more_plots, num_bins=num_bins)

        UVB_object = self.UVBdata
        VIS_object = self.VISdata
        NIR_object = self.NIRdata
        UVBwave = self.UVBwave
        VISwave = self.VISwave
        NIRwave = self.NIRwave
        EUVB = self.errs_data['EUVB']
        EVIS = self.errs_data['EVIS']
        ENIR = self.errs_data['ENIR']

        def scale_factor_objective(scale_factor, data_to_scale, scale_to_data):
            # Scale the data
            scaled_data = data_to_scale * scale_factor
            # Calculate the median of the last 50 values
            dts_median = np.median(scaled_data[-50:])
            # Calculate the median of the first 50 values
            std_median = np.median(scale_to_data[:50])
            # The objective is the squared difference between the medians
            objective = (dts_median - std_median) ** 2
            return objective

        def find_scaling_factor(data_to_scale, scale_to_data):
            # Initial guess for the scaling factor
            Initial_scale_factor = 1.0
            # Minimize the objective function to find the scaling factor
            result = minimize(scale_factor_objective, Initial_scale_factor, args=(data_to_scale, scale_to_data))
            # The scaling factor is in result.x
            scaling_factor = result.x[0]
            return scaling_factor

        # Apply telluric solution
        #if self.body == 'fiege-110' or self.body == 'titan':
        UVB_spec = UVB_object * self.UVB_solution[0]
        EUVB = np.abs(UVB_spec) * np.sqrt((np.array(EUVB) / UVB_spec) ** 2)
        VIS_spec = VIS_object * self.VIS_solution[0]
        EVIS = np.abs(VIS_spec) * np.sqrt((np.array(EVIS) / VIS_spec) ** 2)
        NIR_spec = NIR_object * self.NIR_solution[0]
        ENIR = np.abs(NIR_spec) * np.sqrt((np.array(ENIR) / NIR_spec) ** 2)
        #else:
        #    UVB_spec = UVB_object / self.UVB_solution[0]
        #    VIS_spec = VIS_object / self.VIS_solution[0]
        #    NIR_spec = NIR_object / self.NIR_solution[0]
        # Normalize by finding value that 99% of the data is below
        UVB_spec = (UVB_spec) / np.percentile(UVB_spec, 99)
        EUVB = np.abs(UVB_spec) * np.sqrt((np.array(EUVB) / UVB_spec) ** 2)
        VIS_spec = (VIS_spec) / np.percentile(VIS_spec, 99)
        EVIS = np.abs(VIS_spec) * np.sqrt((np.array(EVIS) / VIS_spec) ** 2)
        norm_idx = self.closest_index(VIS_spec, np.percentile(VIS_spec, 99))
        norm_loc = VISwave[norm_idx]
        NIR_spec = (NIR_spec) / np.percentile(NIR_spec, 99)
        ENIR = np.abs(NIR_spec) * np.sqrt((np.array(ENIR) / NIR_spec) ** 2)

        # Fit, apply, and report scale factors between the UVB and NIR bands with the VIS data
        # Note that this is currently using the wrong sides for the nir and vis specs, which is why
        # the factor is so far off
        uvb_scaling_factor = find_scaling_factor(UVB_spec, VIS_spec)
        nir_scaling_factor = find_scaling_factor(VIS_spec, NIR_spec)
        UVB_spec = UVB_spec * uvb_scaling_factor
        EUVB = np.abs(UVB_spec) * np.sqrt((np.array(EUVB) / UVB_spec) ** 2)
        NIR_spec = NIR_spec / nir_scaling_factor
        ENIR = np.abs(NIR_spec) * np.sqrt((np.array(ENIR) / NIR_spec) ** 2)
        print(f'UVB Scale factor: ', uvb_scaling_factor)
        print(f'NIR Scale Factor: ', 1./nir_scaling_factor)

        # PreMolecfit
        #if bonus_plots == True:
        #    w, h = plt.figaspect(10)
        #    fig10 = plt.figure(10, figsize=(w, h))
        #    plt.subplots_adjust(hspace=0.4)
        #    ax10 = fig10.add_subplot(111)
        #    ax10.plot(self.pwave, self.pspec, color='black', linewidth=0.3)
        #    for i in [[0.686294, 0.691341], [0.759164, 0.787264], [0.822648, 0.822888], [0.93059, 0.955106], [1.11, 1.16],
        #          [1.33, 1.48912], [2.41, 2.48], [1.7846, 1.9654]]:
        #        lidx = (np.abs(self.pwave - i[0])).argmin()
        #        hidx = (np.abs(self.pwave - i[1])).argmin()
        #        ax10.plot(self.pwave[lidx:hidx], self.pspec[lidx:hidx], linewidth=0.3, color='lightgrey')
        #    ax10.set_title('Pre Molecfit')
        #    ax10.set_xscale('log')
        #    ax10.xaxis.set_major_formatter(StrMethodFormatter('{x:.3f}'))
        #    ax10.xaxis.set_minor_formatter(StrMethodFormatter('{x:.3f}'))

        # Define a function for plotting the data and light grey regions
        def plot_data_and_regions(ax, wave, data, title):
            ax.plot(wave, data, color='black', linewidth=0.3)
            corrected_regions = [[0.58,0.60],[0.64,0.66],[0.68,0.75],[0.78,0.86],[0.88, 1.0],
                                 [1.062, 1.244],[1.26,1.57],[1.63,2.48]]
            untrustworthy_regions = [[0.50,0.51],[0.758,0.77],[0.685,0.695],[0.625,0.632],[0.72,0.73], [0.81,0.83], [0.89, 0.98],
                                     [1.107, 1.164],[1.3, 1.5], [1.73, 2.0], [2.38, 2.48], [1.946,1.978],
                                     [1.997,2.032],[2.043,2.080]]
            for region in corrected_regions:
                lidx = (np.abs(wave - region[0])).argmin()
                hidx = (np.abs(wave - region[1])).argmin()
                ax.plot(wave[lidx:hidx], data[lidx:hidx], linewidth=0.3, color='#121212')
            for region in untrustworthy_regions:
                lidx = (np.abs(wave - region[0])).argmin()
                hidx = (np.abs(wave - region[1])).argmin()
                ax.plot(wave[lidx:hidx], data[lidx:hidx], linewidth=0.3, color='lightgrey')
            ax.set_title(title)
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.3f}'))
            ax.xaxis.set_minor_formatter(StrMethodFormatter('{x:.3f}'))
            #ax.set_ylim(0,1.5)

        # Create a figure
        if bonus_plots == True:
            w, h = plt.figaspect(1)
            fig11 = plt.figure(11, figsize=(w, h))
            plt.subplots_adjust(hspace=0.4)

            # Plot the Disk Integrated spectrum
            ax2 = fig11.add_subplot(311)
            plot_data_and_regions(ax2, self.diwave, self.dispec, 'Disk Integrated')

            # Plot the Disk Integrated spectrum after it's been run through Molecfit
            ax3 = fig11.add_subplot(312)
            ranges = [
                (self.UVBwave, self.UVBdata, 'Post Molecfit'),
                (self.VISwave, self.VISdata, 'Post Molecfit'),
                (self.NIRwave, self.NIRdata, 'Post Molecfit')
            ]
            for wave, data, title in ranges:
                plot_data_and_regions(ax3, wave, data, title)

        # Plot the Post Molecfit spectrum after applying the Telluric Solution
            ax11 = fig11.add_subplot(313)

        w, h = plt.figaspect(1)
        fig1 = plt.figure(1, figsize=(w, h))
        plt.subplots_adjust(hspace=0.4)
        ax4 = fig1.add_subplot(211)

        ranges = [
            (UVBwave, UVB_spec, 'Telluric Solution Applied'),
            (VISwave, VIS_spec, 'Telluric Solution Applied'),
            (NIRwave, NIR_spec, 'Telluric Solution Applied')
        ]
        if bonus_plots == True:
            for wave, data, title in ranges:
                plot_data_and_regions(ax11, wave, data, title)
            ax11.set_yscale('log')
            ax11.set_ylim(10e-4, 10e0)
        for wave, data, title in ranges:
            plot_data_and_regions(ax4, wave, data, title)
        ax4.set_yscale('log')
        ax4.set_ylim(10e-4, 10e0)

        # Add in a black body
        wavelength = np.concatenate((UVBwave, VISwave, NIRwave))
        #bb = BlackBody(temperature=5778 * un.K)
        #bb_flux = bb(bb_wave)*10e3
        #ax4.plot(bb_wave, bb_flux, label='Blackbody')
        header_info = {'BB_TEMP': 5778}
        sp = SourceSpectrum(BlackBodyNorm1D, temperature=5778)
        sp_flux = sp(wavelength * un.um)
        norm_loc = self.closest_index(wavelength, norm_loc)
        header_info = {'NORM_IDX': norm_loc}
        sp_flux = sp_flux / sp_flux[norm_loc]
        ax4.plot(wavelength, sp_flux, label='Synphot Blackbody')

        # Add in Filters
        filters = {'johnson_u':'U_PHOT', 'johnson_b':'B_PHOT', 'johnson_v':'V_PHOT',
                   'johnson_r':'R_PHOT', 'johnson_i':'I_PHOT', 'johnson_j':'J_PHOT',
                   'johnson_k':'K_PHOT' }
        # Create a spectrum object with unit info
        spectrum = np.concatenate((UVB_spec, VIS_spec, NIR_spec))
        wave_ang = wavelength * 10000 * u.angstrom
        spectrum = spectrum * units.FLAM
        spectrum_obj = SourceSpectrum(Empirical1D, points=wave_ang, lookup_table=spectrum)

        for f in filters.keys():
            # Load a filter
            band = SpectralElement.from_filter(f)
            # Determine the color
            observation = Observation(spectrum_obj, band, force='taper')
            color = observation.effstim(flux_unit='flam')
            location = observation.effective_wavelength()
            # Add it to the plot
            ax4.plot(location/10000, color, 'o', color='orangered')
            if bonus_plots == True:
                ax11.plot(location / 10000, color, 'o', color='orangered')
            # Add it to the FITS header
            header_info[filters[f]] = color.value

        ax5 = fig1.add_subplot(212)
        plot_data_and_regions(ax5, wavelength, spectrum/sp_flux, 'Albedo')
        ax5.set_yscale('log')
        ax5.set_ylim(10e-4, 10e0)

        # Add a figure title
        if bonus_plots == True:
            current_date = datetime.now().strftime('%Y-%m-%d')
            #fig11.suptitle(f'{self.body.title()}\nGenerated on: {current_date}, '
            #          f'Added in filter colors and Albedo')

        # Prep for FITS file
        flux = np.array(spectrum / sp_flux)
        wav = np.array(wavelength)
        errors = np.concatenate((EUVB, EVIS, ENIR))
        dont_trust = [[0.50, 0.51], [0.758, 0.77], [0.685, 0.695], [0.625, 0.632], [0.72, 0.73], [0.81, 0.83],
                      [0.89, 0.98],
                      [1.107, 1.164], [1.3, 1.5], [1.73, 2.0], [2.38, 2.48], [1.946, 1.978],
                      [1.997, 2.032], [2.043, 2.080]]

        corrected = [[0.58, 0.60], [0.64, 0.66], [0.68, 0.75], [0.78, 0.86], [0.88, 1.0],
                     [1.062, 1.244], [1.26, 1.57], [1.63, 2.48]]
        dont_trust_mask = np.zeros_like(wav, dtype=bool)
        corrected_mask = np.zeros_like(wav, dtype=bool)
        for d in dont_trust:
            dont_trust_mask |= ma.masked_inside(wav, d[0], d[1]).mask
        for c in corrected:
            corrected_mask |= ma.masked_inside(wav, c[0], c[1]).mask
        header_info['FLUX_STD'] = ''
        # Save in FITS File
        albedo_file = self.work_dir + f'MOV_{self.body.title()}_SCI_IFU_ALBEDO.fits'

        # Create new HDUList
        phdu = fits.PrimaryHDU()
        hdu = fits.HDUList([phdu])

        # Append empty image HDUs with proper names
        for name in ['FLUX', 'ERRS', 'RESP_CRV', 'MASK_TEL', 'MASK_WRN']:
            new_hdu = fits.ImageHDU(name=name)
            hdu.append(new_hdu)

        # Assign data
        hdu[0].data = wav
        hdu[1].data = spectrum.value
        hdu[2].data = errors
        hdu[3].data = np.concatenate((self.UVB_solution[0], self.VIS_solution[0], self.NIR_solution[0]))
        hdu[4].data = dont_trust_mask.astype(int)
        hdu[5].data = corrected_mask.astype(int)

        # Set header info
        for label in header_info.keys():
            hdu[0].header[label] = header_info[label]
        hdu[0].header['FLUX_STD'] = self.tell_star

        # Write to disk, overwriting if file exists
        hdu.writeto(albedo_file, overwrite=True)
        print(f'File Saved to: ' + self.work_dir + f'MOV_{self.body.title()}_SCI_IFU_ALBEDO.fits')

        #investigate nir 1.1 - 1.25 and snr difference between nir and vis/uv

if __name__ == "__main__":

    object_list = ['neptune']

    for o in object_list:
        obj = DataAnalysis(body=o)
        obj.apply_telluric_solution(num_bins=40, bonus_plots=True, more_plots=True)
        plt.clf()

# einstein a with oscillator strength -> try to change the cross-section for each planet (linear?)
# level 1 einstein coefficient
# 2 correcting temperature and abundance for each planet (multiplicative)
# 3 full radiative transfer model

# browse planetary spectra modeling tools (google, ads, papers, etc (BART, FORMOSA (look at temperature ranges and other parameters))
# investigate which molecules are important for which planets (set defaults) by eye and lit

# Consider airmass when applying to objects
# (check if eso has more images so i can interpolate solutions for other airmasses, otherwise google for oother solutions)

# Look up saturn image positions to make sure molecfit is consistent for offsets that will be combined
# Look up XSHOOTER lit to see what other people do to deal with the telluric correction

# Thesis checklist/timeline:
# 2 months after submission for beuracracy
# 3 months intensive thesis writing + paper
# 4 months - basic modeling
# 2 months - final prep of data (visual and tables)

# Multiply a single value to see if we can align the different bands
    # try to come up with an instrumental reason as to why the telluric solution works so well for the
    # star and not for the extended sources and why a single value is appropriate if so

# Plot with pre and post molecfit to show correction of specific telluric molecules
# iraf documentation for response curve?
# Search ESO archive for more tellurics (XSHOOTER, IFU prefered) in the nights surrounding ours

# observing log table dates, object, tellurics, conditions
# black body units
# telluric star summary plot