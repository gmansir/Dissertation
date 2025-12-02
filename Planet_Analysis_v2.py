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
from scipy.optimize import curve_fit
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

        """

        self.body = None
        celestial_bodies = ['neptune', 'feige-110', 'titan', 'titan_ltt7897', 'hip09', 'saturn', 'uranus']
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

        self.work_dir = f"C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\{self.body.title()}\\Post_Molecfit\\"

        self.file_paths = {
            'neptune': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Post_Molecfit\\MOV_Neptune_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Post_Molecfit\\MOV_Neptune_SCIENCE_TELLURIC_CORR_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Post_Molecfit\\MOV_Neptune_SCIENCE_TELLURIC_CORR_NIR.fits",
                'PUVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_1\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'PVIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_1\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'PNIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_1\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DIUVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_1\\MOV_Neptune_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits",
                'DIVIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_1\\MOV_Neptune_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits",
                'DINIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\Offset_1\\MOV_Neptune_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits"
        },
            'feige-110': {
                'UVB': r"C:\Users\ninam\Documents\Chile_Stuff\Dissertation\Spec_Files\Neptune\Post_Molecfit\MOV_FEIGE-110_SCIENCE_TELLURIC_CORR_UVB.fits",
                'VIS': r"C:\Users\ninam\Documents\Chile_Stuff\Dissertation\Spec_Files\Neptune\Post_Molecfit\MOV_FEIGE-110_SCIENCE_TELLURIC_CORR_VIS.fits",
                'NIR': r"C:\Users\ninam\Documents\Chile_Stuff\Dissertation\Spec_Files\Neptune\Post_Molecfit\MOV_FEIGE-110_SCIENCE_TELLURIC_CORR_NIR.fits",
                'PUVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\FEIGE-110\\FEIGE-110_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'PVIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\FEIGE-110\\FEIGE-110_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'PNIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\FEIGE-110\\FEIGE-110_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DIUVB': r"C:\Users\ninam\Documents\Chile_Stuff\Dissertation\Spec_Files\Neptune\FEIGE-110\FEIGE-110_onoff_IFU_FLUX_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits",
                'DIVIS': r"C:\Users\ninam\Documents\Chile_Stuff\Dissertation\Spec_Files\Neptune\FEIGE-110\FEIGE-110_onoff_IFU_FLUX_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits",
                'DINIR': r"C:\Users\ninam\Documents\Chile_Stuff\Dissertation\Spec_Files\Neptune\FEIGE-110\FEIGE-110_onoff_IFU_FLUX_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits"
            },
            'saturn': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_4\\MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Post_Molecfit\\MOV_Saturn_SCIENCE_TELLURIC_CORR_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Post_Molecfit\\MOV_Saturn_SCIENCE_TELLURIC_CORR_NIR.fits",
                'PUVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_4\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'PVIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_4\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'PNIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_4\\MOV_Saturn_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DIUVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_4\\MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits",
                'DIVIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_4\\MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits",
                'DINIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Offset_4\\MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits"
        },
            'titan': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_1\\MOV_Titan_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Post_Molecfit\\MOV_Titan_SCIENCE_TELLURIC_CORR_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Post_Molecfit\\MOV_Titan_SCIENCE_TELLURIC_CORR_NIR.fits",
                'PUVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_1\\MOV_Titan_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'PVIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_1\\MOV_Titan_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'PNIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_1\\MOV_Titan_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DIUVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_1\\MOV_Titan_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits",
                'DIVIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_1\\MOV_Titan_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits",
                'DINIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Titan\\Offset_1\\MOV_Titan_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits"
        },
            'hip09': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\LTT7987\\LTT7987_onoff_IFU_FLUX_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Standard2\\Hip095318_TELL_IFU_SCIENCE_TELLURIC_CORR_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Standard2\\Hip095318_TELL_IFU_SCIENCE_TELLURIC_CORR_VIS.fits",
                'PUVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\LTT7987\\LTT7987_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'PVIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Standard2\\Hip095318_TELL_IFU_TELL_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'PNIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Standard2\\Hip095318_TELL_IFU_TELL_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'DIUVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\LTT7987\\LTT7987_onoff_IFU_FLUX_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits",
                'DIVIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Standard2\\Hip095318_TELL_IFU_TELL_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits",
                'DINIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Standard2\\Hip095318_TELL_IFU_TELL_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits"
            },
            'uranus': {
                'UVB': r"C:\Users\ninam\Documents\Chile_Stuff\Dissertation\Spec_Files\Uranus\Offset_1\MOV_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits",
                'VIS': r"C:\Users\ninam\Documents\Chile_Stuff\Dissertation\Spec_Files\Uranus\Offset_1\MOV_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits",
                'NIR': r"C:\Users\ninam\Documents\Chile_Stuff\Dissertation\Spec_Files\Uranus\Offset_1\MOV_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits",
                'PUVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_1\\MOV_Uranus_2_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'PVIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_1\\MOV_Uranus_2_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'PNIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_1\\MOV_Uranus_2_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DIUVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_1\\MOV_Uranus_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits",
                'DIVIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_1\\MOV_Uranus_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits",
                'DINIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Offset_1\\MOV_Uranus_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits"
            },
            'gd71': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\GD71\\GD71_onoff_IFU_FLUX_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\FEIGE-110\\MOV_GD71_SCIENCE_TELLURIC_CORR_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\FEIGE-110\\MOV_GD71-110_SCIENCE_TELLURIC_CORR_NIR.fits",
                'PUVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\GD71\\GD71_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'PVIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\GD71\\GD71_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'PNIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\GD71\\GD71_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DIUVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\GD71\\GD71_onoff_IFU_FLUX_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits",
                'DIVIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\GD71\\GD71_onoff_IFU_FLUX_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits",
                'DINIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\GD71\\GD71_onoff_IFU_FLUX_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits"
            },
        }

        print(f"Instance initiated for {self.body.title()}.")

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

    def extract_fits_data(self, file_path):
        """
        Extracts the wavelength array, flux data, exposure time, and error data
        from a FITS file containing a 3D spectral cube.

        Parameters
        ----------
        file_path : str
            Path to the FITS file to read.

        Returns
        -------
        wave : np.ndarray
            1D array of wavelength values in microns.
        data : list
            1D list of median flux values along the spatial axes.
        EXPTIME : float
            Exposure time from the FITS header.
        errs : list
            1D list of median errors along the spatial axes (zeros if not present).
        """

        # Opens the FITS file and collapses the data into a 1D spectrum
        with fits.open(file_path) as hdul:
            data = hdul[0].data
            data = np.nanmedian(data, axis=(1, 2))
            #data = self.sigma_clipping_1d(data)
            data = data.tolist()

            # Attempt to read the error extension 'ERRS'
            try:
                errs = hdul['ERRS'].data
                errs = np.nan_to_num(errs)
                errs = np.nanmedian(errs, axis=(1, 2))
                errs = errs.tolist()
            except (KeyError, IndexError):
                # If the extension doesn't exist, use zeros as placeholders
                errs = np.zeros(len(data))

            # Extract wavelength calibration from FITS header
            header = hdul[0].header
            CRVAL3 = header['CRVAL3']  # Reference wavelength (start)
            CDELT3 = header['CDELT3']  # Wavelength step per pixel
            NAXIS3 = header['NAXIS3']  # Number of wavelength points
            EXPTIME = header['EXPTIME']  # Exposure time
            air_start = header['ESO TEL AIRM START']
            air_end = header['ESO TEL AIRM END']
            AIRMASS = (air_start + air_end)/2
            try:
                ADUtoE = header['ESO DET OUT1 CONAD'] # Conversion from ADU to electrons
            except (KeyError, IndexError):
                ADUtoE = header['ESO DET CHIP GAIN']

            # Construct the wavelength array and convert to microns
            wave = np.array([CRVAL3 + CDELT3 * i for i in range(NAXIS3)]) / 1000.

            low_lim = self.closest_index(wave, 0.31)

            print(f"File: {file_path}")
            print(f"EXPTIME: {EXPTIME}")
            print(f"AIRMASS: {AIRMASS}")
            print(f"ADUtoE: {ADUtoE}")

        return wave[low_lim:], data[low_lim:], EXPTIME, errs[low_lim:], ADUtoE, AIRMASS

    def prep_xshoo_data(self):
        """
        Load all available spectra for a given body into dictionaries,
        with optional gain scaling applied.

        Args:
            body (str): Name of the body to load (must be in file_paths).

        Returns:
            wave_data (dict): {key: [wave, flux]} for each data type.
            errs_data (dict): {E<key>: errors} for science arms (UVB, VIS, NIR).
        """

        # Atmospheric extinction correction prep
        fname_ext = "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\extinction.dat"
        ext_data = []
        with open(fname_ext, 'r') as file:
            for line in file:
                columns = line.strip().split()
                if len(columns) == 2:
                    ext_data.append((float(columns[0]), float(columns[1])))
        ext_x = np.array([x * 0.0001 for x, _ in ext_data])
        ext_y = np.array([y for _, y in ext_data])
        ext_wav = np.linspace(0.30, 2.50, 2200)  # desired full grid in microns

        # model: Rayleigh-like term (lambda^-4) + aerosol power law
        def k_model(lams, A, B, alpha):
            return A * lams ** (-4.0) + B * lams ** (alpha)

        # fit only in 0.40 - 0.90 um as recommended by Patat
        mask_fit = (ext_x >= 0.311) & (ext_x <= 0.90)
        x_fit = ext_x[mask_fit]
        y_fit = ext_y[mask_fit]

        # initial guesses: A ~ value at blue times lambda^4, B small, alpha ~ -1
        p0 = [0.01, 0.02, -1.0]
        popt, pcov = curve_fit(k_model, x_fit, y_fit, p0=p0, maxfev=10000)
        A, B, alpha = popt

        # extrapolate full-model
        k_extrap = k_model(ext_wav, A, B, alpha)

        # apply empirical blue deficit per Patat: ramp from 0 at 0.40 um to max_deficit at 0.37 um
        max_deficit = 0.03  # mag/airmass at 0.37 um per Patat
        deficit = np.zeros_like(ext_wav)
        # linear ramp between 0.40 -> 0.37, then hold to shorter lambda (or taper further if desired)
        idx_040 = np.searchsorted(ext_wav, 0.40)
        idx_037 = np.searchsorted(ext_wav, 0.37)
        if idx_037 < idx_040:
            ramp = np.linspace(0.0, max_deficit, idx_040 - idx_037, endpoint=True)
            deficit[idx_037:idx_040] = ramp[::-1]  # increase deficit towards shorter lambda
            deficit[:idx_037] = max_deficit  # maintain max deficit at bluest end

        # corrected extinction curve
        k_full = k_extrap - deficit

        # --- enforce physical floor (no negative extinction) ---
        k_full = np.where(k_full < 0.0, 0.0, k_full)

        # --- define a physically-motivated NIR baseline (mag/airmass) ---
        # tweak these values if you have better anchors (Lombardi, TAPAS, Molecfit continuum)
        nir_baseline = np.zeros_like(ext_wav)
        nir_baseline[(ext_wav >= 0.90) & (ext_wav < 1.30)] = 0.06
        nir_baseline[(ext_wav >= 1.30) & (ext_wav < 1.80)] = 0.03
        nir_baseline[(ext_wav >= 1.80) & (ext_wav <= 2.50)] = 0.015

        # for wavelengths <0.90 use the model (k_full) and for >0.9 we'll blend to baseline

        # --- create a smooth blending weight function w(λ) in [0,1] ---
        # w=1 -> use optical model; w=0 -> use NIR baseline
        blend_start = 0.90  # µm, where blending begins
        blend_end = 1.60  # µm, where blending ends (choose based on how quickly you want to hand off)

        # logistic or raised-cosine? use a raised-cosine for a smooth compact support transition
        def raised_cosine_weight(lam, a, b):
            """
            returns weight that is 1 for lam<=a, 0 for lam>=b, and follows a half cosine in between.
            """
            w = np.zeros_like(lam)
            idx1 = lam <= a
            idx3 = lam >= b
            idx2 = (~idx1) & (~idx3)
            w[idx1] = 1.0
            # half-cosine ramp down
            x = (lam[idx2] - a) / (b - a)  # 0..1
            w[idx2] = 0.5 * (1.0 + np.cos(np.pi * x))  # goes 1 -> 0 smoothly
            w[idx3] = 0.0
            return w

        w = raised_cosine_weight(ext_wav, blend_start, blend_end)

        # --- form the blended curve ---
        k_blend = w * k_full + (1.0 - w) * nir_baseline

        # --- optional mild smoothing to remove small-scale ripples (preserve broad shape) ---
        # choose an odd window length that is << length of blend region; convert to nearest odd integer
        def _odd(n):
            n = int(n)
            return n + 1 if n % 2 == 0 else n

        # smoothing window in pixels: pick about 0.01-0.03 micron equivalent or a small fraction of your grid
        # compute delta lambda per pixel
        dlam = ext_wav[1] - ext_wav[0]
        savgol_width_um = 0.01  # smooth scale ~ 0.01 micron (tune as needed)
        savgol_window = _odd(max(5, int(round(savgol_width_um / dlam))))
        savgol_poly = 2

        try:
            k_smoothed = savgol_filter(k_blend, window_length=savgol_window, polyorder=savgol_poly, mode='interp')
        except Exception:
            # fallback: if filter fails (too short array), skip smoothing
            k_smoothed = k_blend.copy()

        # final safety: enforce non-negative and small monotonic floor
        k_final = np.where(k_smoothed < 0.0, 0.0, k_smoothed)# --- enforce physical floor (no negative extinction) ---
        k_full = np.where(k_full < 0.0, 0.0, k_full)

        # --- define a physically-motivated NIR baseline (mag/airmass) ---
        # tweak these values if you have better anchors (Lombardi, TAPAS, Molecfit continuum)
        nir_baseline = np.zeros_like(ext_wav)
        nir_baseline[(ext_wav >= 0.90) & (ext_wav < 1.30)] = 0.06
        nir_baseline[(ext_wav >= 1.30) & (ext_wav < 1.80)] = 0.03
        nir_baseline[(ext_wav >= 1.80) & (ext_wav <= 2.50)] = 0.015

        # for wavelengths <0.90 use the model (k_full) and for >0.9 we'll blend to baseline

        # --- create a smooth blending weight function w(λ) in [0,1] ---
        # w=1 -> use optical model; w=0 -> use NIR baseline
        blend_start = 0.90  # µm, where blending begins
        blend_end = 1.60  # µm, where blending ends (choose based on how quickly you want to hand off)

        # logistic or raised-cosine? use a raised-cosine for a smooth compact support transition
        def raised_cosine_weight(lam, a, b):
            """
            returns weight that is 1 for lam<=a, 0 for lam>=b, and follows a half cosine in between.
            """
            w = np.zeros_like(lam)
            idx1 = lam <= a
            idx3 = lam >= b
            idx2 = (~idx1) & (~idx3)
            w[idx1] = 1.0
            # half-cosine ramp down
            x = (lam[idx2] - a) / (b - a)  # 0..1
            w[idx2] = 0.5 * (1.0 + np.cos(np.pi * x))  # goes 1 -> 0 smoothly
            w[idx3] = 0.0
            return w

        w = raised_cosine_weight(ext_wav, blend_start, blend_end)

        # --- form the blended curve ---
        k_blend = w * k_full + (1.0 - w) * nir_baseline

        # --- optional mild smoothing to remove small-scale ripples (preserve broad shape) ---
        # choose an odd window length that is << length of blend region; convert to nearest odd integer
        def _odd(n):
            n = int(n)
            return n + 1 if n % 2 == 0 else n

        # smoothing window in pixels: pick about 0.01-0.03 micron equivalent or a small fraction of your grid
        # compute delta lambda per pixel
        dlam = ext_wav[1] - ext_wav[0]
        savgol_width_um = 0.01  # smooth scale ~ 0.01 micron (tune as needed)
        savgol_window = _odd(max(5, int(round(savgol_width_um / dlam))))
        savgol_poly = 2

        try:
            k_smoothed = savgol_filter(k_blend, window_length=savgol_window, polyorder=savgol_poly, mode='interp')
        except Exception:
            # fallback: if filter fails (too short array), skip smoothing
            k_smoothed = k_blend.copy()

        # final safety: enforce non-negative and small monotonic floor
        k_final = np.where(k_smoothed < 0.0, 0.0, k_smoothed)

        # produce a conservative uncertainty array:
        # start with fit covariance propagated, then add a systematic floor in blue and NIR
        #k_sig = np.interp(ext_wav, x_fit, np.sqrt(np.diag(pcov)).mean())  # crude baseline
        #k_sig += (ext_wav < 0.40) * 0.03  # extra blue systematic
        #k_sig += (ext_wav > 0.90) * 0.02  # extra NIR systematic (tune as needed)

        # Apply the extinction curve correction

        body_files = self.file_paths[self.body]
        data_keys = list(body_files.keys())

        wave_data = {}
        errs_data = {}

        for key in data_keys:
            wave, data, exp, errs, gain, airmass = self.extract_fits_data(body_files[key])

            # Apply exposure time and gain corrections
            data = np.array(data) / exp
            data = data * gain
            errs = np.array(errs) / exp
            errs = errs * gain

            k_arm = np.interp(wave, ext_wav, k_final)
            data = data * 10**(0.4 * k_arm * airmass)
            errs = errs * 10**(0.4 * k_arm * airmass)

            # Store flux + wavelength
            wave_data[key] = [wave, data.tolist()]

            # Store errors only for science frames
            if key in ['UVB', 'VIS', 'NIR']:
                errs_data['E' + key] = errs.tolist()

        return wave_data, errs_data

    def load_eso_data(self, eso_path):
        eso_data = []
        with open(eso_path, 'r') as file:
            for line in file:
                columns = line.strip().split()
                if len(columns) == 2:
                    eso_data.append((float(columns[0]), float(columns[1])))
        eso_x = np.array([x * 0.0001 for x, _ in eso_data])
        eso_y = np.array([y for _, y in eso_data])
        return eso_x, eso_y

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
        wave_clip = np.array(wave_clip)
        data_clip = np.array(data_clip)
        bin_width = len(wave_clip) // num_bins
        binned_wave = []
        binned_data = []
        for bin_start in range(0, len(wave_clip), bin_width):
            bin_end = min(bin_start + bin_width, len(wave_clip))
            bin_wave = wave_clip[bin_start:bin_end]
            bin_data = data_clip[bin_start:bin_end]
            binned_wave.append(np.median(bin_wave))  # representative wavelength
            binned_data.append(np.nanmedian(bin_data))  # median of data in bin
        return binned_wave, binned_data

    def mask_and_compress(self, wave, data, mask_ranges):
        # Create masked wave based on mask_ranges
        masked_wave = wave.copy()
        for m in mask_ranges:
            masked_wave = np.ma.masked_inside(masked_wave, m[0], m[1])

        # Combine masks: edges + NaNs in data or errs
        nan_mask = np.isnan(data) #| np.isnan(errs)
        combined_mask = masked_wave.mask | nan_mask

        # Create masked arrays
        masked_data = np.ma.masked_array(data, mask=combined_mask, fill_value=np.nan)
        #masked_errs = np.ma.masked_array(errs, mask=combined_mask, fill_value=np.nan)

        # Compress arrays to remove masked elements
        compressed_wave = masked_wave.data[~combined_mask]
        compressed_data = masked_data.data[~combined_mask]
        #compressed_errs = masked_errs.data[~combined_mask]

        return compressed_wave, compressed_data #, compressed_errs

    def blackbody(self, wavelength, temperature):
        # Function to calculate the blackbody spectrum
        wavelength_m = wavelength * 1e-6 # Convert wavelength from um to meters
        exponent = h * c / (wavelength_m * k * temperature)
        intensity = (2 * h * c ** 2) / (wavelength_m ** 5 * (np.exp(exponent) - 1))
        return intensity

    def edge_matching(self, wave_data, bonus_plots=False, dichroic=True, normalize_only=True, save=False, **kwargs):

        # UV-VIS Overlap: 0.545284 - 0.555926
        # VIS-NIR Overlap: 0.994165 - 1.01988

        # Define the data types and their clip ranges as tuples (start, end)
        data_clip_ranges = {
            'UVB': (0.552, 0.555926),
            'VIS': (0.552, 0.555926),
            'VIS2': (0.994165, 1.015),
            'NIR': (0.994165, 1.015),
        }

        # Initialize dictionaries to store the clipped wave and data for each data type
        if not normalize_only:
            clipped_data = {}
            for band, clip_range in data_clip_ranges.items():
                data_key = ('VIS' if band == 'VIS2' else band)
                clip_key = band
                wave, data = self.clip_data(
                    wave_data[data_key][0],
                    wave_data[data_key][1],
                    clip_range[0],
                    clip_range[1]
                )
                clipped_data[clip_key] = (wave, data)

            if bonus_plots:
                fig1, ax1 = plt.subplots(1, 1)
                ax1.plot(clipped_data['UVB'][0], clipped_data['UVB'][1], linewidth=0.8, color='slateblue')
                ax1.plot(clipped_data['VIS'][0], clipped_data['VIS'][1], linewidth=0.8, color='olivedrab')
                ax1.set_title('UV (blue) - VIS (green) Overlap')

                fig2, ax2 = plt.subplots(1, 1)
                ax2.plot(clipped_data['VIS2'][0], clipped_data['VIS2'][1], linewidth=0.8, color='olivedrab')
                ax2.plot(clipped_data['NIR'][0], clipped_data['NIR'][1], linewidth=0.8, color='sienna')
                ax2.set_title('VIS (green) - NIR (red) Overlap')

        if dichroic:
            # Read the dichroic CSV and convert wavelength to microns
            dc = pd.read_csv("C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\D1_and_D2_final.csv")
            dc.rename(columns={dc.columns[0]: 'Wavelength'}, inplace=True)
            dc['Wave'] = dc['Wavelength'] * 0.001  # convert to microns

            for band in ['UVB', 'VIS', 'NIR']:
                # Get the wave grid for this band
                band_key = band[2:]
                err_key = 'E' + band[2:]
                wave = wave_data[band][0]
                flux = np.array(wave_data[band][1], dtype=float)
                #errs = np.array(errs_data[err_key], dtype=float)

                # Clip dichroic table to wavelength range of this band
                idx_low = np.abs(dc['Wave'] - wave[0]).argmin()
                idx_high = np.abs(dc['Wave'] - wave[-1]).argmin() + 1  # include last point
                dc_band = dc.iloc[idx_low:idx_high]

                # Fill NaNs and clip extreme values if needed
                band_column = band_key
                dc_band.loc[:, band_column] = dc_band[band_column].fillna(1).clip(upper=100)

                # Interpolate dichroic transmission to the wave grid
                weights = np.interp(wave, dc_band['Wave'], dc_band[band_column])

                # Apply correction
                corrected_flux = np.nan_to_num(flux / (weights * 0.01))  # convert percent to factor
                wave_data[band][1] = corrected_flux.tolist()
                #corrected_errs = np.nan_to_num(errs / (weights * 0.01))
                #errs_data[band] = corrected_errs.tolist()

        # Initialize dictionaries to store the binned wave and data for each data type
        if not normalize_only:
            num_bins = 8
            binned_data = {}
            for key in clipped_data.keys():
                wave, data = self.bin_data(clipped_data[key][0], clipped_data[key][1], num_bins)
                binned_data[key] = (wave, data)

            if bonus_plots:
                ax1.plot(binned_data['UVB'][0], binned_data['UVB'][1], linewidth=0.5, color='cornflowerblue')
                ax1.plot(binned_data['VIS'][0], binned_data['VIS'][1], linewidth=0.5, color='yellowgreen')
                ax2.plot(binned_data['VIS2'][0], binned_data['VIS2'][1], linewidth=0.5, color='yellowgreen')
                ax2.plot(binned_data['NIR'][0], binned_data['NIR'][1], linewidth=0.5, color='coral')
                plt.show()

            # Define the pairs of keys to fit, along with a name for the fit
            scalefactor_pairs = {
                'UVIS': ('UVB', 'VIS'),
                'VIR': ('NIR', 'VIS2')
            }

            # Compute linear regression for each pair
            scalefactors = {}
            for name, (x_key, y_key) in scalefactor_pairs.items():
                x_data = np.asarray(binned_data[x_key][1][1:-1], dtype=float).ravel()
                y_data = np.asarray(binned_data[y_key][1][1:-1], dtype=float).ravel()

                # Compute median and MAD for both x and y
                x_med, y_med = np.nanmedian(x_data), np.nanmedian(y_data)
                x_mad = 1.4826 * np.nanmedian(np.abs(x_data - x_med))
                y_mad = 1.4826 * np.nanmedian(np.abs(y_data - y_med))

                # Protect against zero MADs (flat data)
                x_mad = x_mad if x_mad > 0 else np.nanstd(x_data)
                y_mad = y_mad if y_mad > 0 else np.nanstd(y_data)

                # Mark points that are "reasonable" in both x and y
                threshold = 4.0  # how aggressive to be; 3–3.5 usually works
                good = (np.abs(x_data - x_med) <= threshold * x_mad) & \
                       (np.abs(y_data - y_med) <= threshold * y_mad)

                # Remove outliers from both arrays
                x_clean, y_clean = x_data[good], y_data[good]

                scalefactors[name] = np.polyfit(x_clean, y_clean, 1)

            if bonus_plots:
                y = np.polyval(scalefactors['UVIS'], np.array(binned_data['UVB'][1]))
                fig4 = plt.figure(4)
                ax4 = fig4.add_subplot(111)
                ax4.plot(binned_data['UVB'][1][1:-1], binned_data['VIS'][1][1:-1], 'o')
                ax4.plot(binned_data['UVB'][1], y)
                ax4.set_title('UV - VIS Linear Regression')
                plt.show()

                y2 = np.polyval(scalefactors['VIR'], np.array(binned_data['NIR'][1]))
                fig5 = plt.figure(5)
                ax5 = fig5.add_subplot(111)
                ax5.plot(binned_data['NIR'][1], binned_data['VIS2'][1], 'o')
                ax5.plot(binned_data['NIR'][1], y2)
                ax5.set_title('VIS - NIR Linear Regression')
                plt.show()

        if normalize_only:
            for key in data_keys:
                flux = np.array(wave_data[key][1])
                norm_fact = np.percentile(np.nan_to_num(flux, nan=0.0), 99)
                # Avoid division by zero
                wave_data[key][1] = np.where(norm_fact != 0, flux / norm_fact, flux)
                #if key in ['UVB', 'VIS', 'NIR']:
                #    ekey = 'E' + key
                #    errs = np.array(errs_data[ekey])
                #    scaled_errs = np.where(
                #        wave_data[key][1] != 0,
                #        np.abs(wave_data[key][1]) * np.sqrt((errs / wave_data[key][1]) ** 2),
                #        0.0
                #    )
                #    errs_data[ekey] = scaled_errs.tolist()

        else:
            wave_data['UVB'][1] = np.polyval(scalefactors['UVIS'], np.array(wave_data['UVB'][1]))
            wave_data['NIR'][1] = np.polyval(scalefactors['VIR'], np.array(wave_data['NIR'][1]))

        # Define the bands and corresponding mask ranges
        mask_ranges = {
            'UVB': [[wave_data['UVB'][0][0], 0.30501], [0.544649, wave_data['UVB'][0][-1]]],
            'VIS': [[wave_data['VIS'][0][0], 0.544649], [1.01633, wave_data['VIS'][0][-1]]],
            'NIR': [[wave_data['NIR'][0][0], 1.01633], [2.192, wave_data['NIR'][0][-1]]]
        }

        # Initialize lists for wave and spec
        wave = []
        spec = []
        #errs = []
        #self.errs_data = {}

        # Loop through the bands and mask data accordingly
        data_keys = wave_data.keys()
        for band in data_keys:
            wave_data_band = wave_data[band]
            mask_range = mask_ranges[band]

            # Fetch errors if they exist, otherwise zeros
            #errs_band = errs_data.get('E' + band, np.zeros(len(wave_data_band[0])))

            wave_band, spec_band = self.mask_and_compress(
                wave_data_band[0],
                wave_data_band[1],
                mask_range
            )

            setattr(self, f"{band}wave", wave_band)
            setattr(self, f"{band}data", spec_band)

            # Append to appropriate containers
            if band in ['UVB', 'VIS', 'NIR']:
                wave.append(wave_band)
                spec.append(spec_band)
             #   errs.append(com_errs)
             #   self.errs_data['E' + band] = com_errs

        # The wave and spec lists now contain the masked and compressed data for each band

        wave = np.concatenate(wave)
        spec = np.concatenate(spec)
        #errs = np.concatenate(errs)

        if bonus_plots:
            fig1, axes = plt.subplots(1, 1)
            axes.plot(wave, spec, linewidth=0.8, color='slateblue', alpha=0.5, label='Before Incline Adjust')
            axes.set_xlabel('Wavelength (microns)')
            plt.suptitle('Edge Matching Results')

        # Uses the edges of the full spectrum to remove any artificial incline or below zero value
        # due to the edge-matching (move to linear regression only)
        def flatten_edges(wave_arr, spec_arr, n_points=1000, eps=1e-4):
            """Remove artificial slope using edges of the spectrum."""
            # make sure n_points doesn't exceed array size
            n_points = min(n_points, len(wave_arr) // 2)

            # grab edges
            wave_edges = np.concatenate([wave_arr[:n_points], wave_arr[-n_points:]])
            spec_edges = np.concatenate([spec_arr[:n_points], spec_arr[-n_points:]])

            # remove NaNs
            mask = ~np.isnan(spec_edges) & ~np.isnan(wave_edges)
            wave_edges = wave_edges[mask]
            spec_edges = spec_edges[mask]

            # fit & subtract slope
            coeff = np.polyfit(wave_edges, spec_edges, 1)
            fit = np.polyval(coeff, wave_arr)
            spec_corr = spec_arr - fit

            # shift so min > 0
            spec_corr += np.abs(np.nanmin(spec_corr)) + eps
            return spec_corr

        # Uses the edges of the full spectrum to remove any artificial incline
        if not normalize_only:
            spec = flatten_edges(wave, spec)

        if bonus_plots:
            axes.plot(wave, spec, linewidth=0.8, color='blue', alpha=0.5, label='After')
            plt.show()


        if save:
            # Define the output file
            clean_spec_file = os.path.join(self.work_dir, f'MOV_{self.body.title()}_SCI_IFU_FULL_SPECTRUM.fits')
            # Make sure the directory exists
            os.makedirs(self.work_dir, exist_ok=True)

            # Create HDUs
            phdu = fits.PrimaryHDU()
            phdu.data = wave
            flux_hdu = fits.ImageHDU(spec, name='FLUX')
            errs_hdu = fits.ImageHDU(errs, name='ERRS')

            hdulist = fits.HDUList([phdu, flux_hdu, errs_hdu])

            # Write to file (overwrite if exists)
            hdulist.writeto(clean_spec_file, overwrite=True)
            print(f'File saved to: {clean_spec_file}')

        return wave, spec

    def create_response_curve(self, bonus_plots=False, num_bins=20, *args, **kwargs):

        files = {
            'neptune': {
                'UVB': r"C:\Users\ninam\Documents\Chile_Stuff\Dissertation\Spec_Files\Neptune\FEIGE-110\FEIGE-110_onoff_IFU_FLUX_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits",
                'VIS': r"C:\Users\ninam\Documents\Chile_Stuff\Dissertation\Spec_Files\Neptune\Post_Molecfit\MOV_FEIGE-110_SCIENCE_TELLURIC_CORR_VIS.fits",
                'NIR': r"C:\Users\ninam\Documents\Chile_Stuff\Dissertation\Spec_Files\Neptune\Post_Molecfit\MOV_FEIGE-110_SCIENCE_TELLURIC_CORR_NIR.fits"
            },
            'feige': {
                'UVB': r"C:\Users\ninam\Documents\Chile_Stuff\Dissertation\Spec_Files\Neptune\FEIGE-110\FEIGE-110_onoff_IFU_FLUX_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits",
                'VIS': r"C:\Users\ninam\Documents\Chile_Stuff\Dissertation\Spec_Files\Neptune\Post_Molecfit\MOV_FEIGE-110_SCIENCE_TELLURIC_CORR_VIS.fits",
                'NIR': r"C:\Users\ninam\Documents\Chile_Stuff\Dissertation\Spec_Files\Neptune\Post_Molecfit\MOV_FEIGE-110_SCIENCE_TELLURIC_CORR_NIR.fits"
            },
            'uranus': {
                'UVB': r"C:\Users\ninam\Documents\Chile_Stuff\Dissertation\Spec_Files\Neptune\FEIGE-110\FEIGE-110_onoff_IFU_FLUX_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits",
                'VIS': r"C:\Users\ninam\Documents\Chile_Stuff\Dissertation\Spec_Files\Neptune\Post_Molecfit\MOV_FEIGE-110_SCIENCE_TELLURIC_CORR_VIS.fits",
                'NIR': r"C:\Users\ninam\Documents\Chile_Stuff\Dissertation\Spec_Files\Neptune\Post_Molecfit\MOV_FEIGE-110_SCIENCE_TELLURIC_CORR_NIR.fits"
            },
            'titan': {
                'UVB': r"C:\Users\ninam\Documents\Chile_Stuff\Dissertation\Spec_Files\Neptune\FEIGE-110\FEIGE-110_onoff_IFU_FLUX_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits",
                'VIS': r"C:\Users\ninam\Documents\Chile_Stuff\Dissertation\Spec_Files\Neptune\Post_Molecfit\MOV_FEIGE-110_SCIENCE_TELLURIC_CORR_VIS.fits",
                'NIR': r"C:\Users\ninam\Documents\Chile_Stuff\Dissertation\Spec_Files\Neptune\Post_Molecfit\MOV_FEIGE-110_SCIENCE_TELLURIC_CORR_NIR.fits"
            },
            'hip': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\LTT7987\\LTT7987_onoff_IFU_FLUX_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Standard2\\Hip095318_TELL_IFU_SCIENCE_TELLURIC_CORR_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Standard2\\Hip095318_TELL_IFU_SCIENCE_TELLURIC_CORR_VIS.fits",
            },
            'gd71': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\GD71\\GD71_onoff_IFU_FLUX_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\GD71\\MOV_GD71_SCIENCE_TELLURIC_CORR_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\GD71\\MOV_GD71_SCIENCE_TELLURIC_CORR_NIR.fits"
            },
            'saturn': {
                'UVB': r"C:\Users\ninam\Documents\Chile_Stuff\Dissertation\Spec_Files\Neptune\FEIGE-110\FEIGE-110_onoff_IFU_FLUX_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits",
                'VIS': r"C:\Users\ninam\Documents\Chile_Stuff\Dissertation\Spec_Files\Neptune\Post_Molecfit\MOV_FEIGE-110_SCIENCE_TELLURIC_CORR_VIS.fits",
                'NIR': r"C:\Users\ninam\Documents\Chile_Stuff\Dissertation\Spec_Files\Neptune\Post_Molecfit\MOV_FEIGE-110_SCIENCE_TELLURIC_CORR_NIR.fits"
            },
        }

        match = re.match(r"([a-z]+)", self.body)
        planet = match.group(1)

        if self.body == 'enceladus' or self.body == 'titan' or 'saturn' or 'LTT7897' in self.body:
            eso_path = 'C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\LTT7987\\fLTT7987.dat'
            self.tell_star = 'LTT7897'
        if self.body.lower() == 'neptune' or self.body.lower() == 'titan' or self.body.lower() == 'feige-110':
            eso_path = "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Neptune\\FEIGE-110\\fFeige110.dat"
            self.tell_star = 'FIEGE-110'
        elif self.body.lower() == 'hip09':
            eso_path = "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\LTT7987\\fLTT7987.dat"
            self.tell_star = 'hip09'
        elif 'uranus' or 'gd71' in self.body:
            eso_path = "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\GD71\\fGD71.dat"
            self.tell_star = 'gd71'

        body_files = files[planet]
        UVBwave, UVBdata, uvb_exptime, _, uvb_ADUtoE, uvb_airmass  = self.extract_fits_data(body_files['UVB'])
        VISwave, VISdata, vis_exptime, _, vis_ADUtoE, vis_airmass = self.extract_fits_data(body_files['VIS'])
        NIRwave, NIRdata, nir_exptime, _, nir_ADUtoE, nir_airmass = self.extract_fits_data(body_files['NIR'])
        #UVBdata = np.array(UVBdata) / 2.7 / uvb_exptime
        #UVBdata = UVBdata / np.nanpercentile(UVBdata, 99)
        #VISdata = np.array(VISdata) / 1.6 / vis_exptime
        #VISdata = VISdata / np.nanpercentile(VISdata, 99)
        #NIRdata = np.array(NIRdata) / 1.5 / nir_exptime
        #NIRdata = NIRdata / np.nanpercentile(NIRdata, 99)

        # Convert from counts (ADU) to ADU/sec and then to electrons/sec
        UVBdata = (np.array(UVBdata) / uvb_exptime) * uvb_ADUtoE
        VISdata = (np.array(VISdata) / vis_exptime) * vis_ADUtoE
        NIRdata = (np.array(NIRdata) / nir_exptime) * nir_ADUtoE

        fig, ax = plt.subplots(1, 1)
        ax.plot(UVBwave, UVBdata, color='b', label='UVB')
        ax.plot(VISwave, VISdata, color='g', label='VIS')
        ax.plot(NIRwave, NIRdata, color='r', label='NIR')
        ax.set_xlabel('Wavelength (microns)')
        ax.set_ylabel('electrons/sec')
        ax.set_ylim(bottom=0, top=1000)
        fig.suptitle('Standard Star Observation')
        plt.show()

        # Atmospheric extinction correction
        fname_ext = "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\extinction.dat"
        ext_data = []
        with open(fname_ext, 'r') as file:
            for line in file:
                columns = line.strip().split()
                if len(columns) == 2:
                    ext_data.append((float(columns[0]), float(columns[1])))
        ext_x = np.array([x * 0.0001 for x, _ in ext_data])
        ext_y = np.array([y for _, y in ext_data])
        ext_wav = np.linspace(0.30, 2.50, 2200)  # desired full grid in microns

        # model: Rayleigh-like term (lambda^-4) + aerosol power law
        def k_model(lams, A, B, alpha):
            return A * lams ** (-4.0) + B * lams ** (alpha)

        # fit only in 0.40 - 0.90 um as recommended by Patat
        mask_fit = (ext_x >= 0.311) & (ext_x <= 0.90)
        x_fit = ext_x[mask_fit]
        y_fit = ext_y[mask_fit]

        # initial guesses: A ~ value at blue times lambda^4, B small, alpha ~ -1
        p0 = [0.01, 0.02, -1.0]
        popt, pcov = curve_fit(k_model, x_fit, y_fit, p0=p0, maxfev=10000)
        A, B, alpha = popt

        # extrapolate full-model
        k_extrap = k_model(ext_wav, A, B, alpha)

        # apply empirical blue deficit per Patat: ramp from 0 at 0.40 um to max_deficit at 0.37 um
        max_deficit = 0.03  # mag/airmass at 0.37 um per Patat
        deficit = np.zeros_like(ext_wav)
        # linear ramp between 0.40 -> 0.37, then hold to shorter lambda (or taper further if desired)
        idx_040 = np.searchsorted(ext_wav, 0.40)
        idx_037 = np.searchsorted(ext_wav, 0.37)
        if idx_037 < idx_040:
            ramp = np.linspace(0.0, max_deficit, idx_040 - idx_037, endpoint=True)
            deficit[idx_037:idx_040] = ramp[::-1]  # increase deficit towards shorter lambda
            deficit[:idx_037] = max_deficit  # maintain max deficit at bluest end

        # corrected extinction curve
        k_full = k_extrap - deficit

        # --- enforce physical floor (no negative extinction) ---
        k_full = np.where(k_full < 0.0, 0.0, k_full)

        # --- define a physically-motivated NIR baseline (mag/airmass) ---
        # tweak these values if you have better anchors (Lombardi, TAPAS, Molecfit continuum)
        nir_baseline = np.zeros_like(ext_wav)
        nir_baseline[(ext_wav >= 0.90) & (ext_wav < 1.30)] = 0.06
        nir_baseline[(ext_wav >= 1.30) & (ext_wav < 1.80)] = 0.03
        nir_baseline[(ext_wav >= 1.80) & (ext_wav <= 2.50)] = 0.015

        # for wavelengths <0.90 use the model (k_full) and for >0.9 we'll blend to baseline

        # --- create a smooth blending weight function w(λ) in [0,1] ---
        # w=1 -> use optical model; w=0 -> use NIR baseline
        blend_start = 0.90  # µm, where blending begins
        blend_end = 1.60  # µm, where blending ends (choose based on how quickly you want to hand off)

        # logistic or raised-cosine? use a raised-cosine for a smooth compact support transition
        def raised_cosine_weight(lam, a, b):
            """
            returns weight that is 1 for lam<=a, 0 for lam>=b, and follows a half cosine in between.
            """
            w = np.zeros_like(lam)
            idx1 = lam <= a
            idx3 = lam >= b
            idx2 = (~idx1) & (~idx3)
            w[idx1] = 1.0
            # half-cosine ramp down
            x = (lam[idx2] - a) / (b - a)  # 0..1
            w[idx2] = 0.5 * (1.0 + np.cos(np.pi * x))  # goes 1 -> 0 smoothly
            w[idx3] = 0.0
            return w

        w = raised_cosine_weight(ext_wav, blend_start, blend_end)

        # --- form the blended curve ---
        k_blend = w * k_full + (1.0 - w) * nir_baseline

        # --- optional mild smoothing to remove small-scale ripples (preserve broad shape) ---
        # choose an odd window length that is << length of blend region; convert to nearest odd integer
        def _odd(n):
            n = int(n)
            return n + 1 if n % 2 == 0 else n

        # smoothing window in pixels: pick about 0.01-0.03 micron equivalent or a small fraction of your grid
        # compute delta lambda per pixel
        dlam = ext_wav[1] - ext_wav[0]
        savgol_width_um = 0.01  # smooth scale ~ 0.01 micron (tune as needed)
        savgol_window = _odd(max(5, int(round(savgol_width_um / dlam))))
        savgol_poly = 2

        try:
            k_smoothed = savgol_filter(k_blend, window_length=savgol_window, polyorder=savgol_poly, mode='interp')
        except Exception:
            # fallback: if filter fails (too short array), skip smoothing
            k_smoothed = k_blend.copy()

        # final safety: enforce non-negative and small monotonic floor
        k_final = np.where(k_smoothed < 0.0, 0.0, k_smoothed)

        # --- diagnostic plot (overlay original, baseline, and final) ---
        plt.figure(figsize=(8, 4))
        plt.plot(ext_wav, k_full, color='C0', lw=1.2, label='optical model')
        plt.plot(ext_wav, nir_baseline, color='C3', lw=1.2, label='nir model')
        plt.plot(ext_wav, k_final, color='k', lw=2.0, label='Final')
        plt.axvline(blend_start, color='0.5', ls='--', lw=0.7)
        plt.axvline(blend_end, color='0.5', ls='--', lw=0.7)
        plt.xlabel('Wavelength (µm)')
        plt.ylabel('Extinction (mag/airmass)')
        plt.legend(loc='upper right')
        plt.title('Smoothed extinction Curve')
        plt.tight_layout()
        plt.show()

        # produce a conservative uncertainty array:
        # start with fit covariance propagated, then add a systematic floor in blue and NIR
        #k_sig = np.interp(ext_wav, x_fit, np.sqrt(np.diag(pcov)).mean())  # crude baseline
        #k_sig += (ext_wav < 0.40) * 0.03  # extra blue systematic
        #k_sig += (ext_wav > 0.90) * 0.02  # extra NIR systematic (tune as needed)

        # Apply the extinction curve correction
        k_uvb = np.interp(UVBwave, ext_wav, k_full)
        k_vis = np.interp(VISwave, ext_wav, k_full)
        k_nir = np.interp(NIRwave, ext_wav, k_full)

        UVBdata = UVBdata * 10**(0.4 * k_uvb * uvb_airmass)
        VISdata = VISdata * 10**(0.4 * k_vis * vis_airmass)
        NIRdata = NIRdata * 10**(0.4 * k_nir * nir_airmass)

        # Mask telluric noise
        UVBmask_ranges = [[UVBwave[0], 0.30501], [0.544649, UVBwave[-1]]]
        VISmask_ranges = [[VISwave[0], 0.544649], [1.01633, VISwave[-1]]]
        NIRmask_ranges = [[NIRwave[0], 1.01633], [2.192, NIRwave[-1]]]
        UVBwave, UVBdata = self.mask_and_compress(UVBwave, UVBdata, UVBmask_ranges)
        VISwave, VISdata = self.mask_and_compress(VISwave, VISdata, VISmask_ranges)
        NIRwave, NIRdata = self.mask_and_compress(NIRwave, NIRdata, NIRmask_ranges)

        fig, ax = plt.subplots(1, 1)
        ax.plot(UVBwave, UVBdata, color='b', label='UVB')
        ax.plot(VISwave, VISdata, color='g', label='VIS')
        ax.plot(NIRwave, NIRdata, color='r', label='NIR')
        ax.set_xlabel('Wavelength (microns)')
        ax.set_ylabel('electrons/sec')
        ax.set_ylim(bottom=0, top=1000)
        fig.suptitle('Standard Star Observation Corrected for Atmospheric Extinction')
        plt.show()




        # Official Standard Star Data Prep
        eso_x, eso_y = self.load_eso_data(eso_path)
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

        # params
        num_bins = int(num_bins)  # keep your existing variable name
        tiny_floor = 1e-30  # safe floor to avoid division by zero
        clip_for_plot = None  # set a numeric value if you want to cap plot y-axis

        # helper: median-binning (returns bin centers and median values; skips empty bins)
        def median_bin(wav, flux, bin_edges):
            """
            Median-bin a spectrum. Works even if wav/flux come in as lists.
            Returns:
                x_bin_fit, y_bin_fit  (good points only)
                x_bin_all, y_bin_all  (include NaNs for empty bins)
            """
            # ensure numpy arrays
            wav = np.asarray(wav)
            flux = np.asarray(flux)

            # which bin each wavelength falls into
            inds = np.digitize(wav, bin_edges[:-1])

            # prepare arrays for all bins
            num_bins = len(bin_edges) - 1
            centers = np.full(num_bins, np.nan)
            medians = np.full(num_bins, np.nan)

            # fill bins
            for i in range(1, num_bins + 1):
                sel = (inds == i)
                if np.any(sel):
                    centers[i - 1] = np.median(wav[sel])
                    medians[i - 1] = np.median(flux[sel])

            # mask valid bins for fitting
            good = np.isfinite(centers) & np.isfinite(medians)
            return centers[good], medians[good], centers, medians

        # helper: fit spline to binned data and evaluate on a desired grid
        def fit_spline_from_bins(x_bin, y_bin, eval_grid, k=3, s=0.0):
            # require at least k+1 points
            if len(x_bin) < (k + 1):
                # fallback: linear interp of raw points (eval_grid must be inside range)
                return np.interp(eval_grid, x_bin, y_bin, left=np.nan, right=np.nan)
            spl = uspline(x_bin, y_bin, k=k, s=s)
            y_eval = spl(eval_grid)
            # enforce non-negative continuum: floor tiny negative wiggles to zero
            y_eval = np.where(y_eval < 0.0, 0.0, y_eval)
            return y_eval

        # helper: safe division for response construction
        def safe_response(eso_resampled, obs_resampled, wave, mask_min_fraction=0.5):
            """
            eso_resampled : reference flux (physical units)
            obs_resampled : observed counts-style flux (e- / s / um)
            wave : wavelengths corresponding to resampled arrays
            returns response and mask of valid points
            """
            # mask where observed is <= 0 or NaN -> cannot divide
            valid = np.isfinite(eso_resampled) & np.isfinite(obs_resampled) & (obs_resampled > tiny_floor)
            # warn if too few valid points
            frac_valid = np.sum(valid) / len(valid)
            if frac_valid < mask_min_fraction:
                print(f"Warning: only {frac_valid * 100:.1f}% valid points for response construction; check inputs.")
            response = np.full_like(eso_resampled, np.nan)
            response[valid] = eso_resampled[valid] / obs_resampled[valid]
            # fill small gaps by interpolation on log-scale where possible (avoid huge spikes)
            good_idx = np.where(valid)[0]
            if good_idx.size >= 2:
                interp_vals = np.interp(np.arange(len(wave)), good_idx, response[good_idx])
                # do not overwrite original good pixels; only fill NaNs
                nan_mask = ~valid
                response[nan_mask] = interp_vals[nan_mask]
            else:
                # not enough anchors to interpolate; leave NaNs
                pass
            # floor tiny or negative responses (should be positive)
            response = np.where(response <= 0, np.nan, response)
            return response, valid

        # ---------- UVB block ----------
        uvb_bins = np.linspace(UVBwave[0], UVBwave[-1], num_bins + 1)
        # OBSERVED (your integrated standard or telluric-corrected star)
        xbin_uvb, ybin_uvb, raw_centers_uvb, raw_medians_uvb = median_bin(UVBwave_obs := UVBwave,
                                                                          UVBdata_obs := UVBdata, uvb_bins)
        uvb_spline_on_obs = fit_spline_from_bins(xbin_uvb, ybin_uvb, UVBwave, k=3,
                                                 s=0.0)  # evaluated on original observed wav
        uvb_resampled_obs = fit_spline_from_bins(xbin_uvb, ybin_uvb, UVBwave, k=3, s=0.0)  # identical here
        # ESO reference (physical flux)
        xbin_eso_uvb, ybin_eso_uvb, _, _ = median_bin(eso_x_uvb, eso_y_uvb, uvb_bins)
        uvb_spline_eso = fit_spline_from_bins(xbin_eso_uvb, ybin_eso_uvb, UVBwave, k=3, s=0.0)

        # build response (S = F_eso / C_obs)
        UVB_solution, UVB_valid = safe_response(uvb_spline_eso, uvb_resampled_obs, UVBwave)
        # diagnostics plot
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 5))
        ax[0].plot(UVBwave, UVBdata, color='0.5', label='Observed (raw)')
        ax[0].plot(UVBwave, uvb_resampled_obs, color='C1', label='Obs spline')
        ax[1].plot(UVBwave, uvb_spline_eso, color='C3', label='ESO spline')
        ax[0].legend(loc='upper right')
        ax[1].legend(loc='upper right')
        ax[0].set_ylabel('flux (obs / ESO units)')
        ax[2].plot(UVBwave, UVB_solution, color='k', label='Response S(λ)')
        ax[2].set_ylabel('S (erg cm^-2 / e^-)')
        ax[2].set_xlabel('Wavelength (um)')
        ax[2].legend(loc='upper right')
        plt.suptitle('UVB: binned spline fits and response')
        plt.tight_layout()
        # save into object
        self.UVB_solution = UVB_solution
        self.UVB_solution_wave = UVBwave

        # ---------- VIS block ----------
        vis_bins = np.linspace(VISwave[0], VISwave[-1], num_bins + 1)
        xbin_vis, ybin_vis, _, _ = median_bin(VISwave_obs := VISwave, VISdata_obs := VISdata, vis_bins)
        vis_resampled_obs = fit_spline_from_bins(xbin_vis, ybin_vis, VISwave, k=3, s=0.0)
        xbin_eso_vis, ybin_eso_vis, _, _ = median_bin(eso_x_vis, eso_y_vis, vis_bins)
        vis_spline_eso = fit_spline_from_bins(xbin_eso_vis, ybin_eso_vis, VISwave, k=3, s=0.0)

        VIS_solution, VIS_valid = safe_response(vis_spline_eso, vis_resampled_obs, VISwave)
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 5))
        ax[0].plot(VISwave, VISdata, color='0.5', label='Observed (raw)')
        ax[0].plot(VISwave, vis_resampled_obs, color='C1', label='Obs spline')
        ax[1].plot(VISwave, vis_spline_eso, color='C3', label='ESO spline')
        ax[0].legend(loc='upper right')
        ax[1].legend(loc='upper right')
        ax[0].set_ylabel('flux (obs / ESO units)')
        ax[2].plot(VISwave, VIS_solution, color='k', label='Response S(λ)')
        ax[2].set_ylabel('S (erg cm^-2 / e^-)')
        ax[2].set_xlabel('Wavelength (um)')
        ax[2].legend(loc='upper right')
        plt.suptitle('VIS: binned spline fits and response')
        plt.tight_layout()
        self.VIS_solution = VIS_solution
        self.VIS_solution_wave = VISwave

        # ---------- NIR block ----------
        # ---------- NIR block with explicit telluric masking ----------
        # telluric ranges to exclude from the fit (microns)
        telluric_ranges = [(1.30, 1.50), (1.80, 2.00)]

        # copy original arrays (do not overwrite originals)
        nir_wav = np.asarray(NIRwave)
        nir_flux = np.asarray(NIRdata)
        eso_nir_wav = np.asarray(eso_x_nir)
        eso_nir_flux = np.asarray(eso_y_nir)

        # build a boolean mask that is True where data is *good* (outside telluric ranges)
        def make_telluric_mask(wav, ranges):
            mask = np.ones_like(wav, dtype=bool)
            for lo, hi in ranges:
                mask &= ~((wav >= lo) & (wav <= hi))
            return mask

        mask_obs = make_telluric_mask(nir_wav, telluric_ranges)
        mask_eso = make_telluric_mask(eso_nir_wav, telluric_ranges)

        # create masked versions with NaNs inside telluric bands (so median_bin will skip them)
        nir_flux_masked = nir_flux.copy()
        nir_flux_masked[~mask_obs] = np.nan
        eso_nir_flux_masked = eso_nir_flux.copy()
        eso_nir_flux_masked[~mask_eso] = np.nan

        # bin edges and median binning on masked data
        nir_bins = np.linspace(nir_wav[0], nir_wav[-1], num_bins + 1)
        xbin_nir, ybin_nir, all_centers_nir, all_medians_nir = median_bin(nir_wav, nir_flux_masked, nir_bins)
        xbin_eso_nir, ybin_eso_nir, _, _ = median_bin(eso_nir_wav, eso_nir_flux_masked, nir_bins)

        # fit splines only on the good binned anchors
        nir_resampled_obs = fit_spline_from_bins(xbin_nir, ybin_nir, nir_wav, k=3, s=0.0)
        nir_spline_eso = fit_spline_from_bins(xbin_eso_nir, ybin_eso_nir, nir_wav, k=3, s=0.0)

        # build raw response S = F_eso / C_obs (will be NaN where obs or eso were NaN)
        NIR_solution, NIR_valid = safe_response(nir_spline_eso, nir_resampled_obs, nir_wav)

        # now fill the masked telluric regions *in the response* via smooth interpolation
        # do interpolation in log-space to avoid creating bumps and to respect multiplicative scaling
        def fill_response_by_log_interp(wav, response, mask_ranges):
            resp = np.array(response, copy=True)
            # mask invalid points (NaN or <= 0)
            valid = np.isfinite(resp) & (resp > 0)
            # if too few good anchors, return original (caller should inspect)
            if np.sum(valid) < 3:
                return resp, valid
            # location indices
            x = np.arange(len(wav))
            # log of response at valid points
            log_resp = np.log(resp[valid])
            # interpolate log-response across all wavelengths
            interp_log = np.interp(x, x[valid], log_resp)
            interp_resp = np.exp(interp_log)
            # for every telluric range, replace response by the interpolated value
            replaced_mask = np.zeros_like(resp, dtype=bool)
            for lo, hi in mask_ranges:
                region = (wav >= lo) & (wav <= hi)
                if np.any(region):
                    resp[region] = interp_resp[region]
                    replaced_mask |= region
            return resp, replaced_mask

        NIR_solution_filled, replaced_mask = fill_response_by_log_interp(nir_wav, NIR_solution, telluric_ranges)

        # mark replaced pixels in a quality array (create or augment an existing qual array)
        # If you have a per-wavelength quality array for the solution, modify it; else create one.
        try:
            NIR_solution_qual  # if exists
        except NameError:
            # make a quality array parallel to solution (0 = good)
            NIR_solution_qual = np.zeros_like(NIR_solution_filled, dtype=int)

        REPLACED_TELLURIC_FLAG = 32  # choose an unused bit
        NIR_solution_qual[replaced_mask] |= REPLACED_TELLURIC_FLAG

        # optional: inflate errors/variance in the replaced regions if you track a variance array for S(λ)
        # if you have a NIR_solution_var array, do:
        #   NIR_solution_var[replaced_mask] *= 4.0

        # diagnostics plot
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(9, 6))
        ax[0].plot(nir_wav, nir_flux, color='0.5', label='Observed (raw)')
        ax[0].plot(nir_wav, nir_resampled_obs, color='C1', label='Obs spline (masked tellurics)')
        ax[1].plot(nir_wav, nir_spline_eso, color='C3', label='ESO spline (masked tellurics)')
        for lo, hi in telluric_ranges:
            ax[0].axvspan(lo, hi, color='red', alpha=0.12)
        ax[0].legend(loc='upper right')
        ax[0].set_ylabel('flux (obs / ESO units)')
        ax[1].legend(loc='upper right')

        ax[2].plot(nir_wav, NIR_solution_filled, color='k', label='Response S(λ) (filled)')
        ax[2].plot(nir_wav[~NIR_valid], np.zeros(np.sum(~NIR_valid)), 'x', color='orange', label='raw invalid points')
        ax[2].plot(nir_wav[replaced_mask], NIR_solution_filled[replaced_mask], 'o', color='red', ms=3,
                   label='filled tellurics')
        ax[2].set_ylabel('S (erg cm^-2 / e^-)')
        ax[2].set_xlabel('Wavelength (um)')
        ax[2].legend(loc='upper right')
        plt.suptitle('NIR: telluric-masked spline & response (masked regions filled by interpolation)')
        plt.tight_layout()
        plt.show()

        # store results back to object
        self.NIR_solution = NIR_solution_filled
        self.NIR_solution_wave = nir_wav
        self.NIR_solution_qual = NIR_solution_qual

        if bonus_plots == True:
            w, h = plt.figaspect(0.5)
            fig2 = plt.figure(2, figsize=(w, h))
            ax2 = fig2.add_subplot(111)
            ax2.plot(UVBwave, self.UVB_solution, color='green')
            ax2.plot(VISwave, self.VIS_solution, color='green')
            ax2.plot(NIRwave, self.NIR_solution, color='green')
            plt.title('Response Curve')
            plt.show()

            w, h = plt.figaspect(0.5)
            fig7 = plt.figure(7, figsize=(w, h))
            ax7 = fig7.add_subplot(111)
            ax7.plot(UVBwave, UVBdata * self.UVB_solution)
            ax7.plot(VISwave, VISdata * self.VIS_solution)
            ax7.plot(NIRwave, NIRdata * self.NIR_solution)
            ax7.set_xscale('log')
            plt.title('Telluric Solution applied to telluric star data')
            plt.show()

    def sigma_clipping_1d(self, spec, outlier_threshold=5, window=100):
        """
        Sigma-clipping for a 1D spectrum (flux, variance) using a sliding window.

        Args:
            outlier_threshold (float): Outlier threshold in units of IQR.
            window (int): Size of the wavelength window for local statistics.

        Returns:
            None: Updates self.sig_clip_data (flux) and self.sig_clip_var (variance).
        """
        spec = np.array(spec)
        if spec.ndim != 1:
            raise ValueError(f"Spectral data must be 1D array, not {spec.ndim}.")

        n_lambda = spec.shape[0]
        clean_data = spec.copy()

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

        return clean_data

    def apply_response_curve(self, num_bins=40, bonus_plots=False, tell_cor_plots=False, edge_plots=False):

        self.create_response_curve(bonus_plots=tell_cor_plots, num_bins=num_bins)

        wave_data, errs_data = self.prep_xshoo_data()

        raw_data = {'UVB': wave_data['PUVB'], 'VIS': wave_data['PVIS'], 'NIR': wave_data['PNIR']}
        di_data = {'UVB': wave_data['DIUVB'], 'VIS': wave_data['DIVIS'], 'NIR': wave_data['DINIR']}
        post_mol_data = {'UVB': wave_data['UVB'], 'VIS': wave_data['VIS'], 'NIR': wave_data['NIR']}
        errs_data = {'UVB': errs_data['EUVB'], 'VIS': errs_data['EVIS'], 'NIR': errs_data['ENIR']}

        raw_wave, raw_spec = self.edge_matching(raw_data, normalize_only=False, dichroic=False, bonus_plots=False, annotate=False, threshold=0.3)
        di_wave, di_spec = self.edge_matching(di_data, normalize_only=False, dichroic=False, bonus_plots=False, annotate=False, threshold=0.3)
        post_mol_wave, post_mol_spec = self.edge_matching(post_mol_data, normalize_only=False, bonus_plots=False, annotate=False, threshold=0.3, dichroic=False)

        # Interpolate telluric solutions onto object wavelength grids
        UVB_interp = interp1d(self.UVB_solution_wave, self.UVB_solution, kind='linear', bounds_error=False,
                              fill_value=np.nan)
        VIS_interp = interp1d(self.VIS_solution_wave, self.VIS_solution, kind='linear', bounds_error=False,
                              fill_value=np.nan)
        NIR_interp = interp1d(self.NIR_solution_wave, self.NIR_solution, kind='linear', bounds_error=False,
                              fill_value=np.nan)
        UVB_sol_on_obj = UVB_interp(post_mol_data['UVB'][0])
        VIS_sol_on_obj = VIS_interp(post_mol_data['VIS'][0])
        NIR_sol_on_obj = NIR_interp(post_mol_data['NIR'][0])

        # Apply response curve
        UVB_spec = post_mol_data['UVB'][1] * UVB_sol_on_obj
        VIS_spec = post_mol_data['VIS'][1] * VIS_sol_on_obj
        NIR_spec = post_mol_data['NIR'][1] * NIR_sol_on_obj
        #EUVB = np.abs(UVB_spec) * np.sqrt((np.array(EUVB) / UVB_spec) ** 2)
        #EVIS = np.abs(VIS_spec) * np.sqrt((np.array(EVIS) / VIS_spec) ** 2)
        #ENIR = np.abs(NIR_spec) * np.sqrt((np.array(ENIR) / NIR_spec) ** 2)

        fig, ax = plt.subplots(2, 1, sharex=True)
        plt.subplots_adjust(hspace=0.1)
        ax[0].plot(post_mol_data['UVB'][0], post_mol_data['UVB'][1], linewidth=0.5, color='blue')
        ax[1].plot(post_mol_data['UVB'][0], UVB_spec, linewidth=0.5, color='blue')
        ax[0].set_ylabel('e/s')
        ax[1].set_ylabel('ergs')
        ax[1].set_xlabel('Wavelength')
        fig.suptitle('UVB')
        plt.show()

        fig, ax = plt.subplots(2, 1, sharex=True)
        plt.subplots_adjust(hspace=0.1)
        ax[0].plot(post_mol_data['VIS'][0], post_mol_data['VIS'][1], linewidth=0.5, color='green')
        ax[1].plot(post_mol_data['VIS'][0], VIS_spec, linewidth=0.5, color='green')
        ax[0].set_ylabel('e/s')
        ax[1].set_ylabel('ergs')
        ax[1].set_xlabel('Wavelength')
        fig.suptitle('VIS')
        plt.show()

        fig, ax = plt.subplots(2, 1, sharex=True)
        plt.subplots_adjust(hspace=0.1)
        ax[0].plot(post_mol_data['NIR'][0], post_mol_data['NIR'][1], linewidth=0.5, color='red')
        ax[0].set_ylim(bottom=0.0, top=1.e6)
        ax[1].plot(post_mol_data['NIR'][0], NIR_spec, linewidth=0.5, color='red')
        ax[1].set_ylim(bottom=0.0, top=5.e-15)
        ax[0].set_ylabel('e/s')
        ax[1].set_ylabel('ergs')
        ax[1].set_xlabel('Wavelength')
        fig.suptitle('NIR')
        plt.show()

        fig, ax = plt.subplots(1, 1)
        ax.plot(post_mol_data['UVB'][0], UVB_spec, linewidth=0.5, color='blue')
        ax.plot(post_mol_data['VIS'][0], VIS_spec, linewidth=0.5, color='green')
        ax.plot(post_mol_data['NIR'][0], NIR_spec, linewidth=0.5, color='red')
        ax.set_ylim(bottom=0.0, top=5.e-13)
        ax.set_ylabel('ergs')
        ax.set_xlabel('Wavelength')
        fig.suptitle('Full')
        plt.show()

        fig, ax = plt.subplots(1, 1)
        ax.plot(post_mol_data['UVB'][0], UVB_spec, linewidth=0.5, color='blue')
        ax.plot(post_mol_data['VIS'][0], VIS_spec, linewidth=0.5, color='green')
        ax.plot(post_mol_data['NIR'][0], NIR_spec, linewidth=0.5, color='red')
        ax.set_xlim(0.54, 0.57)
        ax.set_ylabel('ergs')
        ax.set_xlabel('Wavelength')
        fig.suptitle('UVB-VIS')
        plt.show()

        fig, ax = plt.subplots(1, 1)
        ax.plot(post_mol_data['UVB'][0], UVB_spec, linewidth=0.5, color='blue')
        ax.plot(post_mol_data['VIS'][0], VIS_spec, linewidth=0.5, color='green')
        ax.plot(post_mol_data['NIR'][0], NIR_spec, linewidth=0.5, color='red')
        ax.set_xlim(1.0, 1.1)
        ax.set_ylabel('ergs')
        ax.set_xlabel('Wavelength')
        fig.suptitle('VIS-NIR')
        plt.show()

        res_curv_data = {'UVB': [post_mol_data['UVB'][0], UVB_spec], 'VIS': [post_mol_data['VIS'][0], VIS_spec], 'NIR': [post_mol_data['NIR'][0], NIR_spec]}
        #res_curv_wave, res_curv_spec = self.edge_matching(res_curv_data, normalize_only=False, bonus_plots=False, annotate=False, threshold=0.3, dichroic=False)
        res_curv_wave = []
        res_curv_wave.append(post_mol_data['UVB'][0])
        res_curv_wave.append(post_mol_data['VIS'][0])
        res_curv_wave.append(post_mol_data['NIR'][0])
        res_curv_wave = np.concatenate(res_curv_wave)
        res_curv_spec = []
        res_curv_spec.append(UVB_spec)
        res_curv_spec.append(VIS_spec)
        res_curv_spec.append(NIR_spec)
        res_curv_spec = np.concatenate(res_curv_spec)

        mask = np.isfinite(res_curv_spec)
        res_curv_spec = res_curv_spec[mask]
        res_curv_wave = res_curv_wave[mask]

        w, h = plt.figaspect(0.5)
        fig, ax = plt.subplots(1, 1, figsize=(w, h))
        plt.subplots_adjust(hspace=0.4)
        ax.plot(raw_wave, raw_spec, color='red', alpha=0.5, linewidth=0.5, label='Raw')
        ax.plot(di_wave, di_spec, color='green', alpha=0.5, linewidth=0.5, label='Integrated')
        ax.plot(post_mol_wave, post_mol_spec, color='blue', alpha=0.5, linewidth=0.5, label='Post-Molecfit')
        ax.plot(res_curv_wave, res_curv_spec, color='rebeccapurple', alpha=0.5, linewidth=0.5, label='Response Curve Applied')
        #ax.set_ylim(top=1.25*np.nanmax(post_mol_spec))
        #ax.set_ylim(bottom=-1e4, top=4e15)
        #ax.set_xlim(0.4,0.8)
        ax.set_ylim(top=2.e3, bottom=0.)
        fig.suptitle('Raw, Integrated, Molecfitted, Response Curve')
        plt.legend()
        plt.show()

        w, h = plt.figaspect(0.5)
        fig, ax = plt.subplots(1, 1, figsize=(w, h))
        plt.subplots_adjust(hspace=0.4)
        ax.plot(res_curv_wave, res_curv_spec, color='rebeccapurple', alpha=0.5, linewidth=0.5, label='Response Curve Applied')
        fig.suptitle('Final Corrected')
        plt.legend()
        plt.show()

        # Define spectral regions
        corrected_regions = [
            [0.58, 0.60], [0.64, 0.66], [0.68, 0.75], [0.78, 0.86], [0.88, 1.0],
            [1.062, 1.244], [1.26, 1.57], [1.63, 2.48]
        ]
        untrustworthy_regions = [
            [0.50, 0.51], [0.758, 0.77], [0.685, 0.695], [0.625, 0.632], [0.72, 0.73],
            [0.81, 0.83], [0.89, 0.98], [1.107, 1.164], [1.3, 1.5], [1.73, 2.0],
            [2.38, 2.48], [1.946, 1.978], [1.997, 2.032], [2.043, 2.080]
        ]

        def plot_region(ax, wave, data, regions, color, linewidth=0.3):
            """Plot spectral regions with given color."""
            for low, high in regions:
                lidx = (np.abs(wave - low)).argmin()
                hidx = (np.abs(wave - high)).argmin()
                ax.plot(wave[lidx:hidx], data[lidx:hidx], linewidth=linewidth, color=color)

        def plot_data_and_regions(ax, wave, data, title):
            """Plot spectrum with highlighted regions."""
            ax.plot(wave, data, color='black', linewidth=0.3)

            plot_region(ax, wave, data, corrected_regions, color='#121212')  # dark highlight
            plot_region(ax, wave, data, untrustworthy_regions, color='lightgrey')

            ax.set_title(title)
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.3f}'))
            ax.xaxis.set_minor_formatter(StrMethodFormatter('{x:.3f}'))
            # ax.set_ylim(0, 1.5)  # optional

        if bonus_plots:
            w, h = plt.figaspect(1)
            fig11, axes = plt.subplots(3, 1, figsize=(w, h))
            plt.subplots_adjust(hspace=0.4)

            # Disk Integrated
            plot_data_and_regions(axes[0], di_wave, di_spec, 'Disk Integrated')
            plot_data_and_regions(axes[1], post_mol_wave, post_mol_spec, 'Post Molecfit')

            axes[2].set_title("Post Molecfit + Telluric Solution")

        #ranges = [
        #     (UVBwave, UVB_spec, 'Telluric Solution Applied'),
        #    (VISwave, VIS_spec, 'Telluric Solution Applied'),
        #    (NIRwave, NIR_spec, 'Telluric Solution Applied')
        #]
        #if bonus_plots == True:
            #for wave, data, title in ranges:
            #    plot_data_and_regions(ax11, wave, data, title)
            #ax11.set_yscale('log')
            #ax11.set_ylim(10e-4, 10e0)
        #for wave, data, title in ranges:
        #    plot_data_and_regions(ax4, wave, data, title)
        #ax4.set_yscale('log')
        #ax4.set_ylim(10e-4, 10e0)

        # Add in a black body
        #bb = BlackBody(temperature=5778 * un.K)
        #bb_flux = bb(bb_wave)*10e3
        #ax4.plot(bb_wave, bb_flux, label='Blackbody')
        #sp = SourceSpectrum(BlackBodyNorm1D, temperature=5778)
        #sp_flux = sp(wavelength * un.um)
        #norm_loc = self.closest_index(post_mol_wave, norm_loc)
        #header_info = {'NORM_IDX': norm_loc}
        #sp_flux = sp_flux / sp_flux[norm_loc]
        #ax4.plot(wavelength, sp_flux, label='Synphot Blackbody')

        # Add in Filters
        #filters = {'johnson_u':'U_PHOT', 'johnson_b':'B_PHOT', 'johnson_v':'V_PHOT',
        #           'johnson_r':'R_PHOT', 'johnson_i':'I_PHOT', 'johnson_j':'J_PHOT',
        #           'johnson_k':'K_PHOT' }
        # Create a spectrum object with unit info
        #spectrum = [UVB_spec[0], VIS_spec[0], NIR_spec[0]]
        #spectrum = np.concatenate(spectrum)
        #wave_ang = wavelength * 10000 * u.angstrom
        #spectrum = spectrum * units.FLAM
        #spectrum_obj = SourceSpectrum(Empirical1D, points=wave_ang, lookup_table=spectrum)

        #for f in filters.keys():
            # Load a filter
        #    band = SpectralElement.from_filter(f)
            # Determine the color
        #    observation = Observation(spectrum_obj, band, force='taper')
        #    color = observation.effstim(flux_unit='flam')
        #    location = observation.effective_wavelength()
            # Add it to the plot
            #ax4.plot(location/10000, color, 'o', color='orangered')
            #if bonus_plots == True:
            #    ax11.plot(location / 10000, color, 'o', color='orangered')
            # Add it to the FITS header
        #    header_info[filters[f]] = color.value

        #ax5 = fig1.add_subplot(212)
        #plot_data_and_regions(ax5, wavelength, spectrum/sp_flux, 'Albedo')
        #ax5.set_yscale('log')
        #ax5.set_ylim(10e-4, 10e0)

        # Add a figure title
        #if bonus_plots == True:
        #    current_date = datetime.now().strftime('%Y-%m-%d')
            #fig11.suptitle(f'{self.body.title()}\nGenerated on: {current_date}, '
            #          f'Added in filter colors and Albedo')

        # Prep for FITS file
        #flux = np.array(spectrum / sp_flux)
        #errors= [EUVB[0], EVIS[0], ENIR[0]]
        #errors = np.concatenate(errors)
        dont_trust = [[0.3,0.34], [0.50, 0.51], [0.758, 0.77], [0.685, 0.695], [0.625, 0.632],
                      [0.72, 0.73], [0.81, 0.83],[0.89, 0.98],
                      [1.107, 1.164], [1.3, 1.5], [1.73, 2.0], [2.38, 2.48], [1.946, 1.978],
                      [1.997, 2.032], [2.043, 2.080]]

        corrected = [[0.58, 0.60], [0.64, 0.66], [0.68, 0.75], [0.78, 0.86], [0.88, 1.0],
                     [1.062, 1.244], [1.26, 1.57], [1.63, 2.48]]
        dont_trust_mask = np.zeros_like(res_curv_wave, dtype=bool)
        corrected_mask = np.zeros_like(res_curv_wave, dtype=bool)
        for d in dont_trust:
            dont_trust_mask |= ma.masked_inside(res_curv_wave, d[0], d[1]).mask
        for c in corrected:
            corrected_mask |= ma.masked_inside(res_curv_wave, c[0], c[1]).mask
        header_info = {'BB_TEMP': 5778}
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
        hdu[0].data = res_curv_wave
        hdu[1].data = res_curv_spec
        hdu[2].data = np.zeros_like(post_mol_wave)
        hdu[3].data = np.concatenate((self.UVB_solution, self.VIS_solution, self.NIR_solution))
        hdu[4].data = dont_trust_mask.astype(int)
        hdu[5].data = corrected_mask.astype(int)

        # Set header info
        for label in header_info.keys():
            hdu[0].header[label] = header_info[label]
        hdu[0].header['FLUX_STD'] = self.tell_star

        # Write to disk, overwriting if file exists
        hdu.writeto(albedo_file, overwrite=True)
        print(f'File Saved to: ' + self.work_dir + f'MOV_{self.body.title()}_SCI_IFU_ALBEDO.fits')

        # Main figure
        #w, h = plt.figaspect(1)
        #fig1, axes1 = plt.subplots(2, 1, figsize=(w, h))
        #plt.subplots_adjust(hspace=0.4)
        #plot_data_and_regions(axes1[0], self.diwave, self.dispec, 'Spacialy Integrated')
        #plot_data_and_regions(axes1[1], wav, spectrum.value, 'Fully Corrected')
        #axes1[1].set_ylim(bottom=-1.0, top=np.nanpercentile(np.abs(spectrum.value), 98))
        #plt.show()

        #investigate nir 1.1 - 1.25 and snr difference between nir and vis/uv

if __name__ == "__main__":

    object_list = ['uranus']

    for o in object_list:
        obj = DataAnalysis(body=o)
        obj.apply_response_curve(num_bins=40, bonus_plots=True, tell_cor_plots=True, edge_plots=False)
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
