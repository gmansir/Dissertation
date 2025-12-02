"""
Module Name: Albedo
Module Author: Giovannina Mansir (nina.mansir@gmail.com)
Module Version: 0.0.1
Last Modified: 2023-12-13

Description:
This class takes a compressed and reduced FITS file of a planetary spectrum and contains
methods to analyze it and highlight information

Usage:
import matplotlib.pyplot as plt
import Alebedo
analysis = Albedo.DataAnalysis(body='')
Albedo.plot_albedo()

Dependancies:
-numpy
-matplotlib
-astropy
-copy
-glob
"""

import re
import numpy as np
from astropy import units as un
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from datetime import datetime
from PyAstronomy import pyasl
import astropy.constants as consts
from astropy import units as u
from astropy.table import Table
from synphot import SourceSpectrum, units
from synphot import SpectralElement
from synphot import Observation
from synphot.models import Empirical1D, BlackBodyNorm1D
from brokenaxes import brokenaxes
import os

#matplotlib.use('TkAgg')
#matplotlib.use('Agg')

class AlbedoAnalysis:
    def __init__(self, body):
        self.body = body.lower()

        paths = {'neptune': r"C:\Users\ninam\Documents\Chile_Stuff\Dissertation\Spec_Files\Neptune\Post_Molecfit\MOV_Neptune_SCI_IFU_ALBEDO.fits",
                 'saturn': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Saturn\\Post_Molecfit\\MOV_Saturn_SCI_IFU_ALBEDO.fits",
                 'uranus': "C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\Uranus\\Post_Molecfit\\MOV_Uranus_SCI_IFU_ALBEDO.fits",
                 'titan': r"C:\Users\ninam\Documents\Chile_Stuff\Dissertation\Spec_Files\Titan\Post_Molecfit\MOV_Titan_SCI_IFU_ALBEDO.fits",
                 }

        self.work_dir = os.path.dirname(paths[self.body.lower()])
        self.albedo_file = paths[self.body.lower()]
        self.load_data()
        self.find_albedo()

    def load_data(self):

        try:
            hdu = fits.open(self.albedo_file, mode='readonly')

            self.wav = hdu[0].data
            self.flux = hdu['FLUX'].data
            self.errors = hdu['ERRS'].data
            self.resp_crv = hdu['RESP_CRV'].data
            self.mask_tel = np.invert([bool(x) for x in hdu['MASK_TEL'].data])
            self.mask_wrn = np.invert([bool(x) for x in hdu['MASK_WRN'].data])
            self.bb_temp = hdu[0].header.get('BB_TEMP')
            self.flux_std = hdu[0].header.get('FLUX_STD')
            #self.norm_idx = hdu[0].header.get('NORM_IDX')
            self.clean_name = re.sub(r'\d+', '', self.body)

            print(f"Data loaded for: {self.body}")

        except FileNotFoundError:
            print(f"File not found: {self.albedo_file}")
        finally:
            # Close the file
            if 'hdu' in locals() and hdu is not None:
                hdu.close()

        mask = np.isfinite(self.flux)
        self.wav = self.wav[mask]
        self.flux = self.flux[mask]
        self.current_wav = self.wav
        self.current_flux = self.flux

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

    def find_albedo(self):

        #sp = SourceSpectrum(BlackBodyNorm1D, temperature=self.bb_temp)
        sort_idx = np.argsort(self.wav)
        self.wav = self.wav[sort_idx]
        self.flux = self.flux[sort_idx]
        self.albedo = self.flux
        #bb_flux = sp(self.wav * un.um)
        #self.bb_flux = np.array(bb_flux) / np.max(np.array(bb_flux))
        #self.bb_flux *= np.nanpercentile(self.flux, 99)

        #self.albedo = self.flux / self.bb_flux
        #self.albedo = self.albedo[0]
        self.current_flux = self.albedo

        # Test Plot
        w, h = plt.figaspect(0.5)
        fig, ax = plt.subplots(1, 1, figsize=(w, h))
        plt.subplots_adjust(hspace=0.4)
        ax.plot(self.wav, self.flux, color='b', alpha=0.5, linewidth=0.5, label='Post-Molecfit')
        ax.plot(self.wav, self.albedo, color='cornflowerblue', alpha=0.5, label='Albedo')
        ax.grid(True)
        #ax.set_ylim(top=800)
        fig.suptitle('Starting Spec (Post Molecfit)')
        plt.show()

    def plot_data_and_regions(self, ax, wave, data):
        ax.plot(wave, data, color='black', linewidth=0.3)
        wave_wrn = np.ma.masked_array(wave, mask=self.mask_wrn)
        data_wrn = np.ma.masked_array(data, mask=self.mask_wrn)
        plt.plot(wave_wrn, data_wrn, linewidth=0.3, color='#121212')
        wave_tel = np.ma.masked_array(wave, mask=self.mask_tel)
        data_tel = np.ma.masked_array(data, mask=self.mask_tel)
        plt.plot(wave_tel, data_tel, linewidth=0.3, color='lightgray')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
        ax.xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))

    def load_solar_specs(self):

        # VIS File path
        self.sun_spec_vis = f'C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\NARVAL_Sun.txt'
        self.sun_spec_lisird = f'C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\LISIRD.txt'

        # Load in the data from the file
        self.lisird_data = np.loadtxt(self.sun_spec_lisird, delimiter=',')
        self.vis_data = np.loadtxt(self.sun_spec_vis)

        # Assuming the first column is wavelength, second is flux, and third is errors
        self.lisird_wave = self.lisird_data[:, 0] / 1000
        self.lisird_flux = self.lisird_data[:, 1]

        flux_percentile = np.percentile(self.flux, 99)
        lisird_percentile = np.percentile(self.lisird_flux, 99)
        scale_factor = flux_percentile / lisird_percentile

        self.lisird_flux *=  scale_factor
        self.sun_wave_vis = self.vis_data[:, 0] / 1000
        self.sun_flux_vis = self.vis_data[:, 1]

        # NIR File path
        self.sun_spec_ir = f'C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\SOL_merged.fits'

        #Load in the data from the file
        with fits.open(self.sun_spec_ir) as hdul:
            self.sun_wave_ir = hdul[0].data / 1000
            self.sun_flux_ir = hdul[1].data

        self.solar_absorptions = [0.42280, 0.43092, 0.58915, 0.64025, 0.656475, 0.8501,
                                  0.85440, 0.86641, 1.0832, 1.1285, 1.2824, 1.5650, 1.6152, 1.9522, 2.0582,
                                  1.203485, 1.227412, 1.140696, 1.09180, 1.08727, 1.07899,
                             1.01487, 1.502919, 1.589281, 1.67553, 1.67236,
                             1.711338, 1.945829, 1.951114, 1.972798, 2.116965, 1.97823, 1.98675, 1.05880]

        # 0.39345,0.39695,0.98920
        self.solar_absorptions.sort()
        self.data_absorptions = [0.42264, 0.43076, 0.58896, 0.6400, 0.65626, 0.84980,
                                 0.85421, 0.86621, 1.0830, 1.1283, 1.2822, 1.5648, 1.615, 1.952, 2.058,
                                 1.203104, 1.226985, 1.140339, 1.09144, 1.08689, 1.07864,
                             1.01461, 1.502442, 1.588782, 1.67501, 1.67189,
                            1.710817, 1.945225, 1.950529, 1.9721615, 2.116318, 1.9775, 1.98615, 1.05845]

        #0.39336,0.39682,0.98903,
        self.data_absorptions.sort()

        self.best_lines = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33] # neptune
        #self.best_lines = [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33] # titan
        #self.best_lines = [0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,32,33] # saturn
        self.best_solars = [self.solar_absorptions[i] for i in self.best_lines]
        self.best_datas = [self.data_absorptions[i] for i in self.best_lines]

        # Plot the results
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)
        ax1.plot(self.wav, self.albedo, linewidth=0.3, label=self.clean_name.title(), color='black')
        ax1.plot(self.lisird_wave, self.lisird_flux, linewidth=0.3, label='LISIRD', color='teal')
        ax1.legend()
        ax1.set_xlabel('Wavelength (µm)')
        ax1.set_ylabel('Flux')
        #ax1.set_ylim(bottom=0.0, top=np.max(self.lisird_flux)*1.1)
        #ax1.set_ylim(bottom=0.0, top=700)
        #ax1.set_xlim(left=0.38, right=0.39)
        ax1.minorticks_on()
        plt.savefig(f'C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\solar_spec.png')
        fig1.show()

    def rv_correction(self):

        def find_minimum(wavelengths, flux_data, target_wavelength, search_range):
            differences = np.abs(wavelengths - target_wavelength)
            indices_within_range = np.where(differences <= search_range)
            min_index = np.argmin(flux_data[indices_within_range])
            min_wavelength = wavelengths[indices_within_range][min_index]
            return min_wavelength

        def gaussian(x, amp, mean, stddev, baseline):
            return -amp * np.exp(-((x-mean)/(2*stddev))**2) + baseline

        lisird_shifts = []
        new_data_mins = []
        new_lisird_mins = []
        fig2 = plt.figure(2)
        fig2.tight_layout()
        subplot=1
        index=1
        counter = 0
        for solar_target, data_target in zip(self.best_solars, self.best_datas):
            print(f'Line Number {self.best_lines[counter]}')
            counter += 1

            # Find the central wavelength of the solar absorption
            lisird_min = find_minimum(self.lisird_wave, self.lisird_flux, solar_target, 0.0001)
            print(f'Suggested solar: {solar_target}')
            print(f'Lisird min: {lisird_min}')

            # Find the central wavelength of the data absorption by finding the minimum (back-up technique)
            data_min = find_minimum(self.wav, self.albedo, data_target, 0.001)
            print(f'Suggested Data: {data_target}')
            print(f'Data min: {data_min}')

            # Clip out a range of wavelength values to focus on
            wavelength_range = np.abs(self.wav - data_target) <= 0.01
            x_data = self.wav[wavelength_range]
            y_data = self.albedo[wavelength_range]

            # Search for large absorptions within the clipped region
            peaks, _ = find_peaks((-y_data)) #, prominence=0.0001)

            # Find the peak closest to the target wavelength (assumes I was fairly accurate)
            closest_peak_index = peaks[np.argmin(np.abs(x_data[peaks] - data_target))]

            # Clip a smaller region to really focus on this one peak
            peak_range = np.arange(max(0, closest_peak_index - 8), min(len(x_data), closest_peak_index + 8))

            # Make educated guesses about the amplitude and sigma for initialization
            amp_guess = max(y_data[peak_range]) - min(y_data[peak_range])
            sig_guess = np.abs(x_data[peak_range[0]] - x_data[peak_range[-1]]) / 4.
            initial_guess = [amp_guess, data_min, sig_guess, max(y_data[peak_range])]
            try:
                # Fit gaussians to hone in on the true amp, mean, and sigma, return the best mean
                params = curve_fit(gaussian, x_data[peak_range], y_data[peak_range], p0=initial_guess)
                gauss_min = params[0][1]
                # Plot a few to check result
                if subplot in [1, 2, 3, 4, 5, 6, 7]:
                    if subplot == 1:
                        labels = ['LISIRD', self.clean_name.title(), 'Solar Feature', 'Gaussian Fit', 'LISIRD Feature', 'By Eye', 'Shifted']
                    else:
                        labels = [None, None, None, None, None, None, None]
                    ax2 = fig2.add_subplot(7, 1, index)
                    #low = self.closest_index(self.sun_wave_ir, x_data[0])
                    #high = self.closest_index(self.sun_wave_ir, x_data[-1])
                    l_low = self.closest_index(self.lisird_wave, x_data[0])
                    l_high = self.closest_index(self.lisird_wave, x_data[-1])
                    scale_factor = np.mean(y_data) / np.mean(self.lisird_flux[l_low:l_high])
                    ax2.plot(self.lisird_wave[l_low:l_high], self.lisird_flux[l_low:l_high] * scale_factor, color='teal', label=labels[0])
                    ax2.plot(x_data, y_data, color='black', label=labels[1])
                    #ax2.plot(x_data[peak_range], y_data[peak_range], color='blue')
                    #ax2.plot(x_data[peak_range], gaussian(x_data[peak_range], params[0][0], params[0][1], params[0][2],
                    #                                      params[0][3]), color='deepskyblue')
                    #ax2.axvline(solar_min, linestyle='--', color='orange', label=labels[2])
                    ax2.axvline(gauss_min, linestyle='--', color='black', label=labels[3])
                    ax2.axvline(lisird_min, linestyle='--', color='teal', label=labels[4])
                    #ax2.axvline(data_min, linestyle='--', color='violet', label=labels[5])
                    #ax2.axvline((gauss_min-3.939984916106145e-05)/0.9995970818148777, linestyle='--',
                    #            color='limegreen', label=labels[6])
                    ax2.set_xlim(lisird_min-0.001, lisird_min+0.001)
                    ax2.set_xlabel('Wavelength (µm)')
                    ax2.set_ylabel('Flux')
                    fig2.legend()
                    index += 1
                subplot += 1
                # Save values
                #new_solar_mins.append(solar_min)
                new_data_mins.append(gauss_min)
                new_lisird_mins.append(lisird_min)
                #shifts.append(solar_min-gauss_min)
                lisird_shifts.append(lisird_min-gauss_min)
            except RuntimeError:
                continue
        plt.savefig(f'C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\solar_feature_fit.png')
        fig2.show()

        print(f'Targets: {self.best_solars}')
       # print(f'Shifts: {shifts}')
        print(f'Lisird Shifts: {lisird_shifts}')

        #linregress = np.polyfit(new_solar_mins, new_data_mins, 1)
        lisird_linregress = np.polyfit(new_lisird_mins, new_data_mins, 1)

        #regress_line = np.polyval(linregress, np.array(new_solar_mins))
        lisird_regress_line = np.polyval(lisird_linregress, np.array(new_lisird_mins))

        new_data_mins = np.array(new_data_mins)
        #shifted_data_mins = (new_data_mins-linregress[1])/linregress[0]
        lisird_shifted_data_mins = (new_data_mins-lisird_linregress[1])/lisird_linregress[0]

        #rvs = 2.99792458e8 * (new_data_mins-shifted_data_mins)/shifted_data_mins
        lisird_rvs = 2.99792458e8 * np.abs(new_data_mins-lisird_shifted_data_mins)/lisird_shifted_data_mins
        print(f'RVS: {lisird_rvs}')

        fig3 = plt.figure(3)
        gs = fig3.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0)
        ax3_main = fig3.add_subplot(gs[0])
        #ax3_main.plot(new_solar_mins, new_data_mins, 'o', label='Data')
        ax3_main.plot(new_lisird_mins, new_data_mins, 'o', label='LISIRD', color='teal')
        #ax3_main.plot(new_solar_mins, regress_line, label='Linear Regression')
        ax3_main.plot(new_lisird_mins, lisird_regress_line, label='LISIRD Regression', color='cornflowerblue')
        #ax3_main.set_title(f'Linear Regression: {(1 - lisird_linregress[0]):.6f}, Mean: {np.mean(lisird_shifts):.6f},\nRV: {np.mean(lisird_rvs)/1000} km/s')
        print(f'Linear Regression: {(1 - lisird_linregress[0]):.4f}, Mean: {np.mean(lisird_shifts):.4f},\nRV: {np.mean(lisird_rvs)/1000} km/s')
        ax3_main.set_xlabel('Solar Features (µm)')
        ax3_main.set_ylabel('Planet Features (µm)')
        ax3_residual = fig3.add_subplot(gs[1], sharex=ax3_main)
        #ax3_residual.plot(new_solar_mins, regress_line - new_data_mins, 'o')
        ax3_residual.plot(new_lisird_mins, lisird_regress_line-new_data_mins, 'o', color='teal')
        ax3_residual.axhline(0, color='black', linestyle='--', linewidth=2)
        ax3_residual.set_xlabel('Solar Features (µm)')
        ax3_residual.set_ylabel(r"$\Delta$")
        fig3.tight_layout()
        plt.savefig(f'C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\solar_linear_regression.png')
        fig3.show()

        #self.rv_corr = (self.wav-linregress[1])/linregress[0]
        self.lisird_rv_corr = (self.wav-lisird_linregress[1])/lisird_linregress[0]

    def gaussian_convolution(self):

        #self.sun_lim = self.closest_index(self.sun_wave_ir, self.rv_corr[-1]) + 2
        #self.wav_lim = self.closest_index(self.rv_corr, self.sun_wave_ir[0]) + 1

        self.lisird_low_lim = self.closest_index(self.lisird_wave, self.lisird_rv_corr[0]) - 1
        self.lisird_high_lim = self.closest_index(self.lisird_wave, self.lisird_rv_corr[-1]) + 2

        # Compute Gaussian Convolution and fit for best kernel and sigma
        # Use just the features I picked out and mask with a 4* feature width
        # Form a few sub-groups and compare
        # Wavelength dependance on width? (XSHOOTER Manual?)
        def gaussian_kernel(size, sigma):
            """
            size : positive integer number of samples (will be coerced to odd integer >=3)
            sigma: positive float
            returns normalized kernel (sum == 1)
            """
            # validate inputs
            if not np.isfinite(size) or not np.isfinite(sigma):
                return None

            # coerce size to a sensible integer (odd preferred)
            size_i = int(max(3, round(float(size))))
            if size_i % 2 == 0:
                size_i += 1

            sigma_f = float(sigma)
            if sigma_f <= 0 or np.isnan(sigma_f):
                return None

            # create grid centered at zero
            half = size_i // 2
            x = np.linspace(-half, half, size_i)
            kernel = np.exp(-0.5 * (x / sigma_f) ** 2)
            s = np.sum(kernel)
            if s <= 0 or not np.isfinite(s):
                return None
            return kernel / s

        def fitness_function(guesses, data_wav, data_flux, solar_wav, solar_flux):
            """
            guesses: [kernel_size (float), sigma (float)]
            Return a scalar cost (MSE). On invalid inputs return a large penalty.
            """
            kernel_size, sigma = guesses

            # quick validation
            if (not np.isfinite(kernel_size)) or (not np.isfinite(sigma)):
                return 1e6

            # coerce and guard
            kernel = gaussian_kernel(kernel_size, sigma)
            if kernel is None:
                return 1e6

            # ensure there is data to operate on
            if len(solar_flux) < 3 or len(data_flux) < 3:
                return 1e6

            # convolve and interpolate
            try:
                smoothed_solar_spec = convolve1d(solar_flux, kernel, mode='constant', cval=0.0)
                interp_func = interp1d(solar_wav, smoothed_solar_spec, kind='slinear', bounds_error=False,
                                       fill_value=np.nan)
                interp_smooth = interp_func(data_wav)
                # if interpolation fails or produces NaN, penalize
                if not np.all(np.isfinite(interp_smooth)):
                    return 1e6
                mse = np.mean((interp_smooth - data_flux) ** 2)
                # if mse is NaN for any reason, penalize
                if not np.isfinite(mse):
                    return 1e6
                return mse
            except Exception:
                # any unexpected crash -> big penalty
                return 1e6

        # prepare bounds: kernel_size between 3 and, say, 401; sigma between 0.1 and 100
        bounds = [(3, 401), (0.1, 200.0)]

        initial_guess = [50.0, 10.0]

        lisird_results = []
        for b in self.best_solars:
            print(f'Working on line: {b}')
            data_idx = np.abs(self.lisird_rv_corr - b) <= 0.04
            data_wav = self.lisird_rv_corr[data_idx]
            data_flux = self.flux[data_idx]

            lisird_idx = np.abs(self.lisird_wave - b) <= 0.04
            lisird_wav = self.lisird_wave[lisird_idx]
            lisird_flux = self.lisird_flux[lisird_idx]

            # check there is enough data to fit
            if data_wav.size < 5 or lisird_wav.size < 5:
                print("Skipping line due to insufficient data points.")
                continue

            min_lisird_wav = np.min(lisird_wav)
            max_lisird_wav = np.max(lisird_wav)

            lis_low = self.closest_index(data_wav, min_lisird_wav) + 1
            lis_high = self.closest_index(data_wav, max_lisird_wav) - 1

            l_data_wav = data_wav[lis_low:lis_high]
            l_data_flux = data_flux[lis_low:lis_high]

            # again check the sliced arrays are not empty
            if l_data_wav.size < 5 or lisird_wav.size < 5:
                print("Skipping line after slicing: insufficient overlap.")
                continue

            res = minimize(fitness_function, initial_guess, args=(l_data_wav, l_data_flux, lisird_wav, lisird_flux),
                           method='TNC', bounds=bounds, options={'maxfun': 1000})
            if not res.success:
                print("Optimizer warning/failure:", res.message)
            lisird_results.append(res.x)

        # convert to array safely
        if lisird_results:
            lisird_results = np.array(lisird_results)
            lisird_size = np.mean(lisird_results[:, 0])
            lisird_sigma = np.median(lisird_results[:, 1])
        else:
            lisird_size = np.nan
            lisird_sigma = np.nan
            print("No successful lisird fits were found.")

        # Use these optimal values to smooth the solar spectrum
        lisird_optimal_gaussian = gaussian_kernel(lisird_size, lisird_sigma)
        self.smoothed_lisird_spectrum = convolve1d(self.lisird_flux, lisird_optimal_gaussian, mode='constant')

        fig4 = plt.figure(4)
        ax4 = fig4.add_subplot(211)
        ax4.plot(self.lisird_rv_corr, self.albedo, linewidth=0.3, label=self.clean_name.title(), color='k')
        ax4.plot(self.lisird_wave[self.lisird_low_lim:self.lisird_high_lim], self.lisird_flux[self.lisird_low_lim:self.lisird_high_lim], color='teal', linewidth=0.3,
                 label='Original')
        ax4.plot(self.lisird_wave[self.lisird_low_lim:self.lisird_high_lim], self.smoothed_lisird_spectrum[self.lisird_low_lim:self.lisird_high_lim], linewidth=0.3,
                 color='cornflowerblue', label='Conlvolved')
        ax4.set_ylim(bottom=0.0, top=np.nanmax(self.lisird_flux[self.lisird_low_lim:self.lisird_high_lim])*1.1)
        ax5 = fig4.add_subplot(212)
        rv_low = self.closest_index(self.lisird_rv_corr, 0.48)
        rv_high = self.closest_index(self.lisird_rv_corr, 0.52)
        lis_low = self.closest_index(self.lisird_wave, 0.48)
        lis_high = self.closest_index(self.lisird_wave, 0.52)
        scale_factor_orig = np.mean(self.lisird_flux[lis_low:lis_high]) - np.mean(self.flux[rv_low:rv_high])
        scale_factor_smooth = np.mean(self.smoothed_lisird_spectrum[lis_low:lis_high]) - np.mean(self.flux[rv_low:rv_high])
        ax5.plot(self.lisird_wave[self.lisird_low_lim:self.lisird_high_lim],
                 self.lisird_flux[self.lisird_low_lim:self.lisird_high_lim] - scale_factor_orig, color='teal', linewidth=0.5)
        ax5.plot(self.lisird_wave[self.lisird_low_lim:self.lisird_high_lim],
                 self.smoothed_lisird_spectrum[self.lisird_low_lim:self.lisird_high_lim] - scale_factor_smooth,
                 linewidth=0.8,
                 color='cornflowerblue')
        ax5.plot(self.lisird_rv_corr, self.flux, linewidth=0.5, color='k')
        ax5.set_xlim((0.57, 0.62))
        ax5.set_ylim(bottom=np.min(self.lisird_flux[self.lisird_low_lim:self.lisird_high_lim] - scale_factor_orig), top=np.max(self.lisird_flux[self.lisird_low_lim:self.lisird_high_lim] - scale_factor_orig)*1.1)
        ax5.set_xlabel('Wavelength (µm)')
        ax5.set_ylabel('Relative Flux')
        #fig4.suptitle('Convolution Results')
        ax4.legend()
        plt.savefig(f'C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Convolution_results.png')
        fig4.show()

    def interpolate(self):

        #interp_function = interp1d(self.sun_wave_ir[:self.sun_lim], self.smoothed_solar_spectrum[:self.sun_lim], kind='slinear')
        #self.interp_smoothed = interp_function(self.rv_corr[self.wav_lim:])

        # Adjust limits safely to ensure they are within bounds
        new_lim_low = self.closest_index(self.lisird_wave, self.lisird_rv_corr[0])
        new_lim_high = self.closest_index(self.lisird_wave, self.lisird_rv_corr[-1])

        # Check and correct the upper limit to prevent out-of-bounds indexing
        if new_lim_high > (len(self.lisird_wave) - 1):
            new_lim_high = min(new_lim_high, len(self.lisird_wave) - 1)
            self.current_wav = self.current_wav[:-2]
            self.current_flux = self.current_flux[:-2]

        # Define the interpolation function
        lisird_interp_function = interp1d(
            self.lisird_wave[new_lim_low:new_lim_high + 1],  # Include high limit in the range
            self.smoothed_lisird_spectrum[new_lim_low:new_lim_high + 1],
            kind='slinear',
            bounds_error=False,  # Avoid raising errors for out-of-bounds
            fill_value="extrapolate"  # Allow extrapolation for safety
        )

        # Ensure values in self.lisird_rv_corr are within the interpolation range
        valid_indices = (self.lisird_rv_corr >= self.lisird_wave[new_lim_low]) & (
                self.lisird_rv_corr <= self.lisird_wave[new_lim_high]
        )
        lisird_rv_corr_valid = self.lisird_rv_corr[valid_indices]

        # Apply the interpolation function to valid values only
        self.lisird_interp_smoothed = np.full_like(self.lisird_rv_corr, np.nan)  # Initialize with NaNs
        self.lisird_interp_smoothed[valid_indices] = lisird_interp_function(lisird_rv_corr_valid)


        #fig_number = 5
        #for b in self.best_solars:
            #w, h = plt.figaspect(1)
            #fig1 = plt.figure(fig_number, figsize=(w, h))
            #fig1.subplots_adjust(hspace=0.4)
            #ax1 = fig1.add_subplot(211)
            #self.plot_data_and_regions(ax1, self.rv_corr, self.flux)
            #ax1.set_xlim(left=0.98)
            #left = b - 0.002
            #right = b + 0.002
            #ax1.set_xlim(left, right)
            #ax1.set_ylim(top=1.3, bottom=0.0)
            #ax2 = fig1.add_subplot(212)
            #ax2.plot(self.rv_corr[self.wav_lim:], self.flux[self.wav_lim:]/self.interp_smoothed, linewidth=0.3, color='black')
            #ax2.set_xscale('log')
            #ax2.set_xlim(left, right)
            #ax2.set_ylim(top=0.015, bottom=0.0)
            #ax2.xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
            #ax2.xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
            #fig1.set_xlabel('Wavelength (microns)')
            #fig1.set_ylabel('Relative Flux')
            #fig1.suptitle(f'Albedo: {b}')
            #fig1.show()
            #fig_number += 1

        #w, h = plt.figaspect(1)
        #fig1 = plt.figure(fig_number, figsize=(w, h))
        #fig1.subplots_adjust(hspace=0.4)
        #ax1 = fig1.add_subplot(211)
        #self.plot_data_and_regions(ax1, self.rv_corr, self.flux)
        #ax1.set_xlim(left=0.98)
        #ax1.set_ylim(bottom=0.0, top=0.06)
        #ax2 = fig1.add_subplot(212)
        #ax2.plot(self.rv_corr[self.wav_lim:], self.flux[self.wav_lim:] / self.interp_smoothed, linewidth=0.3, color='black')
        #ax2.set_xscale('log')
        #ax2.xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
        #ax2.xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
        #ax2.set_xlabel('Wavelength (µm)')
        #ax2.set_ylabel('Relative Flux')
        #fig1.suptitle(f'Albedo')
        #fig1.show()

        self.current_wav = self.lisird_rv_corr[1:-1]
        self.current_flux = self.albedo[1:-1] / self.lisird_interp_smoothed[1:-1]

    def calculate_bond_albedo(self):
        """
        Calculate the Bond albedo A_B.

        Parameters:
        - wavelength_reflected: Wavelength array for reflected flux (microns)
        - flux_reflected: Observed flux array of the planet (W/m^2/μm)
        - wavelength_solar: Wavelength array for incident solar spectrum (microns)
        - flux_solar: Solar flux array at 1 AU (W/m^2/μm)
        - distance_sun_planet: Distance from the Sun to the planet in AU (scales solar flux)

        Returns:
        - Bond albedo A_B
        """
        min_len = min(len(self.current_flux), len(self.lisird_interp_smoothed), len(self.current_wav))

        flux = self.current_flux[:min_len]
        irradiance = self.lisird_interp_smoothed[:min_len]
        wavelengths = self.current_wav[:min_len]

        numerator = np.trapz(flux * irradiance, wavelengths)
        denominator = np.trapz(irradiance, wavelengths)
        bond_albedo = numerator / denominator

        print(f'Bond Albedo: {bond_albedo:.4f}')

    def save_fits(self):

        paper_ready_file = self.work_dir + f'\\MOV_{self.body.title()}_SCI_IFU_PAPER_READY.fits'
        # length check
        lengths = [
            len(self.current_wav),
            len(self.albedo[1:-1]),
            #len(self.interp_smoothed),
            len(self.lisird_interp_smoothed)
        ]
        min_length = min(lengths)
        self.current_wav = self.current_wav[:min_length]
        self.current_flux = self.albedo[1:min_length]
        #self.interp_smoothed = self.interp_smoothed[:min_length]
        self.lisird_interp_smoothed = self.lisird_interp_smoothed[:min_length]

        if os.path.exists(paper_ready_file):
            os.remove(paper_ready_file)

        primary_hdu = fits.PrimaryHDU(data=self.current_wav)
        flux_hdu = fits.ImageHDU(data=self.current_flux, name='FLUX')
        #sun_flux_hdu = fits.ImageHDU(data=self.interp_smoothed, name='SUN_FLUX')
        lisird_flux_hdu = fits.ImageHDU(self.lisird_interp_smoothed, name='LISIRD_FLUX')
        resp_crv_hdu = fits.ImageHDU(self.resp_crv, name='RESP_CRV')
        mask_tel_hdu = fits.ImageHDU(self.mask_tel.astype(int), name='MASK_TEL')
        mask_wrn_hdu = fits.ImageHDU(self.mask_wrn.astype(int), name='MASK_WRN')
        hdul = fits.HDUList([primary_hdu, flux_hdu, lisird_flux_hdu, resp_crv_hdu, mask_tel_hdu, mask_wrn_hdu])
        hdul.writeto(paper_ready_file, overwrite=True)

        print(f'File Saved to: ' + paper_ready_file)

    def yiyo_rv(self, skipedge=20, num_pix=1500):

        # File path
        sun_spec_ir = f'C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\SOL_merged.fits'

        #Load in the data from the file
        with fits.open(sun_spec_ir) as hdul:
            sun_wave_ir = hdul[0].data / 1000
            sun_flux_ir = hdul[1].data
        sun_lim = self.closest_index(sun_wave_ir, self.wav[-1]) + 2
        wav_lim = self.closest_index(self.wav, sun_wave_ir[0]) + 1
        sun_wave = sun_wave_ir[:sun_lim]
        sun_flux = sun_flux_ir[:sun_lim]
        wave = self.wav[wav_lim:]
        flux = self.flux[wav_lim:] + 1

        solar_absorptions = [1.203485, 1.227412, 1.140696, 1.09180, 1.08727, 1.07899,
                             0.98920, 1.01487, 1.502919, 1.589281, 1.67553, 1.67236,
                             1.711338, 1.945829, 1.951114, 1.972798, 2.116965]
        solar_absorptions.sort()
        data_absorptions = [1.203104, 1.226985, 1.140339, 1.09144, 1.08689, 1.07864,
                            0.98903, 1.01461, 1.502442, 1.588782, 1.67501, 1.67189,
                            1.710817, 1.945225, 1.950529, 1.9721615, 2.116318]
        data_absorptions.sort()

        best_lines = [1,3,6,7,9,10,11,13,14,15]
        best_solars = [solar_absorptions[i] for i in best_lines]
        best_datas = [data_absorptions[i] for i in best_lines]

        def compute_dRV(w, f, tw, tf, rvmin, rvmax, drv, skipedge=20, plot=False):
            # Function parameters:
            # w: observed wavelengths
            # f: observed flux
            # tw: template wavelengths
            # tf: template flux
            # rvmin, rvmax: minimum and maximum radial velocity values
            # drv: step size for radial velocity computation
            # skipedge: number of points to skip at each edge for cross-correlation
            # plot: flag for plotting intermediate results

            # Plotting the initial spectra and template (optional)
            if plot:
                plt.title('Template (blue) and spectra shifted (red), both normalized, before RV correction')
                plt.plot(tw, tf, 'b.-')
                plt.plot(w, f, 'r.-')
                plt.grid()
                plt.show()

            # Cross-correlation to compute radial velocity
            rv, cc = pyasl.crosscorrRV(w, f, tw, tf, rvmin, rvmax, drv, skipedge=skipedge)
            maxind = np.argmax(cc)

            # Display the result of cross-correlation
            print("Cross-correlation function is maximized at dRV = ", rv[maxind], " km/s")
            if rv[maxind] > 0:
                print(" A red-shift with respect to the template")
            else:
                print(" A blue-shift with respect to the template")

            # Plotting the cross-correlation function (optional)
            if plot:
                plt.plot(rv, cc, 'bp-')
                plt.plot(rv[maxind], cc[maxind], 'ro')
                plt.show()

                # Plotting the template and shifted spectra after RV correction (optional)
                plt.title('Template (blue) and spectra shifted (red), both normalized, after RV correction')
                plt.plot(tw, tf, 'b.-')
                plt.plot(w / (1 + rv[maxind] / consts.c.to(u.km / u.s).value), f, 'r.-')
                plt.grid()
                plt.show()

            # Return the determined radial velocity
            return rv[maxind]


        # Choose the radial velocity range and step size
        rvmin = -150
        rvmax = -111
        drv = 0.1

        # Optional: Choose parameters for cross-correlation (skipedge, plot)
        plot = False

        # Compute radial velocity shift
        sun_idx = [self.closest_index(sun_wave, s) for s in best_datas]
        sun2_idx = [self.closest_index(sun_wave, s) for s in best_solars]
        select = []
        for l,h in zip(sun_idx, sun2_idx):
            low = l-num_pix
            high = h+num_pix
            select.append((sun_wave[low], sun_wave[high]))
        sun_idx_masked = [np.where((sun_wave.data > s[0]) & (sun_wave.data < s[-1])) for s in select]
        data_idx_masked = [np.where((wave.data > s[0]) & (wave.data < s[-1])) for s in select]
        data_idx_masked = [d[0][3:-3] for d in data_idx_masked]
        rvs =[]
        for s, d in zip(sun_idx_masked, data_idx_masked):
            #print("Template Wavelength Range:", sun_wave[s].min(), sun_wave[s].max())
            #print("Data Wavelength Range:", wave[d].min(), wave[d].max())
            rv = compute_dRV(np.array(wave[d].tolist()), np.array(flux[d].tolist()), np.array(sun_wave[s].tolist()), np.array(sun_flux[s].tolist()), rvmin, rvmax,drv, skipedge=skipedge, plot=plot)
            rvs.append(rv)
        rv_shift = np.mean(rvs)

        # Apply RV correction to observed spectra
        #self.w_corr = wave / (1 - rv_shift / consts.c.to(u.km / u.s).value)
        self.w_corr = self.wav/((rv_shift*1000/2.99792458e8)+1)

        # Plot the template and shifted observed spectra after RV correction
        fig1 = plt.figure(6)
        ax1 = fig1.add_subplot(111)
        ax1.plot(sun_wave, sun_flux, color='deepskyblue', linewidth=0.3, label='Solar')
        ax1.plot(wave, flux, color='lightgray', linewidth=0.3, label='Before RV')
        ax1.plot(self.w_corr, self.flux, color='orangered', linewidth=0.3, label='After RV')
        plt.grid()
        plt.legend()
        plt.title(f'After RV correction: {rv_shift}')
        plt.show()

        # Now, w_corr contains the observed wavelengths corrected for the radial velocity shift

    def plot_albedo(self):
        """
        Plots the original cleaned spectrum and blackbody, as well as the
        division of the spectrum by the blackbody in Figure 1
        """
        w, h = plt.figaspect(1)
        fig100 = plt.figure(100, figsize=(w, h))
        fig100.subplots_adjust(hspace=0.4)
        ax100 = fig100.add_subplot(211)
        self.plot_data_and_regions(ax100, self.wav, self.flux)
        ax100.plot(self.wav, self.bb_flux, label='Synphot Blackbody')

        ax200 = fig100.add_subplot(212)
        self.plot_data_and_regions(ax200, self.wav, self.albedo)

        for f in self.filt_colors.keys():
            ax200.plot(self.filt_locs[f], self.filt_colors[f], 'o', color='orangered')

        ax200.set_yscale('log')
        ax200.set_ylim(10e-4, 10e0)
        current_date = datetime.now().strftime('%Y-%m-%d')
        fig100.suptitle(f'{self.body.title()}\nGenerated on: {current_date}, '
                       f'\nCreated new file for analysis')

    def plot_analysis(self, **kwargs):

        zoom_regions = {'titan': [(0.98, 1.14), (1.5, 1.7)]
                        }

        w, h = plt.figaspect(1)
        fig101 = plt.figure(101, figsize=(w, h))
        # Plotting with broken axes using GridSpec
        spec = fig101.GridSpec(2, 1, hspace=0.4)

        # Plotting the full spectrum in the top subplot
        ax101 = fig101.add_subplot(spec[0])
        self.plot_data_and_regions(ax101, self.wav, self.albedo)
        self.annotate_plot_clean(ax101, **kwargs)

        ax101.set_yscale('log')
        ax101.set_ylim(10e-3, 10e0)
        current_date = datetime.now().strftime('%Y-%m-%d')
        fig101.suptitle(f'{self.body.title()}\nGenerated on: {current_date}, '
                      f'\nAdded Plot Annotation and Zoomed Regions')

        # Creating a broken axis for the important regions in the bottom subplot
        #ax2 = fig1.add_subplot(spec[1])
        bax = brokenaxes(xlims=(zoom_regions[self.body]), subplot_spec=spec[1],
                         xscale='log', yscale='log')
        bax.loglog(self.wav, self.albedo, linewidth=0.3, color='black')
        wave_wrn = np.ma.masked_array(self.wav, mask=self.mask_wrn)
        data_wrn = np.ma.masked_array(self.albedo, mask=self.mask_wrn)
        bax.loglog(wave_wrn, data_wrn, linewidth=0.3, color='#121212')
        wave_tel = np.ma.masked_array(self.wav, mask=self.mask_tel)
        data_tel = np.ma.masked_array(self.albedo, mask=self.mask_tel)
        bax.loglog(wave_tel, data_tel, linewidth=0.3, color='lightgray')
        for ax in bax.axs:
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
            ax.xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
        #self.annotate_plot_clean(bax, **kwargs)

    def rv_compare(self):

        self.yiyo_rv(skipedge=20, num_pix=1500)
        self.rv_correction()

        solar_absorptions = [1.203485, 1.227412, 1.140696, 1.09180, 1.08727, 1.07899,
                             0.98920, 1.01487, 1.502919, 1.589281, 1.67553, 1.67236,
                             1.711338, 1.945829, 1.951114, 1.972798, 2.116965]
        solar_absorptions.sort()
        data_absorptions = [1.203104, 1.226985, 1.140339, 1.09144, 1.08689, 1.07864,
                            0.98903, 1.01461, 1.502442, 1.588782, 1.67501, 1.67189,
                            1.710817, 1.945225, 1.950529, 1.9721615, 2.116318]
        data_absorptions.sort()

        best_lines = [1,3,6,7,9,10,11,13,14,15]
        best_solars = [solar_absorptions[i] for i in best_lines]
        best_datas = [data_absorptions[i] for i in best_lines]

        fig = plt.figure(8)
        for idx, wv in enumerate([1,2,3]):
            i = idx + 1
            range = 0.001
            ax = fig.add_subplot(3, 1, i)
            slow = self.closest_index(self.sun_wave_ir, best_solars[wv] - range)
            shigh = self.closest_index(self.sun_wave_ir, best_solars[wv] + range)
            ax.plot(self.sun_wave_ir[slow:shigh], self.sun_flux_ir[slow:shigh] / np.mean(self.sun_flux_ir[slow:shigh]),
                    color='lightgray', linewidth=0.3, label='Sun')
            wlow = self.closest_index(self.wav, best_solars[wv] - range)
            whigh = self.closest_index(self.wav, best_solars[wv] + range)
            ax.plot(self.wav[wlow:whigh], self.flux[wlow:whigh] / np.mean(self.flux[wlow:whigh]), color='black',
                    linewidth=0.9, label='Original')
            ylow = self.closest_index(self.w_corr, best_solars[wv] - range)
            yhigh = self.closest_index(self.w_corr, best_solars[wv] + range)
            ax.plot(self.w_corr[ylow:yhigh], self.flux[ylow:yhigh] / np.mean(self.flux[ylow:yhigh]), color='forestgreen',
                    linewidth=0.9, label='CrossCorr')
            nlow = self.closest_index(self.rv_corr, best_solars[wv] - range)
            nhigh = self.closest_index(self.rv_corr, best_solars[wv] + range)
            ax.plot(self.rv_corr[nlow:nhigh], self.flux[nlow:nhigh] / np.mean(self.flux[nlow:nhigh]),
                    color='rebeccapurple', linewidth=0.9, label='Gaussfit')
            ax.axvline(best_solars[wv], color='orange', linestyle='--')
            ax.axvline(best_datas[wv], color='black', linestyle='--')
        plt.title('RV Correction Comparison')
        plt.legend()
        plt.show()

def multi_object_handler(object_list, **kwargs):
    '''
    Wrapper outside of the class to work with multiple bodies at once

    :param object_list:
    :param kwargs:
    :return:
    '''

    for o in object_list:
        obj = AlbedoAnalysis(body=o)
        print('Loading Solar Data')
        obj.load_solar_specs()
        print('Initializing Radial Velocity Correction')
        obj.rv_correction()
        print('Initializing Gaussian Convolution')
        obj.gaussian_convolution()
        print("Interpolating")
        obj.interpolate()
        obj.calculate_bond_albedo()
        obj.save_fits()

        scale_factor = np.percentile(obj.flux, 95) / np.percentile(obj.current_flux, 95)
        obj.current_flux *= scale_factor
        fig103, ax103 = plt.subplots(1, 1)
        ax103.plot(obj.wav, obj.flux, label='Before', color='k', linewidth=0.3)
        ax103.plot(obj.current_wav[1:], obj.current_flux, label='After', color='cornflowerblue', linewidth=0.3)

    ax103.set_xlabel("Wavelength (µm)")
    ax103.set_ylabel("Relative Flux")
    ax103.set_ylim(bottom=0.0, top=np.percentile(obj.flux, 95)*1.2)
    ax103.legend(loc='upper right')
    #fig103.suptitle("Solar Spectrum Correction Results")
    plt.savefig(f'C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Solar_spectrum_correction_results.png')
    fig103.show()

if __name__ == "__main__":
    bodies = ['uranus']
    alb = multi_object_handler(bodies)
