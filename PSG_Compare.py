import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_widths
from glob import glob


class PlanetarySpectrumAnalyzer:
    def __init__(self, planet_name, directory):
        self.planet_name = planet_name
        self.directory = directory
        self.psg_data = None
        self.fits_data = None

    def load_psg_files(self, filenames=None):
        """Loads and combines PSG spectrum files for the planet."""

        if filenames:
            files = filenames
        else:
            pattern = os.path.join(self.directory, f"psg_{self.planet_name.lower()}*.txt")
            files = [file for file in glob(pattern) if 'cfg' not in file]

        if not files:
            raise FileNotFoundError(f"No PSG files found for {self.planet_name}.")

        wavelengths = []
        fluxes = []
        for file in files:
            print(f'Loading info for: {file}')
            with open(file, 'r') as f:
                lines = f.readlines()
                data = [line.split() for line in lines if not line.startswith('#')]
                wave, flux = zip(*[(float(d[0]), float(d[1])) for d in data])
                wavelengths.append(wave)
                fluxes.append(flux)

        # Combine multiple spectra if necessary
        return {
            'wave': np.median(wavelengths, axis=0),
            'flux': np.median(fluxes, axis=0),
        }

    def load_fits_files(self):
        """Loads and combines FITS spectrum files for the planet."""
        pattern = os.path.join(self.directory, f"MOV_{self.planet_name}*_SCI_IFU_PAPER_READY.fits")
        files = glob(pattern)

        if not files:
            raise FileNotFoundError(f"No FITS files found for {self.planet_name}.")

        waves = []
        fluxes = []
        for file in files:
            with fits.open(file) as hdul:
                wave = hdul[0].data
                flux = hdul[1].data
                waves.append(wave)
                fluxes.append(flux)

        # Combine multiple FITS data if necessary
        self.fits_data = {
            'wave': np.median(waves, axis=0),
            'flux': np.median(fluxes, axis=0),
        }

    def interpolate_psg_to_fits(self, target_wave=None, wave=None, flux=None):
        """Interpolates PSG data to FITS wavelength grid."""
        if target_wave is None:
            target_wave = self.fits_data['wave']
        if wave is None:
            wave = self.psg_data['wave']
        if flux is None:
            flux = self.psg_data['flux']

        interp_func = interp1d(wave, flux, kind='linear', bounds_error=False,
                               fill_value=0)

        return interp_func(target_wave)

    def compute_residuals(self):
        """Computes residuals between FITS and interpolated PSG data."""
        if 'flux_interpolated' not in self.psg_data:
            raise ValueError("PSG data must be interpolated to FITS grid before computing residuals.")

        self.residuals = self.fits_data['flux'] - self.psg_data['flux_interpolated']

    def compare_psg_spectra(self, psg_index_1, psg_index_2, wavelength_window=0.2):
        """Compares two PSG spectra against each other and plots the results."""

        psg1 = self.load_psg_files(filenames=[psg_index_1])
        psg2 = self.load_psg_files(filenames=[psg_index_2])

        # Interpolate PSG2 to PSG1's wavelength grid
        psg2_flux_interp = self.interpolate_psg_to_fits(psg1['wave'], psg2['wave'], psg2['flux'])

        # Compute residuals
        residuals = psg1['flux'] - psg2_flux_interp

        # Plot comparison
        wavelength_ranges = np.arange(psg1['wave'][0], psg1['wave'][-1], wavelength_window)
        n_panels = int(np.ceil(len(wavelength_ranges) / 3))

        for panel in range(n_panels):
            fig, axes = plt.subplots(6, 1, figsize=(10, 15),
                                     gridspec_kw={'height_ratios': [3, 1, 3, 1, 3, 1]})
            for i in range(3):
                idx = panel * 3 + i
                if idx >= len(wavelength_ranges) - 1:
                    break
                mask = (psg1['wave'] >= wavelength_ranges[idx]) & (psg1['wave'] < wavelength_ranges[idx + 1])

                # Plot PSG1 and interpolated PSG2 spectrum
                axes[i * 2].plot(psg1['wave'][mask], psg1['flux'][mask], label=f"PSG {psg_index_1}", color="blue")
                axes[i * 2].plot(psg1['wave'][mask], psg2_flux_interp[mask], label=f"PSG {psg_index_2}", color="green")
                if i == 0:  # Add legend only for the first section
                    axes[i * 2].legend()
                axes[i * 2].set_ylabel("Flux")

                # Plot residuals
                res_mask = residuals[mask]
                axes[i * 2 + 1].plot(psg1['wave'][mask], res_mask, color="pink")
                significant_deviation = np.abs(res_mask) > np.mean(np.abs(residuals)) + 2 * np.std(residuals)
                axes[i * 2 + 1].scatter(psg1['wave'][mask][significant_deviation], res_mask[significant_deviation],
                                        color="red", label="Significant")
                if i == 0:  # Add legend only for the first section
                    axes[i * 2 + 1].legend()
                axes[i * 2 + 1].set_ylabel("Residuals")

            axes[-1].set_xlabel("Wavelength [um]")
            plt.tight_layout()
            plt.show()


    def annotate_plot(self, ax, **kwargs):

        molecules = kwargs.get('molecules', ['H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2', 'NO', 'SO2', 'NO2',
                      'NH3', 'OH', 'HF', 'HCl', 'HBr', 'HI', 'OCS', 'N2', 'HCN', 'PH3',
                      'SF6', 'HO2'])
                               #['H2O', 'CO2', 'NH3', 'CO', 'CH4', 'NO2'])
        threshold = kwargs.get('threshold', 0.2)
        emission_threshold = kwargs.get('emission_threshold', 0.002)

        xlims = ax.get_xlim()
        x_sorted = sorted(xlims)
        ylims = ax.get_ylim()

        print(f'{x_sorted} Molecule list for annotation: {molecules}')
        #******************************************************************************************************************************************
        for mol_num, molecule in enumerate(molecules):
            print(f'Adding annotation for {molecule}')
            with fits.open(f'C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Elements\\hitran\\{molecule}_model.fits') as hdul:
                model_wave = hdul[0].data
                model_data = hdul[1].data
                in_range = (model_wave >= x_sorted[0]) & (model_wave <= x_sorted[1])
                if not np.any(in_range):
                    print(f"No data for {molecule} in range {x_sorted[0]}-{x_sorted[1]}, skipping.")
                    continue
                model_wave = model_wave[in_range]
                model_data = model_data[in_range]
                try:
                    norm_model = model_data / np.max(model_data)
                except ZeroDivisionError:
                    return

            print(f"Observation wavelength range: {x_sorted}")
            print(f"Model wavelength range: {model_wave.min()}-{model_wave.max()}")

            # Find features and calculate their full width
            # Calculate just a single value (try to use the same value, or note when the switch is)
            peaks, properties = find_peaks(norm_model, height=emission_threshold)
            left_edges = []
            right_edges = []
            for peak in peaks:
                left = peak
                while left > 0 and norm_model[left] > emission_threshold:
                    left -= 1
                right = peak
                while right < len(norm_model) - 1 and norm_model[right] > emission_threshold:
                    right += 1
                left_edges.append(left)
                right_edges.append(right)

            # Add all the indicies of the features into a nested list
            absorption_bands_all = []
            for i in range(len(peaks)):
                band_indices = list(range(left_edges[i], right_edges[i] + 1))
                absorption_bands_all.append(band_indices)

            # Check for overlaps within the bands and combine into a single feature
            no_overlaps = []
            try:
                current_band = absorption_bands_all[0]
            except IndexError:
                return
            for band_num in range(len(absorption_bands_all)-1):
                if any(x in absorption_bands_all[band_num] for x in absorption_bands_all[band_num+1]):
                    for i in absorption_bands_all[band_num+1]:
                        current_band.append(i)
                else:
                    no_overlaps.append(current_band)
                    current_band = absorption_bands_all[band_num+1]
            if current_band:
                no_overlaps.append(current_band)

            # Remove duplicate indicies ffrom the bands
            no_duplicates = []
            for band in no_overlaps:
                filtered = sorted(list(set(band)))
                no_duplicates.append(filtered)

            #Ensure that the region between the bands returns to the continuum
            absorption_bands = []
            current_band = no_duplicates[0]
            for band_num in range(len(no_duplicates)-1):
                in_between = range(no_duplicates[band_num][-1], no_duplicates[band_num+1][0])
                #ax22.plot(model_wave[in_between], norm_model[in_between], color='rebeccapurple')
                if np.mean(norm_model[in_between]) > threshold:
                    for i in no_duplicates[band_num+1]:
                        current_band.append(i)
                else:
                    absorption_bands.append(current_band)
                    current_band = no_duplicates[band_num+1]
            if current_band:
                absorption_bands.append(current_band)

            # Plot the annotation for each feature
            for band in absorption_bands:
                band_peaks, _ = find_peaks(norm_model[band])
                two_percent = (ylims[1] - ylims[0]) * 0.02 #+ (mol_num/-0.06)*0.0012
                #yloc = ylims[0] + (ylims[1] - ylims[0]) * 0.90 - two_percent
                yloc = ylims[0] + (ylims[1] - ylims[0]) * (0.90 - mol_num * 0.05)
                ax.fill_between(model_wave[band], yloc - two_percent/2, yloc + two_percent/2, color='rebeccapurple', alpha=0.6)
                #ax.vlines(model_wave[band[0]+band_peaks][-1],  ymin=0.0, ymax=(yloc - two_percent/2),
                #          alpha=0.5, linestyle='--', color='rebeccapurple', zorder=1)
                #cent = (model_wave[band[0]] + model_wave[band[-1]]) / 2
                max_idx = np.argmax(norm_model[band])
                cent = model_wave[band[max_idx]]
                ax.vlines(cent,  ymin=0.0, ymax=(yloc - two_percent/2),
                          alpha=0.5, linestyle='--', color='rebeccapurple', zorder=1)
                ax.text(np.median(model_wave[band]), yloc, molecule, ha='center', va='bottom', color='rebeccapurple')
                models = False
                if models == True:
                    ax.plot(model_wave, norm_model + ylims[0] + (ylims[1] - ylims[0]) * (0.85 - mol_num * 0.05),
                        label=f'{molecule} Model', alpha=0.7)
            #******************************************************************************************************************************************
            mol_num -= 0.06
        ax.set_xlim(xlims)

    def plot_spectra(self, wavelength_window=0.2):
        wavelength_ranges = np.arange(self.fits_data['wave'][0], self.fits_data['wave'][-1], wavelength_window)
        n_panels = int(np.ceil(len(wavelength_ranges) / 3))

        for panel in range(n_panels):
            fig, axes = plt.subplots(6, 1, figsize=(10, 15),
                                     gridspec_kw={'height_ratios': [3, 1, 3, 1, 3, 1]})
            for i in range(3):
                idx = panel * 3 + i
                if idx >= len(wavelength_ranges) - 1:
                    break
                mask = (self.fits_data['wave'] >= wavelength_ranges[idx]) & (self.fits_data['wave'] < wavelength_ranges[idx + 1])
                scaling_factor = np.nanmean(self.fits_data['flux'][mask]) / np.nanmean(self.psg_data['flux_interpolated'][mask])
                print(f'Scaling Factor: {scaling_factor}')
                scaling_factor = np.abs(scaling_factor)
                res = self.fits_data['flux'][mask] - self.psg_data['flux_interpolated'][mask]*scaling_factor

                # Plot data and PSG spectrum
                axes[i * 2].plot(self.fits_data['wave'][mask], self.fits_data['flux'][mask], label="Data", color="black")
                axes[i * 2].plot(self.fits_data['wave'][mask], self.psg_data['flux_interpolated'][mask]*scaling_factor, label="PSG", color="orange")
                if i == 0:  # Add legend only for the first section
                    axes[i * 2].legend()
                axes[i * 2].set_ylabel("Flux")
                self.annotate_plot(axes[i * 2])

                # Plot residuals
                #res_mask = self.residuals[mask]
                axes[i * 2 + 1].plot(self.fits_data['wave'][mask], res, color="pink")
                significant_deviation = np.abs(res) > np.mean(np.abs(res)) + 2 * np.std(res)
                axes[i * 2 + 1].scatter(self.fits_data['wave'][mask][significant_deviation], res[significant_deviation],
                                        color="red", label="Significant")
                if i == 0:  # Add legend only for the first section
                    axes[i * 2 + 1].legend()
                axes[i * 2 + 1].set_ylabel("Residuals")

            axes[-1].set_xlabel("Wavelength [um]")
            plt.suptitle(self.planet_name)
            plt.tight_layout()
            plt.show()

presets = {}

selected_preset = 'Titan'

if __name__ == "__main__":

    if selected_preset == 'psg_compare':
        analyzer = PlanetarySpectrumAnalyzer('Neptune', 'C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files')
        analyzer.compare_psg_spectra("C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\psg_neptune_trial.txt",
                                     "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\psg_nep_trial_no_aero.txt")
    else:
        planet = selected_preset
        analyzer = PlanetarySpectrumAnalyzer(planet, 'C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files')
        analyzer.psg_data = analyzer.load_psg_files()
        analyzer.load_fits_files()
        analyzer.psg_data['flux_interpolated'] = analyzer.interpolate_psg_to_fits()
        analyzer.compute_residuals()
        analyzer.plot_spectra()
        print(f"Start:{analyzer.fits_data['wave'][0]}")
        print(f"End: {analyzer.fits_data['wave'][-1]}")

