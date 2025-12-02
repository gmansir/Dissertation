import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import bottleneck as bn
from astropy.io import fits
from astropy.table import Table
from scipy.signal import find_peaks, peak_widths
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter, uniform_filter1d
import atmospheric_models as am
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker


class PaperReadyPlots:
    def __init__(self, bodies=[], molecules=[], xlims=(0.95, 1.2), ylims=(0., 2.),
                 threshold=1.0, tolerance=0.001, **kwargs):
        self.valid_bodies = ['titan', 'titan_lisird','enceladus', 'neptune1', 'neptune2', 'neptune3', 'neptune4',
                             'uranus1', 'uranus2', 'uranus3', 'uranus4', 'saturn_lisird', 'saturn1', 'saturn2',
                             'saturn3', 'saturn4', 'saturn5', 'saturn6', 'saturn7', 'saturn8',
                             'saturn9', 'saturn10', 'saturn11', 'saturn12', 'telluric', 'sunlike',
                             'ltt7987_titan', 'fiege-110', 'gd71', 'ltt7987_saturn', 'uranus', 'neptune', 'saturn']
        self.data = {}
        self.molecules = molecules
        self.xlims = xlims
        self.ylims = ylims
        self.threshold = threshold
        self.tolerance = tolerance
        self._check_bodies(bodies)
        self._load_data(bodies, **kwargs)

    def _check_bodies(self, bodies):
        for body in bodies:
            if body.lower() not in self.valid_bodies:
                raise ValueError(f"Invalid celestial body: {body}")

    def _load_data(self, bodies, **kwargs):
        for body in bodies:
            body = body.lower()
            print(f"Loading data for: {body}")
            work_dir = f'C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Spec_Files\\{body.title()}\\Post_Molecfit'
            fits_file_path = work_dir + f'\\MOV_{body.title()}_SCI_IFU_PAPER_READY.fits'
            print(f"File Path: {fits_file_path}")

            try:
                with fits.open(fits_file_path) as hdul:
                    primary_hdu = hdul[0].data
                    flux_hdu = hdul[1].data

                    if body == 'uranus' or body == 'neptune' or body == 'titan' or body == 'saturn':
                        window = 101
                        sigma_thresh = 5.0
                        med = bn.move_median(flux_hdu, window, min_count=1)
                        mad = bn.move_median(np.abs(flux_hdu - med), window, min_count=1)
                        bad = np.abs(flux_hdu - med) > sigma_thresh * mad
                        flux_hdu[bad] = med[bad]

                        bin_size = 50
                        n_bins = len(flux_hdu) // bin_size
                        flux_binned = np.nanmean(flux_hdu[:n_bins * bin_size].reshape(n_bins, bin_size), axis=1)
                        min_val = np.nanmin(flux_binned)
                        flux_hdu -= min_val

                    #flux_hdu /= np.nanmax(flux_hdu)


                    lisird_flux = hdul['LISIRD_FLUX'].data
                    resp_crv = hdul['RESP_CRV'].data
                    mask_tel = np.invert([bool(x) for x in hdul['MASK_TEL'].data])
                    mask_wrn = np.invert([bool(x) for x in hdul['MASK_WRN'].data])

                    lengths = [len(primary_hdu),len(flux_hdu),len(lisird_flux),
                               len(resp_crv),len(mask_tel),len(mask_wrn)
                    ]
                    max_length = np.max(lengths)

                    # Pad arrays with NaN to make them the same length
                    primary_hdu = np.pad(primary_hdu, (0, max_length - len(primary_hdu)), constant_values=np.nan)
                    flux_hdu = np.pad(flux_hdu, (0, max_length - len(flux_hdu)), constant_values=np.nan)
                    lisird_flux = np.pad(lisird_flux, (0, max_length - len(lisird_flux)), constant_values=np.nan)
                    resp_crv = np.pad(resp_crv, (0, max_length - len(resp_crv)), constant_values=np.nan)
                    mask_tel = np.pad(mask_tel, (0, max_length - len(mask_tel)), constant_values=np.nan)
                    mask_wrn = np.pad(mask_wrn, (0, max_length - len(mask_wrn)), constant_values=np.nan)

                    data_table = Table([primary_hdu, flux_hdu, lisird_flux, resp_crv, mask_tel, mask_wrn],
                                       names=('wavelength', 'albedo', 'lisird', 'response_curve', 'mask_tel', 'mask_wrn'))
                    self.data[body.lower()] = data_table

            except Exception as e:
                print(f"Error reading the FITS file for {body}: {e}")

    def create_synthetic_spectrum(self, mol_lines, intensities, wavelength_grid, resolution):
        absorption = np.zeros_like(wavelength_grid)
        for line_center, intensity in zip(mol_lines, intensities):
            fwhm = line_center / resolution
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
            absorption += intensity * np.exp(- (wavelength_grid - line_center) ** 2 / (2 * sigma ** 2))
        return absorption

    def annotate_plot(self, ax, **kwargs):

        molecules = kwargs.get('molecules', ['H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2', 'NO', 'SO2', 'NO2',
                                             'NH3', 'OH', 'HF', 'HCl', 'HBr', 'HI', 'OCS', 'N2', 'HCN', 'PH3',
                                             'SF6', 'HO2'])
        # ['H2O', 'CO2', 'NH3', 'CO', 'CH4', 'NO2'])
        threshold = kwargs.get('threshold', 0.2)
        emission_threshold = kwargs.get('emission_threshold', 0.002)

        xlims = ax.get_xlim()
        x_sorted = sorted(xlims)
        ylims = ax.get_ylim()

        print(f'{x_sorted} Molecule list for annotation: {molecules}')
        # ******************************************************************************************************************************************
        for mol_num, molecule in enumerate(molecules):
            print(f'Adding annotation for {molecule}')
            with fits.open(
                    f'C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Elements\\hitran\\{molecule}_model.fits') as hdul:
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
            for band_num in range(len(absorption_bands_all) - 1):
                if any(x in absorption_bands_all[band_num] for x in absorption_bands_all[band_num + 1]):
                    for i in absorption_bands_all[band_num + 1]:
                        current_band.append(i)
                else:
                    no_overlaps.append(current_band)
                    current_band = absorption_bands_all[band_num + 1]
            if current_band:
                no_overlaps.append(current_band)

            # Remove duplicate indicies ffrom the bands
            no_duplicates = []
            for band in no_overlaps:
                filtered = sorted(list(set(band)))
                no_duplicates.append(filtered)

            # Ensure that the region between the bands returns to the continuum
            absorption_bands = []
            current_band = no_duplicates[0]
            for band_num in range(len(no_duplicates) - 1):
                in_between = range(no_duplicates[band_num][-1], no_duplicates[band_num + 1][0])
                # ax22.plot(model_wave[in_between], norm_model[in_between], color='rebeccapurple')
                if np.mean(norm_model[in_between]) > threshold:
                    for i in no_duplicates[band_num + 1]:
                        current_band.append(i)
                else:
                    absorption_bands.append(current_band)
                    current_band = no_duplicates[band_num + 1]
            if current_band:
                absorption_bands.append(current_band)

            # Plot the annotation for each feature
            depth_threshold = kwargs.get('depth_threshold', 0.7)  # Adjust this value as needed

            for band in absorption_bands:
                band_peaks, _ = find_peaks(norm_model[band])

                # Find the deepest point in the absorption band
                min_idx = np.argmax(norm_model[band])
                min_depth = norm_model[band][min_idx]

                # Only annotate if the feature is deep enough
                two_percent = (ylims[1] - ylims[0]) * 0.02
                yloc = ylims[0] + (ylims[1] - ylims[0]) * (0.90 - mol_num * 0.06)

                if min_depth < depth_threshold:
                    ax.fill_between(model_wave[band], yloc - two_percent / 2, yloc + two_percent / 2,
                                color='rebeccapurple', alpha=0.8)

                    ax.vlines(model_wave[band][min_idx], ymin=0.0, ymax=(yloc - two_percent / 2),
                              alpha=0.5, linestyle='--', color='rebeccapurple', zorder=1)
                else:
                    ax.fill_between(model_wave[band], yloc - two_percent / 2, yloc + two_percent / 2,
                                    color='rebeccapurple', alpha=0.6)

                #for band in absorption_bands:
            #    band_peaks, _ = find_peaks(norm_model[band])
            #    two_percent = (ylims[1] - ylims[0]) * 0.02  # + (mol_num/-0.06)*0.0012
            #    # yloc = ylims[0] + (ylims[1] - ylims[0]) * 0.90 - two_percent
            #    yloc = ylims[0] + (ylims[1] - ylims[0]) * (0.90 - mol_num * 0.05)
            #    ax.fill_between(model_wave[band], yloc - two_percent / 2, yloc + two_percent / 2, color='rebeccapurple',
            #                    alpha=0.6)
            #    # ax.vlines(model_wave[band[0]+band_peaks][-1],  ymin=0.0, ymax=(yloc - two_percent/2),
            #    #          alpha=0.5, linestyle='--', color='rebeccapurple', zorder=1)
            #    # cent = (model_wave[band[0]] + model_wave[band[-1]]) / 2
            #    max_idx = np.argmax(norm_model[band])
            #    cent = model_wave[band[max_idx]]
            #    ax.vlines(cent, ymin=0.0, ymax=(yloc - two_percent / 2),
            #              alpha=0.5, linestyle='--', color='rebeccapurple', zorder=1)

                ax.text(np.median(model_wave[band]), yloc+0.003, molecule, ha='center', va='bottom', color='rebeccapurple')

                models = False
                if models == True:
                    ax.plot(model_wave, norm_model + ylims[0] + (ylims[1] - ylims[0]) * (0.85 - mol_num * 0.05),
                            label=f'{molecule} Model', alpha=0.7)
            # ******************************************************************************************************************************************
            mol_num -= 0.06
        ax.set_xlim(xlims)

    def closest_index(self, array, value):
        # Filter out NaN values and get the indices of non-NaN values
        valid_indices = ~np.isnan(array)
        valid_array = array[valid_indices]

        # Find the index of the closest value in the filtered array
        closest_valid_index = np.abs(valid_array - value).argmin()

        # Return the original index corresponding to the closest value
        return np.where(valid_indices)[0][closest_valid_index]

    def plot_albedo(self, object_list, molecules=None, xlims=None, ylims=None, threshold=None, tolerance=None,
                    offset=None, scaling_factor=None, solar_scale=None, solar=False, show_model=True,
                    triplet_plots=True, dynamic_ylims='percentile', **kwargs):

        molecules = molecules if molecules is not None else self.molecules
        xlims = xlims if xlims is not None else self.xlims
        ylims = ylims if ylims is not None else self.ylims
        threshold = threshold if threshold is not None else self.threshold
        tolerance = tolerance if tolerance is not None else self.tolerance
        buffer = offset if offset is not None else 0.06
        scaling_factor = scaling_factor if scaling_factor is not None else 1.0
        solar_scale = solar_scale if solar_scale is not None else 1.0
        show_model = show_model if show_model is not None else False

        master_mask_tel = np.zeros_like(self.data[object_list[0].lower()]['wavelength'], dtype=bool)
        master_mask_wrn = np.zeros_like(self.data[object_list[0].lower()]['wavelength'], dtype=bool)

        for obj_name in ['uranus', 'neptune', 'saturn', 'titan']:
            if obj_name.lower() not in self.data:
                continue
            mask_tel = self.data[obj_name.lower()]['mask_tel']
            mask_wrn = self.data[obj_name.lower()]['mask_wrn']
            if obj_name.lower() in ['titan', 'neptune', 'neptune2']:
                mask_tel = mask_tel[::-1]
                mask_wrn = mask_wrn[::-1]
            master_mask_tel |= mask_tel
            master_mask_wrn |= mask_wrn

        wave = self.data[object_list[0].lower()]['wavelength']

        extra_mask_tel = (
                ((wave >= 1.0) & (wave <= 1.1)) |
                ((wave >= 1.2) & (wave <= 1.3)) |
                ((wave >= 1.6) & (wave <= 1.8))
        )
        master_mask_tel |= extra_mask_tel  # combine it properly


        w, h = plt.figaspect(0.5)
        fig, ax = plt.subplots(figsize=(w, h))

        # Load and normalize the model if show_model is True
        if show_model:
            model = am.AtmosphericModel(molecules)
            model.create_model()
            model.plot_model(wavelength_step=0.2, feature_threshold=0.e-6)
            model.save_model("C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Elements\\planetary_atmosphere_model.fits")
            model_file = 'C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Elements\\planetary_atmosphere_model.fits'
            with fits.open(model_file) as hdul:
                model_wavelength = hdul[0].data - 0.01
                model_data = hdul[1].data
                x_sorted = sorted(xlims)
                in_range = (model_wavelength >= x_sorted[0]) & (model_wavelength <= x_sorted[1])
                model_wavelength = model_wavelength[in_range]
                model_data = model_data[in_range]
                max_value = np.max(model_data)
                if max_value == 0:
                    model_absorption = np.zeros_like(model_data)
                else:
                    model_absorption = 1 - model_data / max_value

        # Function to calculate range within the plotting range
        def calculate_range_within_limits(wave, flux, xlims):
            x_sorted = sorted(xlims)
            in_range = (wave >= x_sorted[0]) & (wave <= x_sorted[1])
            low_lim = np.nanmin(flux[in_range])
            mid_lim = np.nanmedian(flux[in_range])
            top_lim = np.nanmax(flux[in_range])
            std_lim = np.nanstd(flux[in_range])
            return [low_lim, mid_lim, top_lim, std_lim]

        # Calculate ranges for spectra in object_list
        stats = {}
        for obj_name in object_list:
            low, mid, top, std = calculate_range_within_limits(self.data[obj_name.lower()]['wavelength'],
                                                               self.data[obj_name.lower()]['albedo'], xlims)
            stats[obj_name] = {'mean': mid, 'std': std, 'max': top, 'low': low}

        # Calculate cumulative offsets dynamically
        cumulative_offset = 0.
        previous_max = 0.
        offsets = {}
        for obj_name in object_list:
            if obj_name.lower() in stats:
                mean = stats[obj_name]['mean']
                std = stats[obj_name]['std']
                offsets[obj_name] = cumulative_offset * scaling_factor
                cumulative_offset = mean - std + previous_max + buffer
                previous_max = stats[obj_name]['max'] + offsets[obj_name]

        count = 0
        last = len(object_list) - 1
        for obj_name in object_list:
            if obj_name.lower() not in self.data:
                print(f"No data available for {obj_name}")
                continue
            name_clean = re.sub(r'\d+$', '', obj_name)
            data = self.data[obj_name.lower()]
            wave, flux = data['wavelength'], data['albedo']
            mask_tel, mask_wrn = data['mask_tel'][::-1], data['mask_wrn'][::-1]
            mask_outside = np.logical_not(mask_tel & mask_wrn)

            # Predefined colors for objects
            object_colors = {
                'titan': 'goldenrod',
                'uranus': 'blue',
                'neptune': 'red',
                'saturn': 'green'
            }

            telluric_colors = {
                'titan': 'tan',
                'uranus': 'lightslategray',
                'neptune': 'rosybrown',
                'saturn': 'darkkhaki'
            }

            ax.plot(wave, flux, linewidth=0.4, color=object_colors[name_clean], label=name_clean)
            wave_tel = np.ma.masked_array(wave, mask=master_mask_tel)
            data_tel = np.ma.masked_array(flux, mask=master_mask_tel)
            ax.plot(wave_tel, data_tel, linewidth=0.4, color=telluric_colors[name_clean])
            wave_wrn = np.ma.masked_array(wave, mask=master_mask_wrn)
            data_wrn = np.ma.masked_array(flux, mask=master_mask_wrn)
            ax.plot(wave_wrn, data_wrn, linewidth=0.4, color=telluric_colors[name_clean])
            wave_outside = np.ma.masked_array(wave, mask=mask_outside)
            data_outside = np.ma.masked_array(flux, mask=mask_outside)
            #ax.plot(wave_outside, data_outside, color=object_colors[name_clean], linewidth=0.4, label=name_clean)

            # if name_clean == 'uranus':
           #     low1 = self.closest_index(wave, 1.02)
           #     high1 = self.closest_index(wave, 1.095)
           #     ax.plot(wave[low1:high1], flux[low1:high1], linewidth=0.5, color='lightslategray')
           #     low2 = self.closest_index(wave, 1.21)
           #     high2 = self.closest_index(wave, 1.31)
           #     ax.plot(wave[low2:high2], flux[low2:high2], linewidth=0.5, color='lightslategray')

            if show_model:
                # Scale model to fit each planet's spectrum
                scaled_model = model_absorption * (stats[obj_name]['mean']) + offsets.get(
                    obj_name, 0)
                ax.plot(model_wavelength, scaled_model, color='cornflowerblue', linestyle='--', linewidth=0.5, label='Model')

            if xlims is None:
                xlims = ax.get_xlim()
            if ylims is None:
                ylims = ax.get_ylim()


            yloc = self.closest_index(wave, xlims[0])
            xloc = xlims[0] - ((xlims[1] - xlims[0]) * 0.01)
            #clean_name = re.sub(r'\d+', '', obj_name)
            #ax.text(xloc, flux[yloc] + offsets.get(obj_name, 0), obj_name, verticalalignment='center',
            #        horizontalalignment='right')
            count += 1

        #ax.set_ylim(ylims)
        ax.set_xlim(xlims)
        ax.legend()

        if molecules:
            self.annotate_plot(ax, molecules=molecules, threshold=threshold, tolerance=tolerance)

        # Function to convert microns to wavenumber (cm^-1)
        def microns_to_wavenumber(microns):
            return 10000 / microns

        # Add secondary x-axis
       # secax_x = ax.secondary_xaxis('top', functions=(microns_to_wavenumber, microns_to_wavenumber))
       # secax_x.set_xlabel('Wavenumber (cm$^{-1}$)')

        ax.set_xlabel("Wavelength (µm)")
        ax.set_ylabel(r"erg s$^{-1}$ cm$^{-2}$ $\mu$m$^{-1}$", fontsize=20)
        plt.grid()
        #ax.yaxis.set_visible(False)
        #secax = ax.secondary_yaxis('right')
        #ax.spines['top'].set_visible(False)
        #ax.spines['left'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        #fig.suptitle("Comparative Planetology")
        plt.tight_layout()
        plt.savefig(f"C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Full_Specs.png")
        plt.show()

        if triplet_plots:
            # Define step size and calculate ranges
            step_size = kwargs.get('step_size', 0.634)  # Default step size of 0.1 microns
            total_range = xlims[1] - xlims[0]
            num_triplets = int(np.ceil(total_range / (3 * step_size)))  # Number of triplets

            # Loop over triplets
            for triplet_index in range(num_triplets):
                # Calculate the range for this triplet
                start_wavelength = xlims[0] + triplet_index * 3 * step_size
                end_wavelength = min(start_wavelength + 3 * step_size, xlims[1])

                # Create subplots for the triplet
                fig, axs = plt.subplots(3, 1, figsize=(15, 15))

                for subplot_index in range(3):
                    ax = axs[subplot_index]
                    sub_xlim = (
                        start_wavelength + subplot_index * step_size,
                        min(start_wavelength + (subplot_index + 1) * step_size, xlims[1])
                    )

                    # Concatenate albedo data for this range to calculate dynamic y-limits
                    combined_flux = []
                    for obj_name in object_list:
                        if obj_name.lower() not in self.data:
                            continue
                        data = self.data[obj_name.lower()]
                        wave, flux = data['wavelength'], data['albedo']
                        mask = (wave >= sub_xlim[0]) & (wave <= sub_xlim[1])
                        combined_flux.append(flux[mask])

                    combined_flux = np.concatenate(combined_flux)

                    # Plot data for each object
                    for obj_name in object_list:
                    #for obj_name in ['saturn5', 'titan']:
                        if obj_name.lower() not in self.data:
                            print(f"No data available for {obj_name}")
                            continue

                        data = self.data[obj_name.lower()]
                        wave, flux = data['wavelength'], data['albedo']
                        #mask_tel, mask_wrn = data['mask_tel'][::-1], data['mask_wrn'][::-1]
                        mask_outside = np.logical_not(master_mask_tel | master_mask_wrn)

                        name_clean = re.sub(r'\d+$', '', obj_name)
                        color = object_colors.get(name_clean.lower(), 'black')  # Default to black if object not defined

                        wave_tel = np.ma.masked_array(wave, mask=master_mask_tel)
                        data_tel = np.ma.masked_array(flux, mask=master_mask_tel)
                        ax.plot(wave_tel, data_tel, linewidth=0.4, color=telluric_colors[name_clean], alpha=0.6)
                        wave_wrn = np.ma.masked_array(wave, mask=master_mask_wrn)
                        data_wrn = np.ma.masked_array(flux, mask=master_mask_wrn)
                        ax.plot(wave_wrn, data_wrn, linewidth=0.4, color=telluric_colors[name_clean], alpha=0.6)
                        wave_outside = np.ma.masked_array(wave, mask=mask_outside)
                        data_outside = np.ma.masked_array(flux, mask=mask_outside)
                        ax.plot(wave_outside, data_outside, color=color, linewidth=0.4, label=name_clean)

                    if show_model:
                        scaled_model = model_absorption * (stats[obj_name]['mean']) + offsets.get(obj_name, 0)
                        ax.plot(model_wavelength, scaled_model, color='cornflowerblue', linestyle='--', linewidth=0.5,
                                label='Model')

                    ax.set_xlim(sub_xlim)
                    ax.set_ylim(bottom=4e-13)
                    ax.set_yscale('log')
                    self.annotate_plot(ax, molecules=molecules, threshold=threshold, tolerance=tolerance)
                    ax.set_xlabel("Wavelength (µm)", fontsize=20)
                    ax.xaxis.set_tick_params(labelsize=20)
                    #ax.yaxis.set_visible(False)
                    ax.yaxis.set_tick_params(labelsize=20)
                    ax.set_ylabel(r"erg s$^{-1}$ cm$^{-2}$ $\mu$m$^{-1}$", fontsize=16)
                    ax.spines['top'].set_visible(False)
                    #ax.spines['left'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                # Add legend to the first triplet figure
                if triplet_index == 0:
                    handles, labels = [], []
                    for obj_name, color in object_colors.items():
                        if obj_name in [obj.lower() for obj in object_list]:
                            handles.append(plt.Line2D([0], [0], color=color, lw=2))
                            labels.append(obj_name.capitalize())
                    axs[0].legend(handles, labels, loc='upper right')

                # Annotate and add a title for figure
                #plt.tight_layout()
                plt.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.95)
                plt.savefig(f"C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Full_Spec_{triplet_index}.png")
                plt.show()



        dont_trust = [[0.3,0.34], [0.50, 0.51], [0.758, 0.77], [0.685, 0.695], [0.625, 0.632],
                      [0.72, 0.73], [0.81, 0.83],[0.89, 0.98],
                      [1.107, 1.164], [1.3, 1.5], [1.73, 2.0], [2.38, 2.48], [1.946, 1.978],
                      [1.997, 2.032], [2.043, 2.080]]

        corrected = [[0.58, 0.60], [0.64, 0.66], [0.68, 0.75], [0.78, 0.86], [0.88, 1.0],
                     [1.062, 1.244], [1.26, 1.57], [1.63, 2.48]]

        if triplet_plots:
            xlims = [(wave[0], 1.03), (1.03, 1.64), (1.64, 2.12)]
            count = 1
            for xs in xlims:

                fig, ax = plt.subplots(len(object_list), 1, figsize=(10, 5), sharex=True)
                shared_ylabel = r"Flux (erg s$^{-1}$ cm$^{-2}$ $\mu$m$^{-1})$"
                fig.text(0.03, 0.5, shared_ylabel, va='center', rotation='vertical', fontsize=16)

                i = 0
                for obj_name in object_list:
                    data = self.data[obj_name.lower()]
                    wave, flux = data['wavelength'], data['albedo']

                    name_clean = re.sub(r'\d+$', '', obj_name)
                    color = object_colors.get(name_clean.lower(), 'black')  # Default to black if object not defined

                    ax[i].plot(wave, flux, color=color, linewidth=0.4, label=name_clean)
                    for m in dont_trust:
                        l = self.closest_index(wave, m[0])
                        r = self.closest_index(wave, m[1])
                        ax[i].plot(wave[l:r], flux[l:r], color=telluric_colors[obj_name], linewidth=0.4)
                    for m in corrected:
                        l = self.closest_index(wave, m[0])
                        r = self.closest_index(wave, m[1])
                        ax[i].plot(wave[l:r], flux[l:r], color=telluric_colors[obj_name], linewidth=0.4, alpha=0.5)

                    left = self.closest_index(wave, xs[0])
                    right = self.closest_index(wave, xs[-1])
                    mini_sec = flux[left:right]
                    if i == 0:
                        lower = np.nanpercentile(mini_sec, 5)
                    else:
                        lower = np.nanmin(mini_sec)
                    upper = np.nanmax(mini_sec)
                    ax[i].set_ylim(bottom=lower, top=upper)

                    ax[i].spines['top'].set_visible(False)
                    ax[i].spines['right'].set_visible(False)
                    ax[i].tick_params(axis='y', labelsize=10)

                    if count == 1:
                        ax[i].legend(loc='lower right')

                    if i == (len(object_list) - 1):
                        ax[i].set_xlim(xs[0], xs[1])
                        ax[i].set_xlabel("Wavelength (µm)", fontsize=15)
                        ax[i].xaxis.set_visible(True)
                        ax[i].xaxis.set_tick_params(labelsize=15)

                    i += 1


                plt.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9)
                plt.savefig(f"C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\Full_Spec_{count}.png")
                plt.show()
                count += 1



    def spectral_line_fitting(self, object_list, xlims, plot_xlims):

        def gaussian(x, amp, cen, width):
            return amp * np.exp(-(x - cen) ** 2 / (2 * width ** 2))

        def microns_to_wavenumber(microns):
            return 10000 / microns

        def continuum_fit(wavelength, flux, xlims):
            # Mask for the continuum fitting outside the absorption feature (i.e., outside xlims)
            mask_continuum = (wavelength < xlims[0]) | (wavelength > xlims[-1])
            # Fit a polynomial to the non-absorption region (outside xlims)
            fit = np.polyfit(wavelength[mask_continuum], flux[mask_continuum], deg=1)  # Linear fit (deg=1)
            return np.polyval(fit, wavelength)

        colors = {'saturn':'green', 'titan':'goldenrod', 'neptune':'red', 'uranus':'blue'}

        fig = plt.figure(1)
        ax1 = fig.add_subplot(111)
        plt.subplots_adjust(left=0.05, right=0.87, top=0.85, bottom=0.1)
        offset = 0
        for obj_name in object_list:
            clean_name = re.sub(r'\d+', '', obj_name)
            if obj_name.lower() not in self.data:
                print(f"No data available for {obj_name}")
                continue
            data = self.data[obj_name.lower()]
            wavelength, flux = data['wavelength'], data['albedo']

            xlims = sorted(xlims)
            xloc = plot_xlims[0] - ((plot_xlims[1] - plot_xlims[0]) * 0.01)
            mask = (wavelength > xlims[0]) & (wavelength < xlims[-1])
            wavelength_roi = wavelength[mask]
            flux_roi = flux[mask]
            inverse = 1/flux_roi

            plot_xlims = sorted(plot_xlims)
            plot_mask =(wavelength > plot_xlims[0]) & (wavelength < plot_xlims[-1])
            plot_wavelength = wavelength[plot_mask]
            plot_flux = flux[plot_mask]
            flux_error = 0.02 * plot_flux
            ax1.plot(plot_wavelength, plot_flux + offset, label=clean_name, color=colors[clean_name], linewidth=0.8)

            continuum = continuum_fit(plot_wavelength, plot_flux, xlims)
            absorption_continuum = (plot_wavelength > xlims[0]) & (plot_wavelength < xlims[-1])
            area = np.trapz(continuum[absorption_continuum] - flux_roi, wavelength_roi)
            sigma_area = np.sqrt(np.sum((flux_error[absorption_continuum] * (wavelength_roi[1] - wavelength_roi[0])) ** 2))
            ew = area / np.mean(absorption_continuum)
            continuum_mean = np.mean(continuum[absorption_continuum])  # Single value
            continuum_error = np.std(plot_flux[~absorption_continuum] - continuum[~absorption_continuum])
            sigma_ew = ew * np.sqrt((sigma_area / area) ** 2 + (continuum_error / continuum_mean) ** 2)

            exp = int(np.floor(np.log10(abs(ew)))) if ew != 0 else 0
            mant = ew / 10 ** exp
            err_digits = int(round(sigma_ew / 10 ** exp * 100))
            print(f"{clean_name} & ${mant:.2f}({err_digits:02d})\\mathrm{{e}}{{{exp:+d}}}$ \\\\")
            print(f"{clean_name}_ew = [{ew:.5f}]")
            print(f"{clean_name}_err = [{sigma_ew:.5f}]")

            ax1.plot(plot_wavelength, continuum + offset, linestyle='--', color='red')
            ax1.text(xloc, plot_flux[0] + offset, clean_name.title(), verticalalignment='center', horizontalalignment='right')
            ax1.hlines((plot_flux[0])+offset-0.05, xloc-ew, xloc, linewidth=1.2, color='black')

            try:
                # Initial guess for Gaussian parameters: [Amplitude, Center, Width]
                initial_guess = [np.max(inverse), np.mean(xlims), (xlims[1]-xlims[0])]
                params = curve_fit(gaussian, wavelength_roi, inverse, p0=initial_guess)[0]
                gaussian_curve = 1 / gaussian(wavelength_roi, *params) + offset
                gaussian_clipped = np.minimum(gaussian_curve, continuum[absorption_continuum] + offset)

                ax1.plot(wavelength_roi, gaussian_clipped, color='teal', linestyle='--')
                offset += np.max(plot_flux)

            except(RuntimeError):
                print(f'no good fit found for {clean_name}')

        secax_x = ax1.secondary_xaxis('top', functions=(microns_to_wavenumber, microns_to_wavenumber))
        secax_x.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax1.set_xlabel("Wavelength (µm)")
        #ax1.yaxis.set_label_position("right")
        ax1.yaxis.set_visible(False)
        secax = ax1.secondary_yaxis('right')
        secax.set_ylabel("Flux + Offset, Arbitrary Units", fontsize=12, rotation=90, labelpad=10)
        ax1.spines['top'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        fig.suptitle(f'{selected_preset} Band')
        plt.savefig(f'C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\{selected_preset}_spectral_fit.png')
        plt.show()

    def emission_line_fitting(self, object_list, xlims, plot_xlims):
        def gaussian(x, amp, cen, width):
            return amp * np.exp(-(x - cen) ** 2 / (2 * width ** 2))

        colors = {'saturn':'green', 'titan':'goldenrod', 'neptune':'red', 'uranus':'blue'}

        fig = plt.figure(1)
        ax1 = fig.add_subplot(111)
        plt.subplots_adjust(left=0.05, right=0.87, top=0.85, bottom=0.1)
        offset = 0
        for obj_name in object_list:
            if obj_name.lower() not in self.data:
                print(f"No data available for {obj_name}")
                continue
            data = self.data[obj_name.lower()]
            wavelength, flux = data['wavelength'], data['albedo']

            xlims = sorted(xlims)
            mask = (wavelength > xlims[0]) & (wavelength < xlims[-1])
            wavelength_roi = wavelength[mask]
            flux_roi = flux[mask]
            plot_mask = (wavelength > plot_xlims[0]) & (wavelength < plot_xlims[-1])
            wavelength_plot = wavelength[plot_mask]

            # Initial guess for Gaussian parameters: [Amplitude, Center, Width]
            initial_guess = [0.1, np.mean(xlims), 0.001]

            # Fit Gaussian to absorption feature
            # Calculate the area under the Gaussian curves (A = sqrt(2π) * amp * width)
            #params = curve_fit(gaussian, wavelength_roi, flux_roi, p0=initial_guess)[0]
            #wave_endpoints = [wavelength_roi[0], wavelength_roi[-1]]
            #flux_endpoints = [flux_roi[0], flux_roi[-1]]
            #fit = np.polyfit(wave_endpoints, flux_endpoints, deg=1)
            #baseline = np.polyval(fit, wavelength_roi)
            #area = np.trapz(flux_roi - baseline, wavelength_roi)
            #flux_error = 0.02 * flux_roi
            #sigma_area = np.sqrt(np.sum((flux_error * (wavelength_roi[1] - wavelength_roi[0])) ** 2))
            #ew = area / np.mean(baseline)
            #baseline_error = np.std(flux_roi - baseline)
            #sigma_ew = ew * np.sqrt((sigma_area / area) ** 2 + (baseline_error / np.mean(baseline)) ** 2)
            clean_name = re.sub(r'\d+', '', obj_name)
            #print(f'{clean_name} Equivalent Width: {ew:.5e}')
            #print(f'{clean_name} EW Sigma: {sigma_ew:.5e}')
            #print(f"{clean_name} & ${ew / 10 ** int(np.log10(abs(ew))):.2f}({int(round(sigma_ew / 10 ** int(np.log10(abs(ew))) * 100)):02d}) \\times 10^{{{int(np.log10(abs(ew)))}}}$ \\\\")


            plot_xlims = sorted(plot_xlims)
            xloc = plot_xlims[0] - ((plot_xlims[1] - plot_xlims[0]) * 0.01)
            flux_plot = flux[plot_mask]
            ax1.plot(wavelength_plot, flux_plot + offset, label=clean_name, color=colors[clean_name], linewidth=1.)
            #ax1.plot(wavelength_roi, baseline + offset, linestyle='--', color='red')
            #ax1.plot(wavelength_roi, gaussian(wavelength_roi, *params) + offset, color='teal', linestyle='--')
            ax1.text(xloc, flux_plot[0] + offset, clean_name.title(), verticalalignment='center',horizontalalignment='right')
            #ax1.hlines((flux_plot[0])+offset-0.025, xloc-ew, xloc, linewidth=1.2, color='black')
            offset += np.max(flux_plot)
            ax1 = plt.gca()  # Get current axis
            ax1.invert_xaxis()

        def microns_to_wavenumber(microns):
            return 10000 / microns

        secax_x = ax1.secondary_xaxis('top', functions=(microns_to_wavenumber, microns_to_wavenumber))
        secax_x.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax1.set_xlabel("Wavelength (µm)")
        ax1.set_ylabel("Flux + Offset, Arbitrary Units")
        ax1.yaxis.set_visible(False)
        secax = ax1.secondary_yaxis('right')
        secax.set_ylabel("Flux + Offset, Arbitrary Units", fontsize=12, rotation=90, labelpad=10)
        ax1.spines['top'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        fig.suptitle(r"$3\nu_2$ Band")
        plt.savefig(f'C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\{selected_preset}_spectral_fit.png')
        plt.show()

    def hyd_sulf(self, focus_region):

        full_region = (1.49, 1.64)

        uranus_data = self.data['uranus1']
        uwave, uflux = uranus_data['wavelength'], uranus_data['albedo']
        umask = (uwave > full_region[0]) & (uwave < full_region[-1])
        uwave, uflux = uwave[umask], np.array(uflux[umask], dtype='float64')
        uranus_mean_flux = np.mean(uflux)

        neptune_data = self.data['neptune1']
        nwave, nflux = neptune_data['wavelength'], neptune_data['albedo']
        nmask = (nwave > full_region[0]) & (nwave < full_region[-1])
        nwave, nflux = nwave[nmask], nflux[nmask]
        nflux = nflux * 1.7534949899518348

        w, h = plt.figaspect(0.5)
        fig, ax, = plt.subplots(figsize=(w, h))
        ax.plot(uwave, uflux, linewidth=0.4, label='Uranus', color='blue')
        ax.plot(nwave, nflux, linewidth=0.4, label='Neptune', color='red')
        plt.xlabel("Wavelength (µm)", fontsize=16)
        ax.set_ylabel("Relative Flux", fontsize=16)
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)
        plt.legend()
        plt.savefig('C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\UraNepHydSulfFull.png')
        plt.show()

        uranus_data = self.data['uranus1']
        uwave, uflux = uranus_data['wavelength'], uranus_data['albedo']
        umask = (uwave > focus_region[0]) & (uwave < focus_region[-1])
        uwave, uflux = uwave[umask], np.array(uflux[umask], dtype='float64')
        uranus_mean_flux = np.mean(uflux)

        neptune_data = self.data['neptune1']
        nwave, nflux = neptune_data['wavelength'], neptune_data['albedo']
        nmask = (nwave > focus_region[0]) & (nwave < focus_region[-1])
        nwave, nflux = nwave[nmask], nflux[nmask]
        neptune_mean_flux = np.mean(nflux)
        #scaling_factor = uranus_mean_flux / neptune_mean_flux
        nflux = nflux * 1.7534949899518348

        # Apply a median filter with a large kernel size to the flux data
        kernel_size = 150  # Adjust as needed
        flattened_nflux = median_filter(nflux, size=kernel_size)
        flattened_uflux = median_filter(uflux, size=kernel_size)
        adjusted_nflux = nflux - flattened_nflux + np.mean(flattened_nflux)
        adjusted_uflux = uflux - flattened_uflux + np.mean(flattened_uflux)

        # Plotting to visualize the effect
        #w, h = plt.figaspect(0.5)
        #fig2, ax2 = plt.subplots(figsize=(w, h))
        #ax2.plot(uwave, adjusted_uflux, linewidth=0.4, label='Uranus', color='blue')
        #ax2.plot(nwave, adjusted_nflux, linewidth=0.4, label='Neptune', color='red')
        #plt.xlabel("Wavelength (µm)", fontsize=16)
        #ax2.xaxis.set_tick_params(labelsize=14)
        #ax2.yaxis.set_tick_params(labelsize=14)
        #plt.legend()
        #plt.title("Flattened Spectra", fontsize=16)
        #plt.savefig('C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\UraNepFlattened.png')
        #plt.show()

        interpolated_nflux = np.interp(uwave, nwave, adjusted_nflux)
        comparison = interpolated_nflux - adjusted_uflux
        #mid = np.median(comparison)
        #comparison = comparison - mid
        #u_std = np.std(comparison)
        u_dist = np.linalg.norm(comparison)
        #fig3, ax3 = plt.subplots(figsize=(w, h))
        #plt.legend()
        #plt.ylim(-0.2, 0.6)
        #plt.xlabel("Wavelength (µm)", fontsize=16)
        #ax3.xaxis.set_tick_params(labelsize=14)
        #ax3.yaxis.set_tick_params(labelsize=14)
        #plt.title("Comparison", fontsize=16)
        #plt.savefig('C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\UraNepCompare.png')
        #plt.show()


        w, h = plt.figaspect(0.75)
        fig, (ax, ax2) = plt.subplots(2, 1,figsize=(w, h), gridspec_kw={'height_ratios': [4, 1], 'hspace': 0},
            sharex=True)
        ax.plot(uwave, uflux, linewidth=0.4, label='Uranus', color='blue')
        ax.plot(nwave, nflux, linewidth=0.4, label='Neptune', color='red')
        ax2.plot(uwave, comparison, linewidth=0.4, label=f'std: {u_dist:.5f}', color='blue')
        plt.xlabel("Wavelength (µm)", fontsize=16)
        ax.set_ylabel("Relative Flux", fontsize=16)
        ax2.xaxis.set_tick_params(labelsize=14)
        ax2.yaxis.set_tick_params(labelsize=10)
        ax.yaxis.set_tick_params(labelsize=12)
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=2))
        ax2.set_ylabel(r"$\Delta$", fontsize=14)
        ax.legend()
        plt.savefig('C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\UraNepHydSulf.png')
        plt.show()


        colors = {'saturn5':'green', 'titan':'orange'}

        for planet in colors.keys():
            clean_name = re.sub(r'\d+', '', planet)
            planet_data = self.data[planet]
            pwave, pflux = planet_data['wavelength'], planet_data['albedo']
            pmask = (pwave > focus_region[0]) & (pwave < focus_region[-1])
            pwave, pflux = pwave[pmask], pflux[pmask]

            planet_mean_flux = np.mean(pflux)
            scaling_factor = uranus_mean_flux / planet_mean_flux
            pflux = pflux * scaling_factor

            # Apply a median filter with a large kernel size to the flux data
            flattened_pflux = median_filter(pflux, size=kernel_size)
            adjusted_pflux = pflux - flattened_pflux + np.mean(flattened_pflux)

            #w, h = plt.figaspect(0.5)
            #fig2, ax2 = plt.subplots(figsize=(w, h))
            #ax2.plot(uwave, adjusted_uflux, linewidth=0.4, label='Uranus', color='blue')
            #ax2.plot(nwave, adjusted_nflux, linewidth=0.4, label='Neptune', color='red')
            #ax2.plot(pwave, adjusted_pflux, linewidth=0.4, label=clean_name.title(), color=colors[planet])
            #plt.xlabel("Wavelength (µm)", fontsize=16)
            #ax2.xaxis.set_tick_params(labelsize=14)
            #ax2.yaxis.set_tick_params(labelsize=14)
            #plt.legend()
            #plt.title("Flattened Spectra", fontsize=16)
            #plt.savefig(f'C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\{planet.title()}Flattened.png')
            #plt.show()

            #interpolation to the same wavelength grid
            interpolated_pflux_u = np.interp(uwave, pwave, adjusted_pflux)
            interpolated_pflux_n = np.interp(nwave, pwave, adjusted_pflux)

            #differences
            comparison_u = interpolated_pflux_u - adjusted_uflux
            comparison_n = interpolated_pflux_n - adjusted_nflux

            #standard deviations
            #u_std = np.std(comparison_u)
            #n_std = np.std(comparison_n)

            #Euclidean Distances:
            u_dist = np.linalg.norm(comparison_u)
            n_dist = np.linalg.norm(comparison_n)

            #Error estimate
            approx_sigma = np.std(comparison_u)
            n_points = len(comparison_u)
            approx_err_u = approx_sigma * np.sqrt(n_points)

            approx_sigma = np.std(comparison_n)
            n_points = len(comparison_n)
            approx_err_n = approx_sigma * np.sqrt(n_points)

            #fig3, ax3 = plt.subplots(figsize=(w, h))
            #plt.ylim(0., 4.)
            #plt.xlabel("Wavelength (µm)", fontsize=16)
            #ax3.xaxis.set_tick_params(labelsize=14)
            #ax3.yaxis.set_tick_params(labelsize=14)
            #plt.title("Comparison", fontsize=16)
            #plt.legend()
            #plt.savefig(f'C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\{planet.title()}Compare.png')
            #plt.show()

            w, h = plt.figaspect(0.75)
            fig, (ax, ax2) = plt.subplots(2, 1, figsize=(w, h), gridspec_kw={'height_ratios': [4, 1], 'hspace': 0},
                                          sharex=True)
            ax.plot(uwave, uflux, linewidth=0.4, label='Uranus', color='blue')
            ax.plot(nwave, nflux, linewidth=0.4, label='Neptune', color='red')
            ax.plot(pwave, pflux, linewidth=0.4, label=clean_name.title(), color=colors[planet])
            ax2.plot(uwave, comparison_u, linewidth=0.4, label=f'std: {u_dist:.5f}', color='blue')
            ax2.plot(nwave, comparison_n, linewidth=0.4, label=f'std: {n_dist:.5f}', color='red')
            print(f'{clean_name} v uranus: {u_dist:.5f}, {approx_err_u:.5f}')
            print(f'{clean_name} v neptune: {n_dist:.5f}, {approx_err_n:.5f}')
            ax2.yaxis.set_major_locator(MaxNLocator(nbins=2))
            plt.xlabel("Wavelength (µm)", fontsize=16)
            ax.set_ylabel("Relative Flux", fontsize=16)
            ax2.xaxis.set_tick_params(labelsize=14)
            ax2.yaxis.set_tick_params(labelsize=10)
            ax.yaxis.set_tick_params(labelsize=12)
            ax.legend()
            #plt.suptitle(f"{clean_name.title()}", fontsize=16)
            #ax.text(0.95,0.95,f"{clean_name.title()}", fontsize=14, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
            ax2.set_ylabel(r"$\Delta$", fontsize=14)
            plt.savefig(f'C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\{planet.title()}HydSulf.png')
            plt.show()

    def ew_graph(self):

        band = [0.619, 0.727, 1.0, 1.2]
        titan = [0.00102, 0.00580, 0.01410, 0.01197]
        uranus = [0.01189, 0.01261, 0.01907, 0.00336]
        neptune = [0.01307, 0.00809, 0.01083, 0.00230]
        saturn = [0.00198, 0.00484, 0.02403, 0.01846]

        titan_err = [0.00004, 0.00013, 0.00023, 0.00040]
        uranus_err = [0.00056, 0.00082, 0.00145, 0.00005]
        neptune_err = [0.00078, 0.00102, 0.00090, 0.00004]
        saturn_err = [0.00009, 0.00021, 0.00224, 0.00104]

        values = {'saturn':saturn, 'titan':titan, 'neptune':neptune, 'uranus':uranus}
        errors = {'saturn':saturn_err, 'titan':titan_err, 'neptune':neptune_err, 'uranus':uranus_err}
        colors = {'saturn':'green', 'titan':'goldenrod', 'neptune':'red', 'uranus':'blue'}

        fig = plt.figure(1)
        ax1 = fig.add_subplot(111)
        for p in ['saturn', 'titan', 'neptune', 'uranus']:
            #ax1.plot(band, values[p], color=colors[str(p)], label=p, marker='o', linestyle='-')
            ax1.errorbar(band, values[p], yerr=errors[p],
                color=colors[p], label=p.title(), marker='o', linestyle='-',
                capsize=3, elinewidth=1.2
            )
        plt.xlabel('Band (µm)', fontsize=16)
        plt.ylabel('Equivalent Width (µm)', fontsize=16)
        plt.grid()
        plt.legend()
        plt.savefig(f'C:\\Users\\ninam\\Documents\\Chile_Stuff\\Dissertation\\EWs.png')
        plt.show()


# Define presets as dictionaries
presets = {
    'HydSulf': {
        'bodies': ['uranus1', 'neptune1', 'saturn5', 'titan'],
        'molecules': [],
        #'xlims': (1.49, 1.64),
        'xlims': (1.57, 1.62),
        'ylims': (-0.06, 0.025)
    },
    'Full Specs': {
        'bodies': ['titan', 'saturn', 'uranus', 'neptune'],
        'molecules': [],
        'xlims': (0.31, 2.1),
        #'xlims': (1.2, 1.35),
        'offset': 0.8,
        'ylims': (0.0, 1.4),
        'threshold': 0.20,
        'tolerance': 0.0001,
        'solar': True,
        'solar_scale': 5.0,
        'scaling_factor': 0.5,
        'show_model': False,
        'step_size': 1.0,
    },
    'neptune': {
        'bodies': ['neptune'],
        'molecules': [],
        'xlims': (0.31, 2.1),
        'offset': 0.8,
        'ylims': (0.0, 1.e-12),
        'threshold': 0.20,
        'tolerance': 0.0001,
        'solar': True,
        'solar_scale': 5.0,
        'scaling_factor': 0.5,
        'show_model': False,
        'step_size': 1.0,
    },
    'titan': {
        'bodies': ['titan'],
        'molecules': [],
        'xlims': (0.31, 2.1),
        'offset': 0.8,
        'ylims': (0.0, 1.e-12),
        'threshold': 0.20,
        'tolerance': 0.0001,
        'solar': True,
        'solar_scale': 5.0,
        'scaling_factor': 0.5,
        'show_model': False,
        'step_size': 1.0,
    },
    'saturn': {
        'bodies': ['saturn'],
        'molecules': [],
        'xlims': (0.31, 2.1),
        'offset': 0.8,
        'ylims': (0.0, 1.e-12),
        'threshold': 0.20,
        'tolerance': 0.0001,
        'solar': True,
        'solar_scale': 5.0,
        'scaling_factor': 0.5,
        'show_model': False,
        'step_size': 1.0,
    },
    'tisat': {
        'bodies': ['titan','saturn'],
        'molecules': [],
        'xlims': (0.31, 2.1),
        'offset': 0.8,
        'ylims': (0.0, 1.e-12),
        'threshold': 0.20,
        'tolerance': 0.0001,
        'solar': True,
        'solar_scale': 5.0,
        'scaling_factor': 0.5,
        'show_model': False,
        'step_size': 1.0,
    },
    'Lutz, Owen, Cess': {
        'bodies': ['neptune', 'uranus'],
        'molecules': ['ch4'],
        'xlims': (0.4, 0.71),
        'offset': 0.5,
        'ylims': (0.0, 0.8),
        'threshold': 0.10,
        'tolerance': 0.0001,
        'solar': True,
        'solar_scale': 1.0,
        'scaling_factor': 0.1,
        'show_model': False
    },
    'band_test': {
        'bodies': ['Saturn5', 'Titan'],
        #'molecules': ['Methane', 'Ammonia', 'Water', 'CH4', 'CO2', 'H2O', 'HCl',
        #              'HF', 'HNO3', 'HO2', 'HOBr', 'HOCl', 'N2O', 'NH3', 'NO', 'NO2',
        #              'NO+', 'O', 'O3', 'OCS', 'OH', 'PH3', ],
        'molecules': ['H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'NO', 'SO2', 'NO2',
                      'NH3', 'OH', 'HF', 'HCl', 'HBr', 'HI', 'OCS', 'N2', 'HCN', 'PH3',
                      'SF6', 'HO2'],
        #'molecules': ['H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2', 'NO', 'SO2', 'NO2',
        #              'NH3', 'OH', 'HF', 'HCl', 'HBr', 'HI', 'OCS', 'N2', 'HCN', 'PH3',
        #              'SF6', 'HO2'],
        'xlims': (0.3, 2.2),
        #'xlims': (0.6, 0.65),
        'offset': 1.0,
        'ylims': (0.0, 0.35),
        'threshold': 0.20,
        'tolerance': 0.00001,
        'solar': True,
        'solar_scale': 15.0,
        'scaling_factor': 0.3,
        'show_model': True,
        'step_size': 1.0,
    },
    '3nu2': {
        'bodies': ['saturn', 'titan', 'uranus', 'neptune'],
        'xlims': (1.5552, 1.5565),
        #'plot_xlims': (1.55, 1.56)
        'plot_xlims': (1.50, 1.60)
    },
    'Methane-0.54': {
        'bodies': ['titan', 'uranus', 'neptune', 'saturn'],
        'xlims': (0.537, 0.55),
        'plot_xlims': (0.52, 0.57)
    },
    'Methane-0.619': {
        'bodies': ['titan', 'uranus', 'neptune', 'saturn'],
        'xlims': (0.606, 0.63),
        'plot_xlims': (0.60, 0.64)
    },
    'Methane-0.727': {
        'bodies': ['titan', 'uranus', 'neptune', 'saturn'],
        'xlims': (0.715, 0.74),
        'plot_xlims': (0.71, 0.75)
    },
    'Methane-1': {
        'bodies': ['titan', 'uranus', 'neptune', 'saturn'],
        'xlims': (0.94, 1.075),
        'plot_xlims': (0.937, 1.08)
    },
    'Methane-1.2': {
        'bodies': ['titan', 'uranus', 'neptune', 'saturn'],
        'xlims': (1.08, 1.28),
        'plot_xlims': (1.065, 1.3)
    },
    'Ammonia': {
        'bodies': ['saturn', 'titan', 'uranus', 'neptune'],
        'xlims': (1.558, 1.574),
        'plot_xlims': (1.555, 1.58)
    },
    'Ammonia-2': {
        'bodies': ['titan', 'saturn', 'uranus', 'neptune'],
        'xlims': (1.925, 2.025),
        'plot_xlims': (1.923, 2.05)
    },
    'OH': {
        'bodies': ['titan', 'saturn', 'uranus', 'neptune'],
        'xlims': (0.61, 0.63),
        'plot_xlims': (0.59, 0.64)
        #'xlims': (0.6175, 0.6265),
        #'plot_xlims': (0.605, 0.63)

    },
    'EWs': {
        'bodies': ['titan', 'saturn', 'uranus', 'neptune'],

    },
}

selected_preset = ('Full Specs')

if __name__ == "__main__":
    params = presets[selected_preset]
    bands = ['Methane-0.727', 'Methane-0.54', 'Methane-0.619', 'Methane-1', 'Methane-1.2', 'Ammonia',
             'Ammonia-2', 'OH']
    double_bands = ['3nu2']
    EWs = ['EWs']

    prp = PaperReadyPlots(params['bodies'])
    if selected_preset in bands:
        prp.spectral_line_fitting(params['bodies'], params['xlims'], params['plot_xlims'])
    elif selected_preset in double_bands:
        prp.emission_line_fitting(params['bodies'], params['xlims'], params['plot_xlims'])
    elif selected_preset == 'HydSulf':
        prp.hyd_sulf(params['xlims'])
    elif selected_preset in EWs:
        prp.ew_graph()
    else:
        prp.plot_albedo(
            params['bodies'],
            molecules=params['molecules'],
            xlims=params['xlims'],
            offset=params['offset'],
            ylims=params['ylims'],
            threshold=params['threshold'],
            tolerance=params['tolerance'],
            solar=params['solar'],
            solar_scale=params['solar_scale'],
            scaling_factor=params['scaling_factor'],
            show_model=params['show_model']
        )


    #xlims=(0.52, 0.63)
    #xlims=(1.57, 1.54)
    #xlims=(0.31, 2.0)

# TITAN"S MASK IS BACKWARDS and neptune's

#ratio between sun and lisird
#moon phase/altitude => poorly corrected saturn? => justification for better weather in future proposals

# Apply wavelength correction by band
# Line identification below sun
# Instead of Gaussian fits for the lines, use a set cut-off that the absorption must be stronger than
    # Make sure that the cut-off changes with the scale of the model
# Run Moog outside of the vnc (screen)

# Read into the literature and compare well known lines (methane).
    # Find views with clear features and ensure that my labeling is consistant
    # Verify known lines be expolring new features
