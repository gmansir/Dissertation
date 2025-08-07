import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d
from matplotlib.patches import Rectangle


class IRTFCompare:

    def __init__(self):

        self.irtf_path = "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Fry_Sromovsky\\plnt_Neptune.fits"
        self.neptune_path = "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\MOV_Neptune1_SCI_IFU_PAPER_READY.fits"
        #self.neptune_path = "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\MOV_Neptune_SCI_IFU_PAPER_READY_NEW.fits"
        self.hst_path = "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\HST_STIS\\hst_hasp_8661_stis_neptune_o65h\\hst_8661_stis_neptune_sg230l_o65ha4_cspec.fits"
        self.hst2_path = "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\HST_STIS\\hst_hasp_8661_stis_neptune_o65h\\hst_8661_stis_neptune_sg230l_o65ha5_cspec.fits"

    def closest_index(self, array, value):
        # Filter out NaN values and get the indices of non-NaN values
        valid_indices = ~np.isnan(array)
        valid_array = array[valid_indices]

        # Find the index of the closest value in the filtered array
        closest_valid_index = np.abs(valid_array - value).argmin()

        # Return the original index corresponding to the closest value
        return np.where(valid_indices)[0][closest_valid_index]

    def shade_mask_regions(self, ax, wave, mask, color='lightgray', alpha=0.5):
        """Shade regions where mask is True."""
        # Find boundaries of contiguous True regions
        in_region = False
        start = None
        for i in range(len(mask)):
            if mask[i] and not in_region:
                start = wave[i]
                in_region = True
            elif not mask[i] and in_region:
                end = wave[i]
                ax.axvspan(start, end, color=color, alpha=alpha, zorder=0)
                in_region = False
        # If ends in region
        if in_region:
            ax.axvspan(start, wave[-1], color=color, alpha=alpha, zorder=0)

    def compare_irtf_data(self):

        with fits.open(self.irtf_path) as hdul:
            data = hdul[0].data
            irtf_wave = data[0, :]
            irtf_flux = data[1, :]
            irtf_error = data[2, :]

        with fits.open(self.hst_path) as hdul:
            data = hdul[1].data
            hst_wave = data['WAVELENGTH'][0]  # in Angstroms
            hst_wave /= 1e4
            hst_flux = data['FLUX'][0]  # in erg / s / cm^2 / Å

        with fits.open(self.hst2_path) as hdul:
            data = hdul[1].data
            hst2_wave = data['WAVELENGTH'][0]  # in Angstroms
            hst2_wave /= 1e4
            hst2_flux = data['FLUX'][0]  # in erg / s / cm^2 / Å

        with fits.open(self.neptune_path) as hdul:
            primary_hdu = hdul[0].data
            flux_hdu = hdul[1].data
            lisird_flux = hdul['LISIRD_FLUX'].data
            resp_crv = hdul['RESP_CRV'].data
            mask_tel = np.invert([bool(x) for x in hdul['MASK_TEL'].data])
            mask_wrn = np.invert([bool(x) for x in hdul['MASK_WRN'].data])

            lengths = [len(primary_hdu), len(flux_hdu), len(lisird_flux),
                       len(resp_crv), len(mask_tel), len(mask_wrn)
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

        xsh_wave = data_table['wavelength']
        xsh_flux = data_table['albedo']

        # Define strong and weak telluric regions
        warning_regions = [
            (0.675, 0.695),
            (1.350, 1.440),
            (1.510, 1.700),
            (1.800, 2.180)
        ]
        teluric_regions = [
            (0.685, 0.700),
            (1.250, 1.350),
            (1.440, 1.510),
            (1.700, 1.800)
        ]

        def build_mask(wavelengths, regions):
            mask = np.zeros_like(wavelengths, dtype=bool)
            for lower, upper in regions:
                mask |= (wavelengths >= lower) & (wavelengths <= upper)
            return mask

        # Generate masks
        wavelengths = data_table['wavelength']
        mask_tel = build_mask(wavelengths, teluric_regions)
        mask_wrn = build_mask(wavelengths, warning_regions)

        # === Interpolate IRTF to XShooter ===
        irtf_interp_func = interp1d(irtf_wave, irtf_flux, kind='linear', bounds_error=False, fill_value='extrapolate')
        irtf_interp_flux = irtf_interp_func(xsh_wave)

        # === Scale IRTF to XShooter using overlapping region (1.0–1.2 µm) ===
        low = self.closest_index(xsh_wave, 1.0)
        high = self.closest_index(xsh_wave, 1.2)
        scale_irtf_to_xsh = np.nanmedian(xsh_flux[low:high] / irtf_interp_flux[low:high])
        irtf_flux_scaled = irtf_flux * scale_irtf_to_xsh
        irtf_interp_flux_scaled = irtf_interp_flux * scale_irtf_to_xsh

        # === Scale HST to XSH using ~0.31–0.32 µm ===
        hst_interp_func = interp1d(hst_wave, hst_flux, kind='linear', bounds_error=False, fill_value='extrapolate')
        hst_interp_flux = hst_interp_func(xsh_wave)
        hst2_interp_func = interp1d(hst2_wave, hst2_flux, kind='linear', bounds_error=False, fill_value='extrapolate')
        hst2_interp_flux = hst2_interp_func(xsh_wave)

        low = self.closest_index(xsh_wave, 0.31)
        high = self.closest_index(xsh_wave, 0.32)
        scale_hst_to_xsh = np.nanmedian(xsh_flux[low:high] / hst_interp_flux[low:high])
        scale_hst2_to_xsh = np.nanmedian(xsh_flux[low:high] / hst2_interp_flux[low:high])
        hst_flux_scaled = hst_flux * scale_hst_to_xsh
        hst2_flux_scaled = hst2_flux * scale_hst2_to_xsh
        hst_interp_flux_scaled = hst_interp_flux * scale_hst_to_xsh
        hst2_interp_flux_scaled = hst2_interp_flux * scale_hst2_to_xsh

        # calculate residuals
        # IRTF overlaps where both IRTF and XSHOOTER have data
        irtf_min = max(np.nanmin(irtf_wave), np.nanmin(xsh_wave))
        irtf_max = min(np.nanmax(irtf_wave), np.nanmax(xsh_wave))
        irtf_overlap = (xsh_wave >= irtf_min) & (xsh_wave <= irtf_max)

        # HST G230L (hst_wave) overlaps where both HST and XSHOOTER have data
        hst_min = max(np.nanmin(hst_wave), np.nanmin(xsh_wave))
        hst_max = min(np.nanmax(hst_wave), np.nanmax(xsh_wave))
        hst_overlap = (xsh_wave >= hst_min) & (xsh_wave <= hst_max)

        # HST G430L (hst2_wave) overlaps
        hst2_min = max(np.nanmin(hst2_wave), np.nanmin(xsh_wave))
        hst2_max = min(np.nanmax(hst2_wave), np.nanmax(xsh_wave))
        hst2_overlap = (xsh_wave >= hst2_min) & (xsh_wave <= hst2_max)

        # Initialize full-length residual arrays with NaNs
        irtf_residuals = np.full_like(xsh_flux, np.nan)
        hst_residuals = np.full_like(xsh_flux, np.nan)
        hst2_residuals = np.full_like(xsh_flux, np.nan)

        # Fill only the overlapping regions
        irtf_residuals[irtf_overlap] = irtf_interp_flux_scaled[irtf_overlap] - xsh_flux[irtf_overlap]
        hst_residuals[hst_overlap] = hst_interp_flux_scaled[hst_overlap] - xsh_flux[hst_overlap]
        hst2_residuals[hst2_overlap] = hst2_interp_flux_scaled[hst2_overlap] - xsh_flux[hst2_overlap]

        fig, (ax1, ax2) = plt.subplots(2, 1,
                                       gridspec_kw={'height_ratios': [2, 1], 'hspace': 0}, sharex=True)

        self.shade_mask_regions(ax1, xsh_wave, mask_tel, color='lightgray', alpha=0.3)
        self.shade_mask_regions(ax1, xsh_wave, mask_wrn, color='gray', alpha=0.3)
        self.shade_mask_regions(ax2, xsh_wave, mask_tel, color='lightgray', alpha=0.3)
        self.shade_mask_regions(ax2, xsh_wave, mask_wrn, color='gray', alpha=0.3)

        ax1.plot(irtf_wave, irtf_flux_scaled, lw=0.5, color='blue', label='IRTF')
        ax1.plot(xsh_wave, xsh_flux, lw=0.5, color='rebeccapurple', label='X-Shooter')
        ax1.plot(hst_wave, hst_flux_scaled, lw=0.5, color='red', label='HST')
        ax1.plot(hst2_wave, hst2_flux_scaled, lw=0.5, color='orchid', label='HST2')
        ax1.legend(loc='upper left')
        ax1.set_ylim(bottom=-0.1, top=1.25)
        ax1.set_xscale('log')

        ax2.plot(xsh_wave, irtf_residuals, lw=0.5, color='blue')
        ax2.plot(xsh_wave, hst_residuals, lw=0.5, color='red')
        ax2.plot(xsh_wave, hst2_residuals, lw=0.5, color='orchid')
        ax2.set_ylim(bottom=-0.5, top=0.5)
        ax2.set_xscale('log')

        ax2.set_xlabel(r"Wavelength ($\mu$m)")
        ax1.set_ylabel("Relative Flux")
        ax2.set_ylabel(r"$\Delta$")
        #plt.xlim(np.nanmin(xsh_wave), np.nanmax(xsh_wave))
        plt.xlim(0.25, np.nanmax(hst_wave))
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    instance = IRTFCompare()
    instance.compare_irtf_data()