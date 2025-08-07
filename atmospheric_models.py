import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.signal import find_peaks
from astropy.io import fits

class AtmosphericModel:
    def __init__(self, element_names, resolution=1000, wavelength_range=(0.3, 2.5), grid_points=10000):
        """
        Initialize the AtmosphericModel with a list of elements.

        :param elements: List of tuples with element name and file path, e.g., [("Methane", "path_to_file.csv")]
        :param resolution: The spectral resolution of the model
        :param wavelength_range: Tuple specifying the start and end of the wavelength range in microns
        :param grid_points: Number of points in the wavelength grid
        """

        self.element_dict = {}

        directory_path = "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Elements\\hitran"
        for filename in os.listdir(directory_path):
            if filename.endswith(".out"):
                molecule_name = os.path.splitext(filename)[0]
                file_path = os.path.join(directory_path, filename)
                self.element_dict[molecule_name] = file_path
        directory_path = "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Elements"
        for filename in os.listdir(directory_path):
            if filename.endswith("_New.csv"):
                molecule_name = os.path.splitext(filename)[0]
                molecule_name = molecule_name[:-4]
                file_path = os.path.join(directory_path, filename)
                self.element_dict[molecule_name] = file_path

        if "Everything" in element_names:
            element_names.remove("Everything")
            element_names.extend(e for e in self.element_dict.keys())

        # Select elements based on provided names
        self.elements = dict.fromkeys(element_names)
        for name in element_names:
            self.elements[name] = self.element_dict[name]
        print(f"Working on the following elements: {self.elements.keys()}")
        self.element_specs = dict.fromkeys(element_names)
        self.resolution = resolution
        self.wavelength_range = np.linspace(wavelength_range[0], wavelength_range[1], grid_points)
        self.total_spectrum = np.zeros_like(self.wavelength_range)

    def _gaussian(self, wavelength, center, fwhm, amplitude):
        """Generate a Gaussian profile."""
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        return amplitude * np.exp(- (wavelength - center) ** 2 / (2 * sigma ** 2))

    def _add_element_spectrum(self, element_name, file_path, top_percent=50):
        """Load element data and add it to the total spectrum."""
        data = pd.read_csv(file_path, sep='\s+', low_memory=False)
        if "sw" in data.columns: # Catching errors potentially brought up from hitran files
            data = data.rename(columns={"nu": "Wavelength", "sw": "Absorption_Intensity"})
            data["Wavelength"] = 1e4 / data["Wavelength"]
        if "Absorption_Intensity" not in data.columns:
            data = pd.read_csv(file_path, sep='\t')

        strongest_lines = data.nlargest(int(len(data) * top_percent / 100), 'Absorption_Intensity')
        wavelengths = strongest_lines['Wavelength']
        intensities = strongest_lines['Absorption_Intensity']

        fwhms = wavelengths / self.resolution
        element_spectrum = np.zeros_like(self.wavelength_range)
        for center, fwhm, amplitude in zip(wavelengths, fwhms, intensities):
            element_spectrum += self._gaussian(self.wavelength_range, center, fwhm, amplitude)

        self.element_specs[element_name] = element_spectrum
        self.total_spectrum += element_spectrum

    def create_model(self):
        """Create the planetary atmosphere model by combining spectra from all elements."""
        for element_name in self.element_specs.keys():
            print(f'Adding information for {element_name} from file {self.elements[element_name]}')
            self._add_element_spectrum(element_name, self.elements[element_name])

    def plot_model(self, wavelength_step=0.2, feature_threshold=0.e-7):
        """
        Plot the total absorption spectrum with prominent features labeled.
        Only molecules with features in the current wavelength range are plotted.

        Parameters:
            wavelength_step (float): The wavelength span for each set of plots (in microns).
            feature_threshold (float): Minimum absorption intensity to consider as a feature.
        """
        min_wavelength, max_wavelength = min(self.wavelength_range), max(self.wavelength_range)
        current_start = min_wavelength

        while current_start < max_wavelength:
            current_end = current_start + wavelength_step
            mask = (self.wavelength_range >= current_start) & (self.wavelength_range < current_end)
            if not any(mask):
                current_start = current_end
                continue

            plt.figure(figsize=(10, 6))

            plotted_elements = []
            for element_name, spectrum in self.element_specs.items():
                data = spectrum[mask]

                # Check if the molecule has significant features in this range
                if max(data) >= feature_threshold:
                    plotted_elements.append(element_name)
                    plt.plot(self.wavelength_range[mask], data, label=element_name, linewidth=0.5)

                    # Find and annotate peaks
                    peaks, _ = find_peaks(data, height=feature_threshold)
                    for peak in peaks:
                        plt.annotate(f'{element_name}',
                                     (self.wavelength_range[mask][peak], data[peak]),
                                     textcoords="offset points",
                                     xytext=(0, 5),
                                     ha='center', fontsize=8, color='rebeccapurple')

            # Plot the total spectrum
            plt.plot(self.wavelength_range[mask], self.total_spectrum[mask],
                     label='Total Spectrum', linewidth=0.7, c='k')

            plt.xlim(current_start, current_end)
            plt.xlabel('Wavelength (microns)')
            plt.ylabel('Absorption Intensity')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Molecules")
            plt.title(f'Absorption Spectrum: {current_start:.2f} - {current_end:.2f} µm')
            plt.tight_layout()
            plt.show()

            current_start = current_end

    def save_model(self, filename):
        """Save the model to a FITS file."""
        primary_hdu = fits.PrimaryHDU(data=self.wavelength_range)
        save_model = fits.ImageHDU(self.total_spectrum)
        hdu = fits.HDUList([primary_hdu, save_model])
        hdu.writeto(filename, overwrite=True)


if __name__ == "__main__":
    # List of element names you want to include
    selected_element_names = ['H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2', 'NO', 'SO2', 'NO2',
                      'NH3', 'OH', 'HF', 'HCl', 'HBr', 'HI', 'OCS', 'N2', 'HCN', 'PH3',
                      'SF6', 'HO2']

    #selected_element_names = ['CH4']
    #Individuals
    for element in selected_element_names:
        print(f'Element: {element}')
        model = AtmosphericModel([element])
        model.create_model()
        model.plot_model(wavelength_step=0.5, feature_threshold=0.e-6)
        model.save_model(f"C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Elements\\hitran/{element}_model.fits")
        print(f'Saved Model: C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Elements\\hitran/{element}_model.fits')

    #model = AtmosphericModel(selected_element_names)
    #model.create_model()
    #model.plot_model(wavelength_step=1.0, feature_threshold=0.e-4)
    #model.save_model(f"C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Elements\\atmospheric_model.fits")
    #print(f'Saved Model: C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Elements\\atmospheric_model.fits')
