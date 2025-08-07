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
                'UVB': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_1\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_1\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_1\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_1\\MOV_Neptune_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_1\\MOV_Neptune_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_1\\MOV_Neptune_DiskIntegrated_NIR.fits",
            },
            'neptune2': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_2\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_2\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_2\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_2\\MOV_Neptune_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_2\\MOV_Neptune_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_2\\MOV_Neptune_DiskIntegrated_NIR.fits",
            },
            'neptune3': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_3\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_3\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_3\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_3\\MOV_Neptune_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_3\\MOV_Neptune_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_3\\MOV_Neptune_DiskIntegrated_NIR.fits",
            },
            'neptune4': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_4\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_4\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_4\\MOV_Neptune_SCI_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_4\\MOV_Neptune_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_4\\MOV_Neptune_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\Offset_4\\MOV_Neptune_DiskIntegrated_NIR.fits",
            },
            'feige-110': {
                'UVB': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\FEIGE-110\\FEIGE-110_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_UVB.fits",
                'VIS': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\FEIGE-110\\FEIGE-110_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_VIS.fits",
                'NIR': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\FEIGE-110\\FEIGE-110_onoff_IFU_FLUX_IFU_MERGE3D_DATA_OBJ_NIR.fits",
                'DI_UVB': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\FEIGE-110\\MOV_FEIGE-110_DiskIntegrated_UVB.fits",
                'DI_VIS': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\FEIGE-110\\MOV_FEIGE-110_DiskIntegrated_VIS.fits",
                'DI_NIR': "C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\Spec_Files\\Neptune-Check\\FEIGE-110\\MOV_FEIGE-110_DiskIntegrated_NIR.fits",
            }
        }
        self.file_paths = all_paths[object]

    def combine_disk_integrated_spectra(self, object_list, arm):

        spectra = []

        for obj in object_list:
            try:
                path = MolecfitPrep(obj).file_paths[f'DI_{arm}']
                if not os.path.exists(path):
                    continue
                with fits.open(path) as hdul:
                    data = hdul[0].data
                    if data is None or not np.any(data):
                        continue
                    spectra.append(data)
            except Exception as e:
                print(f"Skipping {obj} due to error: {e}")
                continue

        if len(spectra) == 0:
            print(f"No valid DI spectra found for arm {arm}. Falling back to post xshooter pipeline cubes.")
            for obj in object_list:
                try:
                    path = MolecfitPrep(obj).file_paths[arm]
                    if not os.path.exists(path):
                        continue
                    with fits.open(path) as hdul:
                        cube = hdul[0].data
                        if cube is None or not np.any(cube):
                            continue
                        summed = np.nansum(cube, axis=(1, 2))
                        spectra.append(summed)
                except Exception as e:
                    print(f"Skipping post_xpipe for {obj} due to error: {e}")
                    continue

        if len(spectra) == 0:
            raise ValueError(f"No valid spectra found for arm {arm} in either DI or post_xpipe.")

        combined = np.nansum(np.array(spectra), axis=0)
        return combined

    def plot_check(self, wavefile, image_path):

        with fits.open(wavefile) as hdul:
            header = hdul[0].header

            # Wavelength axis
            N = header["NAXIS3"]
            wave = np.zeros(N)
            for i in range(N):
                wave[i] = (i + 1 - header["CRPIX3"]) * header["CDELT3"] + header["CRVAL3"]

        with fits.open(image_path) as hdul:
            data = hdul[0].data

            print(f"Data shape: {data.shape}")  # Confirm shape

            # Plot
            plt.figure(figsize=(10, 5))
            plt.plot(wave, data, lw=0.5)
            plt.xlabel("Wavelength")
            plt.ylabel("Summed Flux")
            plt.title("Collapsed Spectrum Check")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def plot_check_2(self):
        # Open FITS file
        hdul = fits.open(self.file_paths['NIR'])
        header = hdul[0].header
        data = hdul[0].data

        print("Cube shape:", data.shape)

        # Extract wavelength axis info
        N = header["NAXIS3"]
        CRPIX = header["CRPIX3"]  # reference pixel
        CDELT = header["CDELT3"]  # increment per pixel
        CRVAL = header["CRVAL3"]  # reference value

        wave = (np.arange(N) - (CRPIX - 1)) * CDELT + CRVAL
        print("Wavelength range:", wave[0], "to", wave[-1])

        # Collapse cube spatially (sum over Y and X)
        spectrum = data.sum(axis=(1, 2))

        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(wave, spectrum, label='Summed Spectrum')
        plt.xlabel('Wavelength [Angstrom]')
        plt.ylabel('Total Flux')
        plt.title('Summed Spectrum from Cube')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def make_molecfit_ready(self, post_xpipe, DI_file):
        self.post_xpipe = post_xpipe
        self.DI_file = DI_file

        # Step 1: Read the DI file
        with fits.open(self.DI_file) as hdulist_DI:
            di_data = hdulist_DI[0].data  # shape ~ (28000,)

        # Step 2: Open post_xpipe file
        HDUlist_xpipe = fits.open(self.post_xpipe)
        nHDUlist = HDUlist_xpipe.copy()

        # Step 3: Get the shape of the flux HDU and overwrite the data
        flux_data = nHDUlist['FLUX'].data  # shape like (28000, 28, 3)
        shape = flux_data.shape
        print("The shape of the flux data from the post_xpipeline file is", shape)
        if di_data.shape[0] != shape[0]:
            raise ValueError(f"DI spectrum length {di_data.shape[0]} doesn't match flux cube length {shape[0]}.")

        new_flux = np.tile(di_data[:, np.newaxis, np.newaxis], (1, shape[1], shape[2]))
        nHDUlist['FLUX'].data = new_flux

        #Bonus step: Replace the Object card in the header
        hdr = nHDUlist[0].header
        if hdr.get('OBJECT', '').strip().upper() == 'STD, FLUX':
            hdr['OBJECT'] = self.object.upper()

        # Step 4: Write the output
        fname = self.post_xpipe.replace('.fits', '_MOLECFIT_READY.fits').replace("MERGE3D", "MERGE1D")
        nHDUlist.writeto(fname, output_verify="fix+warn", overwrite=True, checksum=True)
        HDUlist_xpipe.close()

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

    def final_check(self):

        fname = self.post_xpipe.replace('.fits', '_MOLECFIT_READY.fits').replace("MERGE3D", "MERGE1D")
        print(f"Checking {fname}:")
        with fits.open(fname) as hdul:
            header = hdul[0].header

            # Wavelength axis
            N = header["NAXIS1"]
            wave = np.zeros(N)
            for i in range(N):
                wave[i] = (i + 1 - header["CRPIX1"]) * header["CDELT1"] + header["CRVAL1"]

            # Data axis
            data = hdul[0].data
            print(f"Data shape: {data.shape}")  # Confirm shape

            print(f"Are any NaNs in the data: {np.isnan(data).any()}")
            print(f"Are any infs in the data: {np.isinf(data).any()}")
            print(f"min: {np.min(data)}, max: {np.max(data)}")

            # Plot
            plt.figure(figsize=(10, 5))
            plt.plot(wave, data, lw=0.5)
            plt.xlabel("Wavelength")
            plt.ylabel("Summed Flux")
            plt.title("Final Check")
            plt.grid(True)
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":

    object_list = ['feige-110']
    arm = ['UVB', 'VIS', 'NIR']

    for a in arm:
        if len(object_list) >= 2:
            prep = MolecfitPrep(object_list[0])

            print(f"Combining images for {a} arm")
            combined_spectrum = prep.combine_disk_integrated_spectra(object_list, a)
            fits.writeto(f"combined_DI_{a}.fits", combined_spectrum, overwrite=True)

            prep.plot_check(prep.file_paths[a], f"combined_DI_{a}.fits")

            print(f"Writing molecfit ready fits file for {a} arm")
            prep.make_molecfit_ready(prep.file_paths[a], f"combined_DI_{a}.fits")
            #prep.final_check()

        else:
            prep = MolecfitPrep(object_list[0])

            prep.plot_check(prep.file_paths[a], prep.file_paths[f'DI_{a}'])

            print(f"Writing molecfit ready fits file for {a} arm")
            prep.make_molecfit_ready(prep.file_paths[a], prep.file_paths[f'DI_{a}'])
            #prep.final_check()
