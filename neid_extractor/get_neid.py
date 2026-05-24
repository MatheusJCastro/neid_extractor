#!/usr/bin/env python3
"""
| NEID Extractor
| Matheus J. Castro

| Get NEID level 2 spectra from FITS files and save them as plain text CSV.
"""

import numpy as np
import matplotlib.pyplot as plt
# from PyAstronomy import pyasl
from astropy.io import fits
from pathlib import Path


class NeidExtractor:
    """
    NeidExtractor main class. Handles one spectrum file and all of its operations.
    """

    def __init__(self, fl: str | Path):
        """
        Init function.

        :param fl: Spectrum file to open.
        """

        self.file = Path(fl)

        self.fits = None
        self.wave_hdu = None
        self.flux_hdu = None
        self.wavelength = None
        self.flux = None
        self.order = 0

        self.open_fits()

    def __getitem__(self, item):
        """
        Get a single HDU object from fits.

        :param item: The HDU index.
        :return: The HDU content.
        """
        return self.fits[item]

    def __len__(self):
        """
        Get the length of the fits HDU.

        :return: The length (int).
        """

        return len(self.fits)

    def open_fits(self):
        """
        Open a fits file using astropy.
        """

        self.fits = fits.open(self.file)

    def info(self):
        """
        Print information about the fits file.
        """

        print(self.fits.info())

    def search_hdus(self, **kwargs: str):
        """
        Search for strings in the name of each HDU.

        :param kwargs: Strings to search.
        """

        if len(kwargs) == 0:
            print("At least one object name is needed.")

        match_hdu = {}
        for args in kwargs.items():
            try:
                match_hdu[args[0]] = self.fits.index_of(args[1])
            except KeyError:
                print("Parameter {} with name {} not found in file.".format(args[0], args[1]))

        return match_hdu

    def check_header(self, hdu: int):
        """
        Print the header of a single HDU.

        :param hdu: HDU to print.
        """

        header = self.fits[hdu].header
        print(header.tostring(sep="\n"))

    def find_in_header(self, hdu: int, value: str):
        """
        Search for a string in a header.

        :param hdu: HDU to search.
        :param value: String to look for.
        """

        header = self.fits[hdu].header
        header = header.tostring(sep="\n")

        if value not in header:
            print("Not found.")
        else:
            for line in header.split("\n"):
                if value in line:
                    print(line)

    def search_hdu_header(self, value: str):
        """
        Search for a string in all available HDU Headers of the fits.
        It prints True or False for each HDU.

        :param value: String to look for.
        """

        print("Search string {} returned:".format(value))
        for i,block in enumerate(self.fits):
            header = block.header.tostring(sep="\n")
            print(value in header, "for HDU={}".format(i))

    def build_wave(self, hdu: int, naxis: str = "NAXIS1",
                   crval: str = "CRVAL1", cdelt: str = "CDELT1"):
        """
        Build a wavelength solution based on header values.

        :param hdu: HDU to use.
        :param naxis: String for NAXIS: Length of the wavelength list.
        :param crval: String for CRVAL: Initial value of the list.
        :param cdelt: String for CDELT: Step of each value.
        """

        self.wavelength = (self.fits[hdu].header[crval] +
                           self.fits[hdu].header[cdelt] *
                           np.arange(0, self.fits[hdu].header[naxis]))

    def assign_wave(self, hdu: int):
        """
        Assign an HDU data to use as wavelength.

        :param hdu: HDU to use.
        """

        self.wavelength = self.fits[hdu].data[self.order]

    def assign_flux(self, hdu: int):
        """
        Assign an HDU data to use as flux.

        :param hdu: HDU to use.
        """

        self.flux = self.fits[hdu].data[self.order]

    def auto_make_spectrum(self, wave_hdu: str = "SCIWAVE",
                           flux_hdu: str = "SCIFLUX",
                           build_wave: bool = False,
                           build_wave_hdu: int = None,
                           naxis: str = "NAXIS1",
                           crval: str = "CRVAL1",
                           cdelt: str = "CDELT1"):
        """
        Automate the process of creating the wavelength and flux solutions based on regular values.

        :param wave_hdu: Name of the wavelength HDU.
        :param flux_hdu: Name of the flux HDU.
        :param build_wave: To use or not the Header values to create wavelength solution.
        :param build_wave_hdu: HDU to use for build_wave.
        :param naxis: String for NAXIS: Length of the wavelength list.
        :param crval: String for CRVAL: Initial value of the list.
        :param cdelt: String for CDELT: Step of each value.
        """

        hdus_to_use = self.search_hdus(whdu=wave_hdu, fhdu=flux_hdu)

        self.assign_flux(hdus_to_use["fhdu"])

        if not build_wave:
            self.assign_wave(hdus_to_use["whdu"])
        elif build_wave_hdu is not None:
            self.build_wave(build_wave_hdu, naxis, crval, cdelt)
        else:
            exit("build_wave_hdu is necessary for build_wave=True")

    def plot_spec(self, fl_name: str | Path = None, plot: bool = False, save: bool = True):
        """
        Plot the spectrum for an order.

        :param fl_name: Name to save. If not passed as an argument, the function automatically determines the name
        and save it in the same directory of the spectrum.
        :param plot: whether to plot or not.
        :param save: whether to save or not.
        """

        if fl_name is None:
            fl_name = Path(self.file).parent
            fl_name = fl_name.joinpath("Plot_" + str(Path(self.file).stem) + "_order{}.pdf".format(self.order))

        plt.figure(figsize=(16, 9))

        plt.title(str(Path(fl_name.stem)), fontsize=22)
        plt.xlabel("Wavelength", fontsize=18)
        plt.ylabel("Flux", fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.xlim(min(self.wavelength), max(self.wavelength))

        plt.plot(self.wavelength, self.flux)

        plt.grid()
        plt.tight_layout()

        if save:
            plt.savefig(fl_name)
        if plot:
            plt.show()
        plt.close()

    def save_spec(self, fl_name: str | Path = None, delimiter: str = ",", fmt: str = "%.5f"):
        """
        Save the spectrum as an ASCII file for an order.

        :param fl_name: Name to save. If not passed as an argument, the function automatically determines the name
        and save it in the same directory of the spectrum.
        :param delimiter: delimiter to use.
        :param fmt: format of data to use.
        """

        if fl_name is None:
            fl_name = Path(self.file).parent
            fl_name = fl_name.joinpath(str(Path(self.file).stem) + "_order{}.csv".format(self.order))

        data = np.array([self.wavelength, self.flux]).T
        np.savetxt(fl_name, data,
                   header="wave,flux", delimiter=delimiter, fmt=fmt)


class MultipleNeid:
    """
    Open and handle multiple spectrum files at once.
    """

    def __init__(self, directory: str | Path):
        """
        Init function.

        :param directory: The directory where the spectra are saved.
        """

        self.dir = Path(directory)

        self.files = list(self.dir.glob("*.fits"))
        self.order = 0

        if len(self.files) == 0:
            exit("No fits files found in {}".format(self.dir))

        self.multi_data = [NeidExtractor(file) for file in self.files]

    def __getitem__(self, item):
        """
        Get one spectrum object.

        :param item: The index to get.
        :return: The spectrum object.
        """

        return self.multi_data[item]

    def __len__(self):
        """
        Get the number of opened files.

        :return: The number of opened files
        """

        return len(self.multi_data)

    def info(self):
        """
        Print info of all files.
        """

        for obj in self.multi_data:
            obj.info()

    def _apply_order(self):
        """
        Apply the desired order for all spectra.
        """

        for obj in self.multi_data:
            obj.order = self.order

    def auto_make_all(self, wave_hdu: str = "SCIWAVE",
                           flux_hdu: str = "SCIFLUX",
                           build_wave: bool = False,
                           build_wave_hdu: int = None,
                           naxis: str = "NAXIS1",
                           crval: str = "CRVAL1",
                           cdelt: str = "CDELT1"):
        """
        Automate the wavelength and flux solutions for all spectrum files.

        :param wave_hdu: Name of the wavelength HDU.
        :param flux_hdu: Name of the flux HDU.
        :param build_wave: To use or not the Header values to create wavelength solution.
        :param build_wave_hdu: HDU to use for build_wave.
        :param naxis: String for NAXIS: Length of the wavelength list.
        :param crval: String for CRVAL: Initial value of the list.
        :param cdelt: String for CDELT: Step of each value.
        """

        self._apply_order()
        for obj in self.multi_data:
            obj.auto_make_spectrum(wave_hdu=wave_hdu,
                                   flux_hdu=flux_hdu,
                                   build_wave=build_wave,
                                   build_wave_hdu=build_wave_hdu,
                                   naxis=naxis,
                                   crval=crval,
                                   cdelt=cdelt)

    def plot_all(self, fl_name: str | Path = None, plot: bool = False, save: bool = True):
        """
        Plot all spectra for an order.

        :param fl_name: Check plot_spec in NeidExtractor.
        :param plot: Check plot_spec in NeidExtractor.
        :param save: Check plot_spec in NeidExtractor.
        """

        for obj in self.multi_data:
            obj.plot_spec(fl_name, plot=plot, save=save)

    def save_all(self, fl_name: str | Path = None, delimiter: str = ",", fmt: str = "%.5f"):
        """
        Save all spectra in ASCII for an order.

        :param fl_name: Check save_spec in NeidExtractor.
        :param delimiter: Check save_spec in NeidExtractor.
        :param fmt: Check save_spec in NeidExtractor.
        """

        for obj in self.multi_data:
            obj.save_spec(fl_name=fl_name, delimiter=delimiter, fmt=fmt)
