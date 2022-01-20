"""
This module provides the high-level event loop to calculate
the tau point source effective area.
"""
import pickle
from typing import Any, Dict, List, Tuple, Union

import attr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from tqdm import tqdm

import poinsseta.antenna as antenna
import poinsseta.decay as decay
import poinsseta.geometry as geometry
import poinsseta.tauola as tauola
from poinsseta.efield import EFieldParam
from poinsseta.tauexit import TauExitLUT


# we provide the results of the effective area calculation in a tuple
@attr.s
class EffectiveArea:
    # the total number of trials thrown for this result
    N0: np.ndarray = attr.ib()

    # the elevation angles at which the effective area is sampled
    elevation: np.ndarray = attr.ib()

    # the effective area in kilometers at each angle
    effective_area: np.ndarray = attr.ib()

    # the geometric area in kilometers at each angle
    geometric: np.ndarray = attr.ib()

    # the exit probability at each angle
    pexit: np.ndarray = attr.ib()

    # the decay probability at each angle
    pdecay: np.ndarray = attr.ib()

    # the trigger probability at each angle
    ptrigger: np.ndarray = attr.ib()

    # the arguments used to construct this effective area
    args: Dict[str, Any] = attr.ib()

    # allow two results be added
    def __add__(self, other: "EffectiveArea") -> "EffectiveArea":
        """
        Add two effective areas together. This implements
        the average of two effective areas.

        These two effective areas must have been
        sampled at the same elevation angles.

        Parameters
        ----------
        other: EffectiveArea
            Another effective area calculation.

        """

        # check that the angles are the same size
        if not self.elevation.size == other.elevation.size:
            raise ValueError(
                (
                    "Effective areas must have been "
                    "evaluated at the same elevation angles"
                )
            )

        # check that the angles are the same
        if not np.isclose(self.elevation, other.elevation).all():
            raise ValueError(
                (
                    "Effective areas must have been "
                    "evaluated at the same elevation angles"
                )
            )

        # and add the total number of trials
        N0 = self.N0 + other.N0

        # check if the args are the same
        if self.args != other.args:
            msg = "Effective areas generated with different arguments!"
            msg += f"self: \n{self.args}\n"
            msg += f"other: \n{other.args}\n"
            raise ValueError(msg)

        # and average all the quantities together
        effective_area = 0.5 * (self.effective_area + other.effective_area)
        geometric = 0.5 * (self.geometric + other.geometric)
        pexit = 0.5 * (self.pexit + other.pexit)
        pdecay = 0.5 * (self.pdecay + other.pdecay)
        ptrigger = 0.5 * (self.ptrigger + other.ptrigger)

        # and create a new EffectiveArea
        return EffectiveArea(
            N0,
            self.elevation,
            effective_area,
            geometric,
            pexit,
            pdecay,
            ptrigger,
            self.args,
        )

    def plot(self,) -> Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:

        # and let's create a test plot as we work
        fig, ax = plt.subplots(figsize=(8, 4))

        # and sample some colors from a colormap (avoid the last color)
        colors = plt.cm.inferno(np.linspace(0, 1.0, 5))

        # plot the geometric area
        ax.semilogy(
            self.elevation,
            self.geometric,
            label=r"$\langle A_g \rangle$",
            color=colors[1],
            lw=1.0,
        )

        # and the geometric area times the exit probability
        ax.semilogy(
            self.elevation,
            self.geometric * self.pexit,
            label=r"$\langle A_g \rangle\ \bar{P}_{\mathrm{exit}}$",
            color=colors[2],
            lw=1.0,
        )

        # and incorporate the decay probability
        ax.semilogy(
            self.elevation,
            self.geometric * self.pexit * self.pdecay,
            label=r"$\langle A_g\rangle\ \bar{P}_{\mathrm{exit}}"
            r" \bar{P}_{\mathrm{decay}}$",
            color=colors[3],
            lw=1.0,
        )

        # plot the factorized geometric area
        ax.semilogy(
            self.elevation,
            self.geometric * self.pexit * self.pdecay * self.ptrigger,
            label=(
                r"$\langle A_g  \rangle\ \bar{P}_{\mathrm{exit}} "
                r"\bar{P}_{\mathrm{decay}} \bar{P}_{\mathrm{trig}}$"
            ),
            color=colors[0],
            alpha=0.5,
            lw=1.0,
        )

        # and plot the true effective area
        ax.semilogy(
            self.elevation,
            self.effective_area,
            label=(
                r"$\langle A_g  P_{\mathrm{exit}} "
                r"P_{\mathrm{decay}} P_{\mathrm{trig}} \rangle$"
            ),
            color=colors[0],
            lw=1.0,
        )

        # and some labels
        ax.set(
            xlabel=r"Payload Elevation Angle [$^\circ$]",
            ylabel=r"Effective Area [km$^2$]",
            xlim=[self.elevation.min(), self.elevation.max()],
            ylim=[1e-12, 1e4],
        )

        # we want every order of magnitude
        ax.yaxis.set_major_locator(mtick.LogLocator(base=10.0, numticks=20))

        # enable the legend
        plt.legend()

        # and return the figures and the axes
        return fig, ax


def calculate(
    Enu: float,
    elev: np.ndarray,
    altitude: float = 3.87553,
    prototype: int = 2018,
    maxview: float = np.radians(3.0),
    icethickness: int = 0,
    N: Union[np.ndarray, int] = 1_000_000,
    antennas: int = 4,
    minfreq: float = 30,
    maxfreq: float = 80,
    trigger_SNR: float = 5.0,
    azimuths: np.ndarray = np.array([0]),
) -> EffectiveArea:
    """
    Calculate the effective area of BEACON to a point source
    tau flux.

    Parameters
    ----------
    Enu: float
        The energy of the neutrino that is incident.
    elev: np.ndarray
       The elevation angle (in radians) to calculate the effective area at.
    altitude: float
       The altitude of BEACON (in km) for payload angles.
    prototype: int
        The prototype number for this BEACON trial.
    maxview: float
        The maximum view angle (in radians).
    icethickness: int
        The thickness of the ice (in km).
        We currently support 0, 1, 2, 3, 4.
    N: Union[int, np.ndarray]
        The number of trials to use for geometric area.
    antennas: int
        The number of antennas.
    minfreq: float
        The minimum frequency (in MHz).
    maxfreq: float
        The maximum frequency (in MHz).
    trigger_SNR: float
        The SNR threshold for a trigger.
    azimuths: np.ndarray
        The azimuths (in degrees) to calculate the gain at.

    Returns
    -------
    Aeff: EffectiveArea
        A collection of effective area components across elevation.
    """

    # we make sure that elevation is at least an array
    elev = np.atleast_1d(elev)

    # make N an array if it's not already
    if not isinstance(N, np.ndarray):
        N = N * np.ones(elev.size, dtype=int)

    # load the corresponding tau exit LUT
    tauexit = TauExitLUT(energy=Enu, thickness=icethickness)

    # load the field parameterization.
    altitudes = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 37.0])
    i_altitude = np.abs(altitudes - altitude).argmin()
    altitude_file = altitudes[i_altitude]

    efield_filename = "interpolator_efields_" + str(altitude_file) + "km"
    voltage = EFieldParam(filename=efield_filename)

    # arrays to store the output of the effective area at each elevation
    effective_area = np.zeros((elev.size, azimuths.size))
    geometric = np.zeros_like(elev)
    pexit = np.zeros_like(elev)
    pdecay = np.zeros_like(elev)
    ptrigger = np.zeros_like(elev)

    # the frequencies over which we calculate field quantities
    freqs = np.arange(minfreq, maxfreq, 10.0)
    # the central frequencies of each 10 MHz sub-band
    center_freqs = np.arange(minfreq + 5, maxfreq, 10.0)

    # calculate the integrated noise voltage across the band
    Vn_spectrum = antenna.noise_voltage(center_freqs, prototype, antennas).reshape(
        (-1, 1)
    )

    # loop over each elevation angle
    for i in tqdm(np.arange(elev.shape[0])):

        # compute the geometric area at the desired elevation angles
        Ag = geometry.geometric_area(
            altitude, maxview, elev[i], 0, N=N[i], ice=icethickness
        )

        # if we didn't get any passing events, just skip this trial
        if Ag.emergence.size == 0:
            continue

        # get the exit probability at these elevation angles
        # this is a masked array and will be masked
        # if no tau's exitted at these angles
        Pexit, Etau = tauexit(90.0 - np.degrees(Ag.emergence))

        # get a random set of decay lengths at these energies
        decay_length = tauola.sample_range(Etau)

        # we now need the decay probability
        Pdecay = decay.probability(decay_length, Ag.dbeacon)

        # calculate the view angle from the decay point
        view = geometry.decay_view(Ag.view, Ag.dbeacon, decay_length)

        # and the sample the energy of the tau's
        Eshower = tauola.sample_tau_energies(Etau, N=Pdecay.size)
        intensity = (Eshower/1e9)*(np.exp(-7.7-0.39*view))
        #*(np.exp(-12-0.5*view))
       #(np.exp(-7.6-0.4*view)) 
        Ptrig = intensity>0.0000000024
        # get the zenith angle at the decay
        # decay_zenith = geometry.decay_zenith(Ag.emergence, decay_length)

        # and get the altitude at the decay point
        decay_altitude = geometry.decay_altitude(
            Ag.emergence, decay_length, icethickness
        )

        # get the zenith angle at the exit point
        exit_zenith = (np.pi / 2.0) - Ag.emergence

        wrapped_azimuths = azimuths % 360

        gain = antenna.directivity(prototype, wrapped_azimuths)

        # compute the voltage at each of these off-axis angles and at each frequency
        #volt = voltage(
            #np.degrees(view),
            #np.degrees(exit_zenith),
            #decay_altitude,
            #freqs,
            #Eshower,
            #altitude,
            #gain,
            #antennas,
        #)

        # calculate the SNR
        #SNR = np.sum(volt, axis=0) / np.sqrt(np.sum(Vn_spectrum ** 2.0))

        # throw a random rician for the realized SNR for each trial
        #SNR_realized = np.sqrt(
            #np.random.normal(loc=SNR, scale=1, size=SNR.shape) ** 2.0
           # + np.random.normal(loc=0, scale=1.0, size=SNR.shape) ** 2.0
        #)

        # and check for a trigger
        #Ptrig = SNR_realized > trigger_SNR

        # we now apply some cuts to determine if there are
        # events that would not have been seen in the current
        # analysis. Primarily, we cut events that are seen as
        # above-horizon events or that are geometrically hidden
        # by the physical horizon

        # this is the location of the decay point
        decay_point = Ag.trials + decay_length.reshape((-1, 1)) * Ag.axis

        # the vector from BEACON to the decay point
        v = decay_point - Ag.beacon.reshape((1, -1))

        # calculate the normalized BEACON location vector
        beacon = Ag.beacon / np.linalg.norm(Ag.beacon)

        # now compute the dot product between BEACON's zenith and the view
        # vector to the decay point
        viewdot = np.einsum(
            "ij,ij->i",
            beacon.reshape((1, -1)),
            v / np.linalg.norm(v, axis=1).reshape((-1, 1)),
        )

        # and use this to compute the angle below BEACON's horizontal
        theta = np.pi / 2.0 - np.arccos(viewdot)

        # the distance from BEACON to the decay point
        D = np.linalg.norm(v, axis=1)

        # calculate the distance (km) to the horizon from BEACON
        horizon_distance = geometry.distance_to_horizon(
            height=altitude, thickness=icethickness
        )

        # the decay points that are further away than the horizon
        beyond = D > horizon_distance
        del horizon_distance

        # and the particles that appear to be below the horizon
        # remember: more negative is below the horizon
        below = theta < geometry.horizon_angle(altitude, icethickness)

        # those that are beyond the horizon and below the horizon
        invisible = np.logical_and(beyond, below)
        del beyond, below

        # if the trial is invisible, there's no way we can trigger on it
        Ptrig[invisible] = 0.0

        # if the event is above ANITA's horizon, we would not find
        # them in the search as they would be treated as background
        Ptrig[theta > 0.0] = 0.0

        # the number of trials that we used in this iteration
        ntrials = float(N[i])

        # and save the various effective area coefficients at these angles
        geometric[i] = (Ag.area * np.sum(Ag.dot)) / ntrials
        pexit[i] = np.mean(Pexit)
        pdecay[i] = np.mean(Pdecay)
        ptrigger[i] = np.mean(Ptrig)
        effective_area[i, :] = (
            np.sum((Ag.area * Ag.dot * Pexit * Pdecay) * Ptrig, axis=0)
            / ntrials
        )

    # construct a dictionary of the arguments
    args = {
        "Enu": Enu,
        "altitude": altitude,
        "prototype": prototype,
        "maxview": maxview,
        "icethickness": icethickness,
        "antennas": antennas,
        "gain": gain,
        "trigger_SNR": trigger_SNR,
        "minfreq": minfreq,
        "maxfreq": maxfreq,
    }
    # and now return the computed parameters
    return EffectiveArea(
        N, np.degrees(elev), effective_area, geometric, pexit, pdecay, ptrigger, args,
    )


def from_file(filename: str) -> EffectiveArea:
    """
    Load an effective area result from a file.
    Parameters
    ----------
    filename: str
        The filename containing a pickled EffectiveArea
    Returns
    -------
    Aeff: EffectiveArea
        The loaded effective area.
    """

    with open(filename, "rb") as f:
        return pickle.load(f)


def from_files(filenames: List[str]) -> EffectiveArea:
    """
    Load and combine effective area result from multiple files
    Parameters
    ----------
    filename: str
        The filename containing a pickled EffectiveArea
    Returns
    -------
    Aeff: EffectiveArea
        The loaded effective area.
    """

    # if we don't get any files, report an error
    if len(filenames) == 0:
        raise ValueError("No filenames were given to `from_files`.")

    # load the first file
    Aeff = from_file(filenames[0])

    # and load the rest of the files
    for f in filenames[1:]:
        A = from_file(f)
        if A.args["altitude"] == 3.87553:
            continue
        Aeff += A

    # and return the combined effective area
    return Aeff


# this lets us load pickled files from older poinsseta versions.
AcceptanceResult = EffectiveArea
