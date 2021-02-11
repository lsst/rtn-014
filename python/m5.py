#!/usr/bin/env python
"""Pre-calculate a grid of 5-sigma limiting mags for all times, filters, and DDF fields
"""

# imports
import sys
import warnings
import logging
from argparse import ArgumentParser

import yaml
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.coordinates
from astropy.time import Time

import lsst.sims.utils
from lsst.sims.downtimeModel import ScheduledDowntimeData
from lsst.sims.skybrightness_pre import SkyModelPre
from lsst.sims.seeingModel import SeeingModel

# constants

DEFAULT_CONFIG = {
    "start_time": "2022-11-01T16:00:00Z",
    "end_time": "2033-11-01T16:00:00Z",
    "time_freq": "10min",
    "site_name": "LSST",
    "fields": {
        "Elias S1": SkyCoord("00h37m48s", "-44d00m00s"),
        "XMM-LSS": SkyCoord("02h22m50s", "-04d45m00s"),
        "ECDFS": SkyCoord("03h32m30s", "-28d06m00s"),
        "COSMOS": SkyCoord("10h00m24s", "+02d10m55s"),
        "Euclid 1": SkyCoord("03h55m52.8s", "-49d16m48s"),
        "Euclid 2": SkyCoord("04h14m24s", "-47d36m00s"),
    },
    "max_sun_alt_deg": -18.0,
    "max_field_airmass": 2.6,
}

SKY_MODEL = SkyModelPre()

# exception classes

# interface functions


def calc_m5(config=None):
    """Calculate the 5-sigma limiting magnitudes for a set of fields

    Parameters
    ----------
    config : `dict'
        Configuration parameters, with the following contents:
        start_time : `str` or period-like, default None
            Start of time range
        end_time : `str` or period-like, default None
            End of time range
        time_freq : `str`
            Time frequency, designated by a pandas frequency string
        fields : `dict` [`str`, ~astropy.coordinates.SkyCoord`]
            The pointings for which to calculate limiting mags.
        site_name : `str`
            The name of the observatory site.
        max_sun_alt_deg: `float`
            Maximum altitude of sun for observing time, in degrees
        max_field_airmass: `float`
            Maximum airmass at which to observe

    Returns
    -------
    m5limits : `pandas.DataFrame`
        A `pandas.Series` of the 5 sigma limiting magnitudes, indexed by the `mjd` and field name.

    Notes
    -----

    For more on pandas frequency strings, see `the pandas documentation
<https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    """

    config = DEFAULT_CONFIG if config is None else config

    # Work around bug in lsst.sims.utils.Site, which compares by reference
    # rather than equivilence
    site_name = "LSST" if config["site_name"] == "LSST" else config["site_name"]
    site = lsst.sims.utils.Site(site_name)
    location = astropy.coordinates.EarthLocation(
        lat=site.latitude, lon=site.longitude, height=site.height
    )

    periods = _init_periods(config, location)

    field_coords = _init_field_coords(config)

    logger.debug("Building the sampled_fields DataFrame")
    # Cross join using the hacky pandas method
    periods["dummy"] = True
    field_coords["dummy"] = True
    sampled_fields = (
        periods.merge(field_coords, on="dummy", how="outer")
        .drop(columns=["dummy"])
        .set_index(["field_name", "period"], drop=False)
    )
    sampled_fields.query("observing", inplace=True)

    logger.debug("Calculating field airmass and filtering")
    field_zd_rad = np.pi / 2 - (
        SkyCoord(
            ra=sampled_fields.field_ra,
            dec=sampled_fields.field_decl,
            frame="icrs",
            unit="deg",
        )
        .transform_to(
            astropy.coordinates.AltAz(
                location=location, obstime=Time(sampled_fields["mjd"], format="mjd")
            )
        )
        .alt.rad
    )
    sampled_fields["field_airmass"] = 1.0 / np.cos(field_zd_rad)
    max_field_airmass = config["max_field_airmass"]
    sampled_fields.query(f"field_airmass < {max_field_airmass}", inplace=True)

    logger.debug("Calculating field angle with the moon")
    sampled_fields["moon_angle"] = (
        SkyCoord(
            ra=sampled_fields["field_ra"],
            dec=sampled_fields["field_decl"],
            frame="icrs",
            unit="deg",
        ).separation(
            SkyCoord(
                ra=sampled_fields["moon_ra"],
                dec=sampled_fields["moon_decl"],
                frame="icrs",
                unit="deg",
            )
        )
    ).deg

    logger.debug("Calculating sky brightness")
    sampled_fields = sampled_fields.groupby("mjd").apply(_get_sky_mags)

    sampled_fields = sampled_fields.groupby("band").apply(_compute_band_fwhms)

    sampled_fields = sampled_fields.groupby("band").apply(_compute_band_m5)

    sampled_fields.set_index(["field_name", "period"], drop=False, inplace=True)

    logger.debug("Finished constructing the m5 DataFrame")
    return sampled_fields


# classes

# internal functions & classes


def _init_periods(config, location):

    logger.debug("Laying down time period boundries")
    period_index = pd.period_range(
        config["start_time"],
        config["end_time"],
        freq=config["time_freq"],
        name="period",
    )
    times = Time(
        pd.to_datetime(period_index.to_timestamp()),  # pylint: disable=no-member
        location=location,
    )

    logger.debug("Calculating solar ephemeris")
    sun_coords = astropy.coordinates.get_sun(times)

    logger.debug("Calculating lunar ephemeris")
    moon_coords = astropy.coordinates.get_moon(times)
    moon_elongation = moon_coords.separation(sun_coords)

    logger.debug("Converting solar and lunar coordinates to horizon system")
    horizon_coordinate_system = astropy.coordinates.AltAz(location=location)
    sun_hzn = sun_coords.transform_to(horizon_coordinate_system)
    moon_hzn = moon_coords.transform_to(horizon_coordinate_system)

    logger.debug("Building the period DataFrame")
    periods = pd.DataFrame(
        {
            "mjd": times.mjd,
            "time": times,
            "lst": times.sidereal_time("mean"),
            "sun_ra": sun_coords.ra.deg,
            "sun_decl": sun_coords.dec.deg,
            "sun_alt": sun_hzn.alt.deg,
            "moon_ra": moon_coords.ra.deg,
            "moon_decl": moon_coords.dec.deg,
            "moon_alt": moon_hzn.alt.deg,
            "moon_elongation": moon_elongation.deg,
        },
        index=period_index,
    )

    periods["moon_waxing"] = (
        periods["moon_elongation"].shift(-1) > periods["moon_elongation"]
    )

    previous_waxing = periods["moon_waxing"].shift(
        1, fill_value=periods.iloc[0]["moon_waxing"]
    )
    periods["new_moon"] = periods["moon_waxing"] & ~previous_waxing
    periods["lunation"] = periods["new_moon"].cumsum()

    mean_local_solar_jd = 2400000.5 + periods["mjd"] + (location.lon.deg / 360.0)
    night_local_solar_jd = np.floor(mean_local_solar_jd).astype(int)
    periods["night"] = night_local_solar_jd - np.min(night_local_solar_jd) + 1

    # Mark daytime periods
    periods["observing"] = periods.sun_alt <= config["max_sun_alt_deg"]

    # Mark scheduled downtime
    periods.reset_index(inplace=True)
    periods.set_index("mjd", inplace=True)
    for down_time in ScheduledDowntimeData(times.min())():
        periods.loc[down_time["start"].mjd : down_time["end"].mjd, "observing"] = False
    periods.reset_index(inplace=True)
    periods.set_index("period", drop=False, inplace=True)
    return periods


def _init_field_coords(config):
    fields = config["fields"]

    logger.debug("Building the fields DataFrame")
    field_coords = (
        pd.DataFrame(
            {
                fld: {
                    "field_name": fld,
                    "field_ra": fields[fld].ra.deg,
                    "field_decl": fields[fld].dec.deg,
                }
                for fld in fields
            }
        )
        .T.set_index("field_name")
        .apply(pd.to_numeric)
        .reset_index()
    )

    logger.debug("Adding healpix to field coordinates")
    field_coords["field_hpix32"] = lsst.sims.utils.raDec2Hpid(
        32, field_coords.field_ra, field_coords.field_decl
    )

    return field_coords


def _get_sky_mags(sampled_fields_mjd):
    mjd = sampled_fields_mjd.mjd[0]
    assert np.all(mjd == sampled_fields_mjd["mjd"])

    mags = SKY_MODEL.returnMags(mjd, sampled_fields_mjd.field_hpix32, badval=-np.inf)
    sampled_fields_band_mjds = []
    for band in mags:
        sampled_fields_band_mjd = sampled_fields_mjd.copy()
        sampled_fields_band_mjd["band"] = band
        sampled_fields_band_mjd["sky_mag"] = mags[band]
        sampled_fields_band_mjds.append(sampled_fields_band_mjd)
    out_sampled_fields_mjd = pd.concat(sampled_fields_band_mjds)
    return out_sampled_fields_mjd


def _compute_band_fwhms(sampled_fields_band):
    band = sampled_fields_band["band"][0]
    assert np.all(band == sampled_fields_band["band"])

    logger.debug("Calculating the model seeing in %s", band)
    seeing_model = SeeingModel()

    band_idx = ["u", "g", "r", "i", "z", "y"].index(band)
    sampled_fields_band = sampled_fields_band.copy()
    sampled_fields_band["fwhm"] = seeing_model(
        0.7, sampled_fields_band["field_airmass"].values
    )["fwhmEff"][band_idx]
    return sampled_fields_band


def _compute_band_m5(sampled_fields_band):
    band = sampled_fields_band["band"][0]
    assert np.all(band == sampled_fields_band["band"])

    logger.debug("Calculating the 5-sigma magnitude limit in %s", band)
    sampled_fields_band = sampled_fields_band.copy()
    sampled_fields_band["m5"] = lsst.sims.utils.m5_flat_sed(
        band,
        sampled_fields_band["sky_mag"],
        sampled_fields_band["fwhm"],
        30.0,
        sampled_fields_band["field_airmass"],
        nexp=1.0,
    )
    return sampled_fields_band


def read_config(fname):
    """Read m5 configuration file

    Parameters
    ----------
    fname: `str`
        The name of the file to read configuration from.

    Return
    ------
    config: `dict`
        Dictionary of configuration values
    """
    logger.debug("Reading configuration from %s", fname)

    with open(fname, "r") as config_file:
        config = yaml.load(config_file.read(), Loader=yaml.FullLoader)

    # Convert field coordinates into astroy.SkyCoord objects
    for field in config["fields"]:
        config["fields"][field] = SkyCoord(**config["fields"][field])

    return config


def main():
    """Parse command line arguments and config file, and run"""
    parser = ArgumentParser()
    parser.add_argument("config", help="configuration file")
    parser.add_argument("output", help="file in which to write results")

    args = parser.parse_args()
    config_fname = args.config
    output_fname = args.output

    warnings.filterwarnings(
        "ignore",
        message=".*Tried to .* for times after IERS data is valid.*",
        category=astropy.utils.exceptions.AstropyWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*dubious year.*",
        category=astropy.utils.exceptions.AstropyWarning,
    )

    config = read_config(config_fname)

    m5_limits = calc_m5(config)

    (
        m5_limits.set_index(["field_name", "mjd"])
        .drop(columns=["period"])
        .to_hdf(output_fname, "m5")
    )

    return 0


def _init_logger(log_level=logging.DEBUG):
    """Create the ddfpresched logger and set initial configuration"""
    ddfpresched_logger = logging.getLogger("ddfpresched")
    ddfpresched_logger.setLevel(log_level)
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    handler.setFormatter(formatter)
    ddfpresched_logger.addHandler(handler)
    return ddfpresched_logger


if __name__ == "__main__":
    logger = _init_logger()
    status = main()  # pylint: disable=invalid-name
    sys.exit(status)
