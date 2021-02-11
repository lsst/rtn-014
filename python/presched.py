#!/usr/bin/env python
"""Pre-schedule DDF sequences
"""
# pylint: disable=no-member

# imports
import sys
import logging
from argparse import ArgumentParser

import yaml
import numpy as np
import pandas as pd
import astropy.coordinates
import astropy.units as u

import lsst.sims.utils

# constants

# exception classes

# interface functions


def schedule_all(mag_limit, location, config):
    """Schedule one field on one band.

    Parameters
    ----------
    m5 : `pandas.DataFrame`
        Has a multilevel index with the following levels:
        field_name : `str`
            the field name
        band : `str`
            the band

        Includes the following columns:
        mjd : `float`
             MJD of candidate time
        m5 : `float`
             5-sigma limiting magnitude of the field if observed at that time
    `location` : `astropy.coordinates.EarthLocation`
        the location of the observatory
    config : `dict`
        Configuration parameters

    Return
    ------
    schedule : `pandas.DataFrame`
        includes three columns:
        mjd : `float`
            the best time at which to start the sequence of exposures
        why : `str`
            an indicator of why this sequence was scheduled
        night : `int`
            the MJD of the night (at midnight) on which the sequence is to be scheduled
        sequence : `str`
            which sequence this is

    """
    seq_schedules = []

    for seq_config in config["sequences"]:
        logger.info(f'Scheduling {seq_config["label"]}')
        seq_schedule = schedule_sequence(mag_limit, location, seq_config)
        seq_schedule["sequence"] = seq_config["label"]
        logger.info(f'Computing scheduled for {seq_config["label"]}')
        mag_limit["scheduled"] = _compute_scheduled(
            mag_limit, seq_schedule, seq_config["sequence_duration"]
        )
        seq_schedules.append(seq_schedule)

    logger.info("Compiling full schedule")
    full_schedule = (
        pd.concat(seq_schedules).sort_values("mjd").set_index("mjd", drop=False)
    )
    return full_schedule


def schedule_sequence(mag_limit, location, config):
    """Schedule one set of sequences.

    Parameters
    ----------
    m5 : `pandas.DataFrame`
        Has a multilevel index with the following levels:
        field_name : `str`
            the field name
        band : `str`
            the band

        Includes the following columns:
        mjd : `float`
             MJD of candidate time
        m5 : `float`
             5-sigma limiting magnitude of the field if observed at that time
    `location` : `astropy.coordinates.EarthLocation`
        the location of the observatory
    config : `dict`
        Configuration parameters, with the following contents:
        field_name : `str`
            the name of the field to schedule
        mag_lim_band : `str`
            the name of the filter to schedule
        sequence_duration : `astropy.units.Quantity`
            the duration of a block of one sequence of exposures
        caninocal_gap : `astropy.units.Quantity`
            the desired time between sequences of exposures
        min_gap: `astropy.units.Quantity`
            the minimum gap for which "bridge" exposures should be scheduled
        max_gap: `astropy.units.Quantity`
            the target maximum time between sequences of exposures
        season_gap :  `astropy.units.Quantity`
            the gap time greater than which no bridges should be attempted
        mag_limit : `dict` of `str`: `float`
            target magnitude limits in each band

    Return
    ------
    schedule : `pandas.DataFrame`
        includes three columns:
        mjd : `float`
            the best time at which to start the sequence of exposures
        why : `str`
            an indicator of why this sequence was scheduled
        night : `int`
            the MJD of the night (at midnight) on which the sequence is to be scheduled

    """
    # pylint: disable=too-many-locals
    these_m5 = (
        mag_limit.sort_index()
        .loc[(config["field_name"], config["mag_lim_band"])]
        .sort_index()
        .copy()
    )

    min_m5 = _compute_rolling_m5(these_m5, config["sequence_duration"]).set_index(
        "mjd", drop=False
    )

    min_m5["night_mjd"] = compute_night_mjd(min_m5["mjd"], location)

    bridge_nights = _find_bridge_nights(mag_limit, location, config)
    bridge_gap = config["bridge_gap"]
    maintain_cadence = config["maintain_cadence_in_gap"]

    scheduled_sequences = []
    for night_mjd in range(min_m5.night_mjd.min(), min_m5.night_mjd.max()):
        if night_mjd in bridge_nights["night_before_mjd"].values:
            why = "pregap"
            attempt_tonight = True
            force_tonight = True
        elif bridge_gap and (night_mjd in bridge_nights["bridge_night_mjd"].values):
            why = "bridge"
            attempt_tonight = True
            force_tonight = True
        elif night_mjd in bridge_nights["night_after_mjd"].values:
            why = "postgap"
            attempt_tonight = True
            force_tonight = True
        elif len(scheduled_sequences) == 0:
            # We are just starting
            why = "start"

            attempt_tonight = True
            force_tonight = False
        elif (night_mjd - scheduled_sequences[-1]["night_mjd"]) * u.day >= config[
            "canonical_gap"
        ]:
            why = "cadence"
            attempt_tonight = True
            force_tonight = maintain_cadence
        else:
            continue

        if not attempt_tonight:
            continue

        candidate_times = min_m5.query(f"night_mjd == {night_mjd}")
        if len(candidate_times) < 1:
            assert maintain_cadence or not force_tonight
            continue

        best_time = min_m5.loc[candidate_times["m5"].idxmax()]
        if isinstance(best_time, pd.DataFrame):
            best_time = best_time.sort_values("count", ascending=True).iloc[-1]
        if (not force_tonight) and (best_time.m5 < config["mag_limit"]):
            continue

        if best_time.m5 < config["gap_mag_limit"]:
            continue

        scheduled_sequences.append({"mjd": best_time.mjd, "why": why})
        scheduled_sequences[-1]["night_mjd"] = compute_night_mjd(
            best_time.mjd, location
        )

    schedule = pd.DataFrame(scheduled_sequences)

    return schedule


def compute_night_mjd(mjd, location):
    """Convert the floating point mjd to the integer local Julian date for the night.

    Parameters
    ----------
    mjd : `float`, `pandas.Series`, or `numpy.ndarray`

    Returns
    -------
    jd : `int`, `pandas.Series`, or `numpy.ndarray`
    """
    # add longitude to get into the local timezone,
    # round to find the nearest midnight
    night_mjd = np.round(mjd + (location.lon.deg / 360.0)).astype(int)
    return night_mjd


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

    # Apply units
    for seq_config in config["sequences"]:
        seq_config["sequence_duration"] = u.Quantity(
            seq_config["sequence_duration"]
        ).to(u.second)
        seq_config["max_gap"] = u.Quantity(seq_config["max_gap"]).to(u.day)
        seq_config["min_gap"] = u.Quantity(seq_config["min_gap"]).to(u.day)
        seq_config["season_gap"] = u.Quantity(seq_config["season_gap"]).to(u.day)
        seq_config["canonical_gap"] = u.Quantity(seq_config["canonical_gap"]).to(u.day)

    site_name = "LSST" if config["site_name"] == "LSST" else config["site_name"]
    site = lsst.sims.utils.Site(site_name)
    config["location"] = astropy.coordinates.EarthLocation(
        lat=site.latitude, lon=site.longitude, height=site.height
    )

    return config


# classes

# internal functions & classes


def _infer_time_sampling(mag_limit):
    mjds = pd.Series(mag_limit["mjd"].unique()).sort_values()
    timestep_duration = ((mjds - mjds.shift(1)).median() * u.day).to(u.minute)
    return timestep_duration


def _compute_rolling_m5(mag_limit, roll_window):
    mag_limit = mag_limit.query("not scheduled").copy().sort_index()
    mag_limit["datetime"] = pd.to_datetime(
        mag_limit.mjd + 2400000.5, origin="julian", unit="D"
    )
    mag_limit["counter"] = 1
    mag_limit.set_index("datetime", inplace=True, drop=False)
    roll_seconds = roll_window.to("second").value
    mag_limit_roll = mag_limit.rolling(f"{int(roll_seconds)}s")
    min_mag_limit = mag_limit_roll[["mjd", "moon_angle", "night", "m5"]].min()
    min_mag_limit["start_datetime"] = pd.to_datetime(
        min_mag_limit.mjd + 2400000.5, origin="julian", unit="D"
    )
    min_mag_limit["count"] = mag_limit_roll["counter"].sum().astype(int)
    min_mag_limit = (
        min_mag_limit.reset_index()
        .rename(columns={"datetime": "end_datetime"})
        .set_index("start_datetime", drop=False)
    )
    min_mag_limit["m5"] = min_mag_limit["m5"].fillna(-np.inf)

    # Infer which windows do not have a full set of samples, and toss them
    sample_dt = _infer_time_sampling(mag_limit)
    expected_samples = int(np.floor((roll_window.to(sample_dt.unit) / sample_dt).value))

    min_mag_limit.query(
        f"(count == {expected_samples}) or (count == {expected_samples+1})",
        inplace=True,
    )
    min_mag_limit.sort_values("count", ascending=False).groupby(
        level="start_datetime"
    ).first()

    return min_mag_limit


def _find_gaps(mjds, min_gap, season_gap, location, night_epoch_mjd=0):
    gaps = pd.DataFrame({"start": np.unique(np.sort(mjds))})
    gaps["end"] = gaps.start.shift(-1)
    gaps.dropna(inplace=True)
    gaps["duration"] = gaps["end"] - gaps["start"]
    gaps["mjd"] = 0.5 * (gaps["end"] + gaps["start"])
    gaps["night_before"] = compute_night_mjd(gaps["start"], location) - night_epoch_mjd
    gaps["night_after"] = compute_night_mjd(gaps["end"], location) - night_epoch_mjd
    gaps["gap_nights"] = gaps["night_after"] - gaps["night_before"]
    gaps.query(
        f"({min_gap} <= gap_nights) and ({season_gap} > gap_nights)", inplace=True
    )
    gaps.set_index("mjd", inplace=True)
    gaps.sort_index(inplace=True)
    return gaps


def _find_bridge_nights(all_mag_limit, location, config):
    oversampled_mag_limit = (
        all_mag_limit.sort_index()
        .loc[(config["field_name"], config["mag_lim_band"])]
        .sort_index()
        .copy()
    )
    mag_limit = _compute_rolling_m5(oversampled_mag_limit, config["sequence_duration"])
    good_mag_limit = mag_limit.query(f'm5>{config["mag_limit"]}')
    night_epoch_mjd = (
        compute_night_mjd(mag_limit.iloc[0].mjd, location) - mag_limit.iloc[0].night
    )
    gaps = _find_gaps(
        good_mag_limit.mjd,
        config["min_gap"].to(u.day).value,
        config["season_gap"].to(u.day).value,
        location,
        night_epoch_mjd,
    )
    gaps["bridge_mjd"] = np.nan
    gaps["has_bridge"] = False
    max_gap = config["max_gap"].to(u.day).value
    for mjd, gap in gaps.iterrows():
        candidate_bridges = mag_limit.query(
            f"(night > {gap.night_before}) and (night < {gap.night_after})"
        ).query(f"(mjd < {gap.start+max_gap}) and (mjd > {gap.end-max_gap})")
        if len(candidate_bridges) == 0:
            continue
        best_bridge = candidate_bridges.loc[candidate_bridges["m5"].idxmax()]
        # Sometimes there can be two time windows with the same starting,
        # differing by a sample time.
        if isinstance(best_bridge, pd.DataFrame):
            best_bridge = best_bridge.sort_values("count").iloc[-1]

        gaps["has_bridge"] = True
        gaps.loc[mjd, "bridge_mjd"] = best_bridge["mjd"]

    gaps["bridge_night_mjd"] = compute_night_mjd(gaps["bridge_mjd"].fillna(0), location)
    gaps["night_before_mjd"] = (gaps["night_before"] + night_epoch_mjd).astype(int)
    gaps["night_after_mjd"] = (gaps["night_after"] + night_epoch_mjd).astype(int)
    return gaps


def _compute_scheduled(m5_limits, schedule, sequence_duration):
    scheduled = (
        m5_limits["scheduled"]
        .reset_index()
        .set_index("mjd", drop=False)
        .sort_index()
        .copy()
    )

    seq_days = sequence_duration.to(u.day).value
    for _, obs_seq in schedule.iterrows():
        start_mjd = obs_seq.mjd
        end_mjd = obs_seq.mjd + seq_days
        scheduled.loc[start_mjd:end_mjd, "scheduled"] = True

    scheduled.set_index(m5_limits.index.names, inplace=True)
    return scheduled["scheduled"]


def main():
    """Parse command line arguments and config file, and run"""
    parser = ArgumentParser()
    parser.add_argument("config", help="configuration file")
    parser.add_argument("m5", help="file from which to load limiting magnitudes")
    parser.add_argument("output", help="file in which to write results")

    args = parser.parse_args()
    config_fname = args.config
    m5_fname = args.m5
    output_fname = args.output

    config = read_config(config_fname)

    logger.info("Reading m5 from %s", m5_fname)
    m5_limits = (
        pd.read_hdf(m5_fname)
        .reset_index()
        .query("sun_alt < -18")
        .set_index(["field_name", "band", "mjd"], drop=False)
        .assign(scheduled=False)
    )

    schedule = schedule_all(m5_limits, config["location"], config)
    schedule.to_csv(output_fname, sep="\t", index=False, header=True)

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
