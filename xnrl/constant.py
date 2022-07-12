# # Distribution Statement A. Approved for public release. Distribution unlimited.
# # 
# # Author:
# # Naval Research Laboratory, Marine Meteorology Division
# # 
# # This program is free software:
# # you can redistribute it and/or modify it under the terms
# # of the NRLMMD License included with this program.
# # If you did not receive the license, see
# # https://github.com/U-S-NRL-Marine-Meteorology-Division/
# # for more information.
# # 
# # This program is distributed WITHOUT ANY WARRANTY;
# # without even the implied warranty of MERCHANTABILITY
# # or FITNESS FOR A PARTICULAR PURPOSE.
# # See the included license for more details.

import logging
import os
from collections import defaultdict

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_DIR = os.path.join(THIS_DIR, "..", "tests", "data")
WORK_DIR = os.environ.get("WORKDIR", os.environ.get("SCRATCH", ""))

LOGGER_FMT = "%(asctime)s %(levelname)8s -- %(message)s"
logging.basicConfig(format=LOGGER_FMT, datefmt="%m/%d/%y %H:%M:%S")
logging.addLevelName(
    logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING)
)
LOG = logging.getLogger("xnrl")
LOG_LEVS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

NUM_THREADS = 1
MASK_VALUE = -9999
LOG_LEV = LOG_LEVS[1]
METHOD = "nearest"
TEMPORAL_DIM = "tau"

COAMPS = "coamps"
NAVGEM = "navgem"
NEPTUNE = "neptune"
NEPTUNE_LAM = "neptune_lam"
GENERIC = "generic"
RAOB = "raob"
CM1 = "cm1"
RADIANCE = "radiance"
GFS = "gfs"

GRIB = "grib"
HDF5 = "hdf5"
FLATFILE = "flatfile"
NETCDF = "netcdf"
TGZ = "tgz"
GRADS = "grads"
VISUAL = "visual"

DTG_FMT = "%Y%m%d%H"

META_COLS = ["num_xs", "num_ys", "grid_dim", "directory", "file", "path"]
BASE_DIMS = ["field", "exp", "mbr", "ini", TEMPORAL_DIM, "lev"]
GRID_DIMS = ["lat", "lon"]
XY_DIMS = ["y", "x"]
BASE_COLS = ["lev_type", "grid_dim"]
COLS_DIMS = BASE_COLS + BASE_DIMS + ["time"]
UNTITLED = "untitled"
ROW_DLM = ", "

METHODS = {
    "select": ["nearest"],
    "reindex": ["ffill", "pad", "bfill", "backfill"],
    "interp": ["linear", "nearest_regrid", "zero", "slinear", "quadratic", "cubic"],
}

FILE_FMTS = {
    HDF5: "{exp}_xoutf_{grid_dim}_{ini}_{tau}.hdf5",
    FLATFILE: {
        NAVGEM: (
            "{field:*<6.6}_{lev_type:*<3.3}_{lev:04.0f}.0_0000.0_"
            "{grid_dim:*<11.11}_{ini:%Y%m%d%H}_{hh:04.0f}{mm:02.0f}{ss:02.0f}_fcstfld"
        ),
        COAMPS: (
            "{field:*<6.6}_{lev_type:*<3.3}_{lev:06.0f}_000000_"
            "{grid_dim:*<11.11}_{ini:%Y%m%d%H}_{hh:04.0f}{mm:02.0f}{ss:02.0f}_fcstfld"
        ),
    },
    NETCDF: "{model}_{lev_type}_{grid_dim}_{exps}.nc",
}

COORD_ATTR = {
    "exp": {"long_name": "experiment"},
    "mbr": {"standard_name": "realization", "long_name": "ensemble member"},
    "sfc": {
        "standard_name": "altitude",
        "long_name": "surface",
        "positive": "up",
        "units": "m",
    },
    "pre": {
        "standard_name": "air_pressure",
        "long_name": "pressure level",
        "positive": "down",
        "units": "hPa",
    },
    "prl": {
        "standard_name": "air_pressure",
        "long_name": "pressure level",
        "positive": "down",
        "units": "hPa",
    },
    "sig_m": {
        "standard_name": "atmosphere_sigma_coordinate",
        "long_name": "sigma at layer midpoints",
        "positive": "up",
        "units": "",
    },
    "sig_w": {
        "standard_name": "atmosphere_sigma_coordinate",
        "long_name": "sigma at layer edges",
        "positive": "up",
        "units": "",
    },
    "zht": {
        "standard_name": "height",
        "long_name": "height",
        "units": "m",
        "positive": "up",
    },
    "flt": {
        "standard_name": "height",
        "long_name": "height",
        "units": "ft",
        "positive": "up",
    },
    "ini": {"standard_name": "forecast_reference_time", "long_name": "initialization"},
    "tau": {"standard_name": "forecast_period", "long_name": "forecast time"},
    "time": {"standard_name": "time", "long_name": "valid time"},
    "crs": {"long_name": "coordinate reference system"},
    "lat": {
        "standard_name": "latitude",
        "long_name": "latitude",
        "units": "degrees_north",
    },
    "lon": {
        "standard_name": "longitude",
        "long_name": "longitude",
        "units": "degrees_east",
    },
    "idx": {"long_name": "native grid index"},
    "native": {"long_name": "native grid index"},
    "m2d": {"long_name": "native grid weights", "units": "m^2"},
    "y": {"standard_name": "northing", "long_name": "northing", "units": "m"},
    "x": {"standard_name": "easting", "long_name": "easting", "units": "m"},
    "msl": {
        "standard_name": "air_pressure_at_sea_level",
        "long_name": "mean sea level pressure",
        "units": "Pa",
    },
    "terrht": {"long_name": "terrain height", "units": "m"},
    "lndsea": {"long_name": "land mask"},
    "pressure": {
        "standard_name": "air_pressure",
        "long_name": "pressure level",
        "units": "Pa",
    },
}

MODEL_META = defaultdict(dict)
MODEL_META[COAMPS][FLATFILE] = {
    # cldmix_sig_028030_000010_1a0108x0092_2017060100_00000000_fcstfld
    "meta_name": ["latitu", "longit", "terrht", "lndsea", "datahd", "movehd"],
    "indices": {
        "field": slice(0, 6),  # cldmix
        "lev_type": slice(7, 10),  # sig
        "lev": slice(11, 17),  # 028030
        "lev_lower": slice(18, 24),  # 000010
        "nest_num": 25,  # 1
        "nest_type": 26,  # a
        "num_xs": slice(27, 31),  # 0108
        "num_ys": slice(32, 36),  # 0092
        "grid_dim": slice(25, 36),  # 1a0108x0092 redundant
        "ini": slice(37, 47),  # 2017060100
        "tau": slice(48, 52),  # 0000 (0000) the last four is useless
    },
    "proj": {
        1: "mercator",
        2: "lambert_conformal_conic",
        3: "polar_stereographic",
        4: "latitude_longitude",
        5: "orthographic",
    },
    "long_name": {
        "airtmp": "Air Temperature",
        "albedo": "Albedo",
        "ceilht": "Cloud Ceiling Height",
        "cldbas": "Cloud Base Height",
        "cldbrh": "Cloud Base Height thru RH",
        "cldmix": "Cloud Mixing Ratio",
        "cldtop": "Cloud Top Height",
        "cldtrh": "Cloud Top Height thru RH",
        "cltopr": "Cloud Top Temperature thru RH",
        "cltopt": "Cloud Top Temperature",
        "conpac": "Accumulated Convection Precipitation",
        "diverg": "Divergence",
        "dwptdp": "Dewpoint Depression",
        "emixht": "Eddy Mixing Coefficient for Heat",
        "emixmt": "Eddy Mixing Coefficient for Momentum",
        "evapdh": "Evaporative Duct Height",
        "geopht": "Geopotential Height",
        "gmrefr": "Modified Refractivity Gradient",
        "grdtmp": "Ground / Sea Surface Temperature",
        "grdwet": "Ground Wetness",
        "icemix": "Ice Mixing Ratio",
        "lahflx": "Latent Heat Flux",
        "latitu": "Latitude",
        "lndsea": "Land Sea Mask",
        "lonflx": "Longwave Radiation",
        "longit": "Longitude",
        "nradfl": "Net Radiation",
        "pblzht": "Planetary Boundary Layer Height",
        "perprs": "Perturbation Pressure",
        "pottmp": "Potential Temperature",
        "qqstar": "Scale Mixing Ratio for Surface",
        "radhtr": "Radiative Heat Rate",
        "ranmix": "Rain Mixing Ratio",
        "relhum": "Relative Humidity",
        "roguhl": "Surface Roughness",
        "seaice": "Sea Ice Coverage",
        "seatmp": "Sea Surface Temperature",
        "sehflx": "Sensible Heat Flux",
        "slpres": "Sea Level Pressure",
        "snomix": "Snow Mixing Ratio",
        "snowdp": "Snow Depth",
        "solflx": "Solar Radiation",
        "soltmp": "Deep Soil Temperature",
        "stapac": "Accumulated Stable Precipitation",
        "terrht": "Terrain Height",
        "totflx": "Surface Heat Flux",
        "trdown": "Downward Total Radiation",
        "trpres": "Tropopause Pressure",
        "ttlcvr": "Total Cloud Coverage",
        "ttlpcp": "Bucket Total Precipitation",
        "ttlprs": "Total Pressure",
        "ttlsac": "Accumulated Snow",
        "ttstar": "Scale Temperature for Surface",
        "turbke": "Turbulent Kinetic Energy",
        "ustrue": "True U-Component of Wind Stress",
        "uustar": "Scale Velocity for Surface Wind",
        "uutrue": "True U-Velocity Component",
        "uuwind": "Grid U-Velocity Component",
        "visibl": "Visibility",
        "vpress": "Water Vapor Pressure",
        "vstrue": "True V-Component of Wind Stress",
        "vvtrue": "True V-Velocity Component",
        "vvwind": "Grid V-Velocity Component",
        "wstres": "Wind Stress",
        "wvapor": "Water Vapor Mixing Ratio",
        "wwwind": "Vertical W-Velocity Component",
    },
    "units": {
        "airtmp": "K",
        "albedo": "K",
        "ceilht": "fraction",
        "cldbas": "m",
        "cldbrh": "m",
        "cldmix": "g/g",
        "cldtop": "m",
        "cldtrh": "m",
        "cltopr": "K",
        "cltopt": "K",
        "conpac": "kg/m^2",
        "diverg": "1/s",
        "dwptdp": "K",
        "emixht": "mm/s",
        "emixmt": "mm/s",
        "evapdh": "m",
        "geopht": "m",
        "gmrefr": "1/km",
        "grdtmp": "K",
        "grdwet": "fraction",
        "icemix": "g/g",
        "lahflx": "W/m^2",
        "latitu": "degree",
        "lndsea": "numeric",
        "lonflx": "W/m^2",
        "longit": "degree",
        "nradfl": "W/m^2",
        "pblzht": "m",
        "perprs": "?",
        "pottmp": "K",
        "qqstar": "kg/kg",
        "radhtr": "K/s",
        "ranmix": "g/g",
        "relhum": "percent",
        "roguhl": "percent",
        "seaice": "percent",
        "seatmp": "K",
        "sehflx": "W/m^2",
        "slpres": "hPa",
        "snomix": "g/g",
        "snowdp": "kg/m^2",
        "pcph2o": "kg/m^2",
        "solflx": "W/m^2",
        "soltmp": "K",
        "stapac": "kg/m^2",
        "terrht": "m",
        "totflx": "W/m^2",
        "trdown": "W/m^2",
        "trpres": "hPa",
        "ttlcvr": "percent",
        "ttlpcp": "kg/m^2",
        "ttlprs": "hPa?",
        "ttlsac": "kg/m^2",
        "ttstar": "K",
        "turbke": "m^2/s^2",
        "ustrue": "Nt/m^2",
        "uustar": "m/s",
        "uutrue": "m/s",
        "uuwind": "m/s",
        "visibl": "km",
        "vpress": "hPa",
        "vstrue": "Nt/m^2",
        "vvtrue": "m/s",
        "vvwind": "m/s",
        "wstres": "Nt/m^2",
        "wvapor": "kg/kg",
        "wwwind": "m/s",
    },
}

MODEL_META[NAVGEM][FLATFILE] = {
    # ph2otu_sfc_0000.0_0000.0_glob720x361_2018030312_00000000_fcstfld
    "indices": {
        "field": slice(0, 6),  # ph2otu
        "lev_type": slice(7, 10),  # sfc
        "lev": slice(11, 17),  # 0000.0
        "num_xs": slice(28, 32),  # b720 (the b gets stripped later)
        "num_ys": slice(33, 36),  # 361
        "grid_dim": slice(25, 36),  # glob720x361 redundant
        "nest_num": [],
        "ini": slice(37, 47),  # 2017060100
        "tau": slice(48, 52),  # 0000 (0000) the last four is useless
    },
    "proj": {1: "latitude_longitude"},
    "long_name": {
        "totpcp": "Bucket Total Precipitation",
        "pcph2o": "Total Precipitable Water",
        "lsitab": "Land Sea Mask",
        "icecon": "Sea Ice Concentration",
        "spchum": "Specific Humidity",
        "lspacc": "Cumulative Large Scale Precipitation Flux",
        "cumacc": "Cumulative Convective Precipitation Flux",
        "sntlsc": "Cumulative Large Scale Snowfall Flux",
        "sntcum": "Cumulative Convective Precipitation Flux",
        "evpacc": "Cumulative Total Evaporation Flux",
        "swdacc": "Cumulative Downward Shortwave Flux",
        "swnacc": "Cumulative Net Shortwave Flux",
        "lwdacc": "Cumulative Downward Longwave Radiation Flux",
        "lwnacc": "Cumulative Net Longwave Radiation Flux",
        "snfacc": "Cumulative Sensible Heat Flux",
        "lhfacc": "Cumulative Latent Heat Flux",
        "ustacc": "Cumulative Zonal Momentum Flux",
        "vstacc": "Cumulative Meridional Momentum Flux",
        **MODEL_META[COAMPS][FLATFILE]["long_name"],
    },
    "standard_name": {
        "geopht": "geopotential_height",
        "ttlprs": "air_pressure",
        "airtmp": "air_temperature",
        "uuwind": "eastward_wind",
        "vvwind": "northward_wind",
        "spchum": "specific_humidity",
        "relhum": "relative_humidity",
        "clouds": "cloud_area_fraction_in_atmosphere_layer",
        "cldwtr": "mass_fraction_of_cloud_liquid_water_in_air",
        "cldice": "mass_fraction_of_cloud_ice_air",
        "swflxd": "surface_downwelling_shortwave_flux_in_air",
        "lwflxd": "surface_downwelling_longwave_flux_in_air",
        "swflxu": "surface_upwelling_shortwave_flux_in_air",
        "lwflxu": "surface_upwelling_longwave_flux_in_air",
        "lahflx": "surface_upward_latent_heat_flux",
        "sehflx": "surface_upward_sensible_heat_flux",
        "solflx": "toa_incoming_shortwave_flux",
        "pcph2o": "atmosphere_mass_content_of_water_vapor",
        "grdtmp": "surface_temperature",
        "terrht": "surface_altitude",
        "albedo": "surface_albedo",
        "snowdp": "surface_snow_amount",
    },
    "units": {
        "lsitab": "fraction",
        "icecon": "fraction",
        "spchum": "kg/kg",
        "lspacc": "kg/m^2",
        "cumacc": "kg/m^2",
        "sntlsc": "kg/m^2",
        "sntcum": "kg/m^2",
        "evpacc": "kg/m^2",
        "swdacc": "W/m^2*s",
        "swnacc": "W/m^2*s",
        "lwdacc": "W/m^2*s",
        "lwnacc": "W/m^2*s",
        "snfacc": "W/m^2*s",
        "lhfacc": "W/m^2*s",
        "ustacc": "kg/m/s",
        "vstacc": "kg/m/s",
        **MODEL_META[COAMPS][FLATFILE]["units"],
    },
}

MODEL_META[GFS][FLATFILE] = {
    # dvala1201703081213200000100000000prl
    "indices": {
        "field": slice(0, 4),  # dval
        "grid_dim": slice(4, 6),  # 1a
        "ini": slice(6, 16),  # 2017030812
        "tau": slice(16, 19),  # 132
        "lev": slice(24, 28),  # 1000
        "lev_type": slice(33, 36),  # prl
        "num_xs": [],
        "num_ys": [],
        "nest_num": [],
    }
}
MODEL_META[COAMPS][GRIB] = {
    # US058GMET-GR1dyn.COAMPS-CENCOOS_CENCOOS-n1-c1_00600F0NL2019061112_0100_007000-000000wnd_vtru
    "meta_name": ["latitude", "longitude", "terr_ht", "land_sea"],
    "indices": {
        "field": slice(84, None),  # wnd_vtru
        "lev_type": slice(67, 70),  # 100
        "lev": slice(71, 76),  # 00700
        "lev_lower": slice(78, 83),  # 00000
        "nest_type": 40,  # 1
        "nest_num": 41,  # 1
        "grid_dim": 41,  # not in filename; just use nest_num
        "ini": slice(55, 65),  # 2019061112
        "tau": slice(46, 49),  # 006,
    },
    "lev_type": {
        1: "sfc",
        100: "pressure",
        102: "msl",
        105: "zht",
        107: "sigma",
        250: "flt",
    },
    "proj": {
        1: "latitude_longitude",
        2: "lambert_conformal_conic",
        3: "polar_stereographic",
        4: "latitude_longitude",
        5: "orthographic",
    },
    "long_name": {
        "pres": "pressure",
        "pres_msl": "mean sea level pressure",
        "pres_tend": "pressure tendancy",
        "geop_ht": "geopotential height",
        "height": "height geometric",
        "total_ozone": "ozone total",
        "air_temp": "air temperature",
        "virt_air_temp": "virtual temperature",
        "ptnl_temp": "potential temperature",
        "max_air_temp": "maximum air temperature",
        "min_air_temp": "minimum air temperature",
        "dwpt_temp": "dewpoint temperature",
        "dwpt_dprs": "dewpoint depression",
        "temp_lapse_rate": "lapse rate temperature",
        "visib": "prevailing visibility",
        "temp_dif": "standard air temperature difference",
        "wnd_dir": "wind direction",
        "wind_spd": "wind speed",
        "wnd_ucmp": "U-wind component",
        "wnd_vcmp": "V-wind component",
        "strm_func": "stream function",
        "vel_ptnl": "velocity potential",
        "wnd_vert_vel": "vertical velocity",
        "wnd_wcmp": "W-wind component",
        "abs_vort": "absolute vorticity",
        "div": "absolute divergence",
        "rltv_vort": "relative vorticity",
        "curr_ucmp": "U-current component",
        "curr_vcmp": "V-current component",
        "spec_hum": "specific humidity",
        "rltv_hum": "relative humidity",
        "mix_ratio": "water vapor mixing ratio",
        "prcp_h20": "precipitable water",
        "vpr_pres": "vapor pressure",
        "cld_ice": "cloud ice",
        "rain_rate": "rain rate",
        "ttl_prcp": "accumulated total precipitation",
        "ttl_prcp_01": "accumulated total bucket precipitation",
        "ttl_prcp_03": "accumulated total bucket precipitation",
        "ttl_prcp_06": "accumulated total bucket precipitation",
        "ttl_prcp_12": "accumulated total bucket precipitation",
        "stab_prcp": "accumulated non-convective precipitation",
        "stab_prcp_03": "accumulated bucket non-convective precipitation",
        "stab_prcp_06": "accumulated bucket non-convective precipitation",
        "stab_prcp_12": "accumulated bucket non-convective precipitation",
        "conv_prcp": "accumulated convective precipitation",
        "conv_prcp_03": "accumulated bucket convective precipitation",
        "conv_prcp_06": "accumulated bucket convective precipitation",
        "conv_prcp_12": "accumulated bucket convective precipitation",
        "ttl_snow": "snowfall accumulation water equivalent",
        "snw_dpth": "snow depth",
        "pbl_depth": "planetary boundary layer height",
        "ttl_cld_cvr": "total cloud cover",
        "conv_cld": "convective cloud cover",
        "low_cld_cvr": "low cloud cover",
        "mid_cld_cvr": "mid cloud cover",
        "hi_cld_cvr": "high cloud cover",
        "cld_wtr": "cloud water",
        "sea_temp": "sea surface temperature",
        "land_sea": "land sea flag",
        "wtr_sfc_elev": "deviation from mean sea level",
        "roughness_len": "roughness length",
        "albedo": "albedo",
        "soil_temp": "soil temperature",
        "sal": "salinity",
        "air_dens": "air density",
        "ice_cvrg": "ice coverage",
        "ice_thkn": "ice thickness",
        "ice_vel_ucmp": "ice velocity U-component",
        "ice_vel_vcmp": "ice velocity V-component",
        "ice_div": "ice divergence",
        "sig_wav_ht": "significant combined wave height",
        "wnd_wav_dir": "wind wave direction",
        "wnd_wav_ht": "wind wave height",
        "wnd_wav_per": "wind wave period",
        "swl_wav_dir": "swell wave direction",
        "swl_wav_ht": "swell wave height",
        "swl_wav_per": "swell wave period",
        "pr_wav_dir": "primary wave direction",
        "pr_wav_per": "primary wave period",
        "scdy_wav_dir": "secondary wave direction",
        "scdy_wav_per": "secondary wave period",
        "sol_rad": "solar radiation flux",
        "ir_flux": "infrared flux",
        "ltnt_heat_flux": "latent heat flux",
        "snsb_heat_flux": "sensible heat flux",
        "wnd_strs_ucmp": "U-component wind stress",
        "wnd_strs_vcmp": "V-component wind stress",
        "duct_base_ht": "duct base height",
        "duct_strength": "air duct strength",
        "qstar": "surface mixing ratio scale",
        "tstar": "surface temperature scale",
        "grad_mod_refr": "gradient of modified refractivity",
        "grnd_sea_temp": "skin temperature",
        "geop_ht_mn": "geopotential height mean",
        "cld_base": "cloud base height",
        "cld_top_temp": "cloud top temperature",
        "aero_vis_rng": "visual range",
        "clr_air_turb": "elrod turbulence index",
        "contrail_prbl": "contrail probability",
        "wtr_ice_intl": "first integral liquid water equivalency",
        "geop_ht_mn_trnc18": "truncated geopotential height mean",
        "wnd_ucmp_mn_trnc18": "truncated mean U-component wind",
        "wnd_vcmp_mn_trnc18": "truncated mean V-component wind",
        "geop_ht_24hr_dif": "geopoential height difference",
        "dynm_ht": "dynamic height",
        "wtr_ice_top": "water ice top",
        "duct_mexcess": "duct M excess",
        "duct_thickness": "duct thickness",
        "duct_top": "duct height top",
        "gale_wnd_prob": "wind gale probability",
        "abs_vort_ens_stdev": "standard deviation of absolute vorticity of ensemble members",
        "air_temp_ens_stdev": "standard deviation of air temperature of ensemble members",
        "rltv_hum_ens_stdev": "standard deviation of relative humidity of ensemble members",
        "wcap_prbl": "white cap wave probability",
        "vrtx_wav": "spectral atmosphere wave vortex",
        "wnd_vect_dif_mgtd": "shear magnitude",
        "turb_ke": "turbulent kinetic energy",
        "soil_type": "soil type",
        "aero_concen_du": "total dust concentration",
        "cape": "CAPE",
        "cin": "CIN",
        "frozen_rain": "accumulated bucket frozen precipitation",
        "aero_concen_sm": "total smoke concentration",
        "ocn_heat_cntnt": "ocean heat content",
        "soil_moist_liq_frzn": "volumetric total soil moisture",
        "aero_concen_su": "total sulfate concentration",
        "aero_concen_sa": "total salt concentration",
        "geop_ht_ens_stdev": "standard deviation of geopotential height of ensemble members",
        "prob_sig_wav_ht_gt12ft": "wave height > 12ft probability",
        "prob_sig_wav_ht_gt18ft": "wave height > 18ft probability",
        "pres_msl_ens_stdev": "standard deviation of mean sea level pressure of ensemble members",
        "evap_duct_ht": "evaporative duct height",
        "peak_wav_per_ens_stdev": "standard deviation of peak wave period of ensemble members",
        "pres_ens_stdev": "standard deviation of terrain pressure of ensemble members",
        "sig_wav_ht_ens_stdev": "standard deviation of significant wave height of ensemble members",
        "swl_wav_ht_ens_stdev": "standard deviation of swell wave height of ensemble members",
        "aero_optdep": "aerosol optical depth",
        "aero_extinct_part": "partition aerosol extinction",
        "ozone_mix_ratio": "ozone mixing ratio",
        "aero_extinct_uv": "ultraviolet aerosol extinction",
        "aero_extinct_vis": "visible aerosol extinction",
        "aero_extinct_nir": "near infrared aerosol extinction",
        "ttl_prcp_ens_stdev": "standard deviation of total precipitation of ensemble members",
        "aero_extinct_mw": "mid infrared aerosol extinction",
        "aero_extinct_lw": "long infrared aerosol extinction",
        "aero_asym_uv": "ultraviolet asymmetry parameter",
        "aero_asym_vis": "visible asymmetry parameter",
        "aero_asym_nir": "near infrared asymmetry parameter",
        "aero_asym_mw": "mid infrared asymmetry parameter",
        "aero_asym_lw": "long infrared asymmetry parameter",
        "aero_scatter_uv": "ultraviolet aerosol scattering",
        "aero_scatter_vis": "visible aerosol scattering",
        "aero_scatter_nir": "near infrared aerosol scattering",
        "ttl_prcp_ens_stdev_06": "standard deviation of 6 hr total precipitation of ensemble members",
        "aero_optdep_sm": "smoke optical depth",
        "aero_optdep_su": "sulphate optical depth",
        "aero_optdep_sa": "salt optical depth",
        "aero_optdep_du": "dust optical depth",
        "temp_grad": "temperature gradient",
        "freezing_rain": "accumulated freezing rain",
        "peak_wav_dir": "peak wave direction",
        "peak_wav_per": "peak wave period",
        "eq_rad_ref_fact_hym-SU_mi_Mi": "vertical column maximum of the sum of equivalent radar reflectivity factor",
        "shrt_wav": "short wave spectral decomposition",
        "rsdl_wav": "residual spectral decomposition",
        "lng_wav": "long wave spectral decomposition",
        "mag_ditch_hdg": "magnetic ditch heading",
        "wind_ucmp_mn": "mean U-component wind",
        "wind_vcmp_mn": "mean V-component wind",
        "wind_dif_ucmp_mn": "mean U-component wind difference",
        "wind_dif_vcmp_mn": "mean V-component wind difference",
        "grnd_wet": "ground wetness",
        "max_wav_ht": "maximum wave height",
        "snsb_ltnt_heat_flux": "sensible and latent heat flux",
        "ttl_heat_flux": "total heat flux",
        "frnt_anal": "frontal analysis",
        "fog": "fog probability",
        "wnd_spd_ens_stdev": "standard deviation of wind speed of ensemble members",
        "frz_ht": "freezing height",
        "sea_sfc_ht_corr": "sea surface height anomaly",
        "ceil_ht": "cloud ceiling height",
        "wnd_strs": "wind stress",
        "snw_age": "snow age",
        "terr_ht": "terrain height",
        "latitude": "latitude",
        "longitude": "longitude",
        "cld_top": "cloud top height",
        "depth": "depth",
        "sea_temp_anom": "sea temperature anomaly",
        "wtr_ice_int2": "second integral integrated liquid water equivalency",
        "stab_cld": "stable cloud cover",
        "lift_cdns_lev": "lifting condensation level",
        "thkn": "thickness",
        "peak_wnd_spd": "peak wind speed",
        "cld_wtr_dnst": "cloud water density",
        "cld_ice_dnst": "cloud ice density",
        "wnd_utru": "U-component true wind",
        "wnd_vtru": "V-component true wind",
        "ustar": "scale surface frictional velocity",
        "wtr_ice_int3": "third integral integrated liquid water equivalency",
        "sig_wav_ht_prob": "significant wave height probability",
        "ttl_prcp_prob": "total precipitation probability",
        "wnd_spd_prob": "wind speed probability",
        "air_temp_AM_mi_Mi": "average air temperature across all ensemble members",
        "air_temp_SM_mi_Mi": "standard deviation of air temperature WRT mean of ensemble members",
        "geop_ht-AM_mi_Mi": "unweighted mean of geopotential height across all ensemble members",
        "pres_AM_mi_Mi": "unweighted mean of pressure across all ensemble members",
        "dwpt_dprs-AM_mi_Mi": "unweighted mean of dew point depression across all ensemble members",
        "wnd_ucmp-AM_mi_Mi": "unweighted mean of U-component wind across all ensemble members",
        "rltv_hum-AM_mi_Mi": "unweighted mean of relative humidity across all ensemble members",
        "wnd_vert_vel-AM_mi_Mi": "unweighted mean of vertical velocity across all ensemble members",
        "abs_vort-AM_mi_Mi": "unweighted mean of absolute vorticity across all ensemble members",
        "rltv_hum-SM_mi_Mi": "standard deviation of relative humidity across all ensemble members",
        "geop_ht-SM_mi_Mi": "standard deviation of geopotential height across all ensemble members",
        "peak_wav_per-AM_mi_Mi": "unweighted mean of peak wave period across all ensemble members",
        "peak_wav_per-SM_mi_Mi": "standard deviation of peak wave period across all ensemble members",
        "sig_wav_ht-AM_mi_Mi": "unweighted mean of significant wave height across all ensemble members",
        "sig_wav_ht-SM_mi_Mi": "standard deviation of significant wave height across all ensemble members",
        "swl_wav_ht-AM_mi_Mi": "unweighted mean of swell wave height across all ensemble members",
        "cld_typ": "cloud type",
        "elect_dnst": "electron density",
        "wq_rad_ref_fact_hym": "vertical-column-maximum of the sum of equivalent radar reflectivity factor",
        "ice_cvrg_err": "ice coverage statistical error",
        "ir_flux_down": "downward flux of longwave radiation",
        "ir_flux_up": "upward flux of longwave radiation",
        "lwp_c": "vertically integrated water/ice content",
        "lwp_s": "vertically integrated water/ice content using radiances",
        "mix_ratio_cld_wtr": "cloud water mixing ratio",
        "mix_ratio_graupel": "graupel mixing ratio",
        "mix_ratio_ice_wtr": "ice mixing ratio",
        "mix_ratio_rain": "rain mixing ratio",
        "mix_ratio_snw": "snow mixing ratio",
        "mod_refr": "modified refractivity",
        "sea_temp_err": "sea temperature statistical error",
        "snw_h2o_equiv": "estimated depth of snow in water equivalency",
        "soil_moist_liq": "liquid soil moisture",
        "sol_rad_down": "downward flux of shortwave radiation",
        "sol_rad_up": "upward moving flux of shortwave radiation",
        "ttl_cld_cvr-AM_mi_Mi": "total cloud cover across all ensemble members",
        "veg_cnpy_sfc_wtr": "moisture on the surface of the plant canopy",
        "veg_typ": "vegetation type",
        "wnd_swl_dir": "direction of wind waves and swell",
        "wnd_ucmp_marine": "U-component marine wind",
        "wnd_vcmp_marine": "V-component marine wind",
        "zero_wav_per": "second integral spectral wave period",
        "zero_wav_per-A_mi_Mi": "second integral mean spectral wave period",
        "min_air_temp_-AM-mi_Mi": "unweighted mean of lowest air temperature across all ensemble members",
        "max_air_temp_-AM-mi_Mi": "unweighted mean of greatest air temperature across all ensemble members",
        "wnd_vcmp-AM_mi_Mi": "unweighted mean of V-component wind speeds across all ensemble members",
        "wnd_vert_vel-SM-mi_Mi": "standard deviation of vertical velocity across all ensemble members",
        "wnd_spd-AM_mi_Mi": "unwieghted mean of wind speed across all ensemble members",
        "wnd_spd-SM-mi_Mi": "standard deviation of wind speed across all ensemble members",
        "rltv_vort-AM_mi_Mi": "unweighted mean of relative vorticity across all ensemble members",
        "rltv_vort-SM-mi_Mi": "standard deviation of relative vorticity across all ensemble members",
        "ttl_prcp-AM_mi_Mi-AC_Xeq6_tauhr": "unwieghted mean of total precipitation 6 hr accumulations for all ensemble members",
        "ttl_prcp-SM_mi_Mi-AC_Xeq6_tauhr": "standard deviation of total precipitation 6 hr accumulations for all ensemble members",
        "wnd_spd_prob-PM_Xgt35d0_kt": "probability of wind speeds above 35 kts",
        "wnd_spd_prob-PM_Xgt20d0_kt": "probability of wind speeds above 20 kts",
        "wnd_spd_prob-PM_Xgt50d0_kt": "probability of wind speeds above 50 kts",
        "wnd_spd_prob-PM_Xgt5d0_kt": "probability of wind speeds above 5 kts",
        "wnd_spd_prob-PM7ltXlt13_kt": "probabilty of wind speeds between 7 to 13 kts",
        "sig_wav_ht_prob-PM_Xgt24d0_ft": "probability of significant wave height above 24 ft",
        "sig_wav_ht_prob-PM_Xgt4d0_ft": "probability of significant wave height above 4 ft",
        "sig_wav_ht_prob-PM_Xgt8d0_ft": "probability of significant wave height above 8 ft",
        "sig_wav_ht_prob-PM_Xgt12d0_ft": "probability of significant wave height above 12 ft",
        "sig_wav_ht_prob-PM_Xgt18d0_ft": "probability of significant wave height above 18 ft",
        "ttl_prcp_prob-PM_Xge0d25_in-AC_Xeq6_tauhr": "probability that total precipitation 6 hr accumulations will be greater than or equal to 0.25 inches",
        "pres_msl-AM_mi_Mi": "unweighted mean of mean sea level across all ensemble members",
        "pres_msl-SM_mi_Mi": "pressure reduced to MSL standard deviation",
        "surface": "surface",
        "isth_lev_0C": "level of 0 deg C isotherm",
        "max_wnd_lev": "maximum wind level",
        "trpp_lev": "topopause level",
        "atms_top": "top of atmosphere",
        "isth_lev": "isothermal level temp 1/100k",
        "isbr_lev": "isobaric level",
        "isbr_lay": "layer between two isobaric layers",
        "msl": "mean sea level",
        "ht_msl": "height above mean sea level",
        "ht_sfc": "height above the surface",
        "ht_lay": "layer between two altitudes above ground",
        "sgma_lev": "sigma level",
        "atms_ix": "atmospheric index",
        "dpth_sfc": "depth below sea surface",
        "mix_lay": "ocean shallow mix layer",
        "wavelen_sfc": "wavelength surface",
        "sky_cvr": "full depth of atmosphere",
        "ocn": "ocean",
        "blw_snd_lay": "below sound layer",
        "shlw_snd_chan": "shallow sound channel",
        "deep_snd_chan": "deep sound channel",
        "snd_lay": "sound layer",
        "dpth_lnd_sfc": "depth below land surface",
        "low_cld": "low cloud layer",
        "isbr_lev_pa": "isobaric level",
        "mid_cld": "middle cloud layer",
        "hi_cld": "high cloud layer",
        "deep_lay": "between 100 and 1000 mb",
        "dpth_ix": "depth index",
        "thcl_lev": "thermocline level",
        "flight_lev": "aircraft altitude above ground",
        "grnd_lay": "ground layer",
    },
    "units": {},
}
MODEL_META[NAVGEM][GRIB] = {
    # US058GMET-GR2mdl.0018_0056_00000F0RL2018081312_0250_010000-000000air_temp.gr2
    "indices": {
        "field": slice(65, -4),  # air_temp
        "lev_type": slice(47, 51),  # 0250
        "lev": slice(52, 57),  # 01000
        "grid_dim": slice(6, 9),  # not in filename
        "ini": slice(36, 46),  # 2018081312
        "tau": slice(27, 30),  # 000,
    },
    "lev_type": {**MODEL_META[COAMPS][GRIB]["lev_type"]},
    "proj": {1: "latitude_longitude"},
    "long_name": {},
    "units": {**MODEL_META[COAMPS][GRIB]["units"]},
}
MODEL_META[NEPTUNE][HDF5] = {
    "varying_attrs": {
        "experiment": "exp",
        "nx_out": "num_xs",
        "ny_out": "num_ys",
        "grid_dimensions": "grid_dim",
        "initialization": "ini",
        "tau": "tau",
    },
    "meta_lev_types": [
        "global_attributes",
        "global attributes",
        "latitude",
        "longitude",
        "x",
        "y",
    ],
    "meta_coords": [
        "latitude",
        "longitude",
        "m2d",
        "z_height",
        "terrain",
        "sigma_z",
        "x",
        "y",
    ],
}
MODEL_META[NEPTUNE_LAM][HDF5] = MODEL_META[NEPTUNE][HDF5]
MODEL_META[NAVGEM][HDF5] = {
    "varying_attrs": {"NumLon": "num_xs", "NumLat": "num_ys", "NumLev": "num_levs"},
    "meta_lev_types": ["Geometry", "Spectral"],
    "meta_coords": ["latitude", "longitude", "m2d"],
}

for model in [GENERIC, NAVGEM, COAMPS, NEPTUNE, NEPTUNE_LAM]:
    if FLATFILE not in MODEL_META[model]:
        if model == NEPTUNE_LAM:
            MODEL_META[model][FLATFILE] = MODEL_META[COAMPS][FLATFILE]
        else:
            MODEL_META[model][FLATFILE] = MODEL_META[NAVGEM][FLATFILE]
    MODEL_META[model][NETCDF] = {"long_name": {}, "units": {}}
    MODEL_META[model][GRADS] = {"long_name": {}, "units": {}}
MODEL_META[GENERIC][GRIB] = {"long_name": {}, "units": {}}
MODEL_META[COAMPS][VISUAL] = {"long_name": {}, "units": {}, "meta_name": ["LAT", "LON"]}

ALL_FILE_TYPES = [GRIB, FLATFILE, HDF5, TGZ, GRADS, NETCDF]
