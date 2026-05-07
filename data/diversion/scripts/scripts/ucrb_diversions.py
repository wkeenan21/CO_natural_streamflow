"""
Script Name: ucrb_diversions.py
Authors: Jacob Knight, Samuel Lopez
Date: 2024-09-30

Description:
    This script automates retrieval of daily time series withdrawal records,
    harmonizes these records with other site records that were not available
    for automated online retrieval, and processes all data into a centralized,
    consistent format. The script utilizes the Master Table and historical time
    series withdrawal records contained in the "input" folder. Output results
    are both CSV (time series records) and PDF (time series plots) files.

Contact Information:
    Emails: jknight@usgs.gov, slopez@usgs.gov

Disclaimer:
    The associated database, identified as “Database of surface water diversion 
    sites and daily withdrawals for the Upper Colorado River Basin, 1980–2022,” 
    has been approved for release by the U.S. Geological Survey (USGS). Although
    this database has been subjected to rigorous review and is substantially
    complete, the USGS reserves the right to revise the data pursuant to 
    further analysis and review. Furthermore, the database is released on
    condition that neither the USGS nor the U.S. Government shall be held
    liable for any damages resulting from its authorized or unauthorized use. 

    As this database contains records sourced from non-USGS portals, neither the
    authors, USGS, nor the U.S. Government shall be held liable for inaccurate
    records. Furthermore, this script and associated database are subject to 
    periodic updates and revision.

Usage:
    To run the script, use the following command in the terminal:
    python ucrb_diversions.py

    Make sure to have the environment ucrb_diversions installed (see ucrb_diversions.yml in scripts folder)
"""

import sys
# sys.path.append('../../ucrb_utils/python_packages_static')
sys.path.append('python_packages_static')
import os
import shutil
import platform
import requests
import re
from io import StringIO
import json

import numpy as np
import pandas as pd
import geopandas as gp
from shapely.geometry import Point

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from calendar import monthrange
import matplotlib.dates as mdates
years1 = mdates.YearLocator()
years5 = mdates.YearLocator(5)
years10 = mdates.YearLocator(10)
years20 = mdates.YearLocator(20)
years_fmt = mdates.DateFormatter('%Y')
from datetime import date

#import spnspecs
#spnspecs.set_graph_specifications()

if 'window' in platform.platform().lower():
    newln = '\n'
else:
    newln = '\r\n'


def divfilter(df=None, mindiff=25., minfact=2., sig=3.):
    """
    Function to identify time series outliers using an initial and secondary filter.

    Parameters
    ----------
    
    df : pandas dataframe
        must contain two columns:
            "datetime" of type np.datetime64
            "discharge_cfs" of type float
        (default is None)
    
    mindiff : float
        minimum amount (in cfs) greater than the median annual max at which a record
        can be deemed an outlier in the first pass filter
        (default is 25.)

    minfact : float
        minimum factor by which a record must be greater than the median annual max
        to be deemed an outlier in the first pass filter
        (default is 2.)

    sig : float
        number of standard deviations above the median annual max at which a record can be deemed
        an outlier in the second pass filter
        (default is 3.)

    Returns
    ----------
    df : pandas dataframe
        identical to input dataframe with additional columns:
            "mdflag" of type int, where a value of 1 indicates records identified as outliers in the first pass filter
            "sdflag" of type int, where a value of 1 indicates records identified as outliers in the second pass filter

    See Also
    --------

    Notes
    -----
    The first pass filter uses the median of the annual maximum values and a minimum difference and minimum factor
    to identify potential outliers. Records meeting the criteria of potential outliers are not included in the
    calculation of the standard deviation of the annual maximum values.

    The second pass filter uses a provided sigma distance to identify potential outliers that are greater than a specified
    number of standard deviations above the median annual maximum rate. The standard deviation value is  calculated on the
    values from the first pass that did not meet the potential outlier criteria of minimum difference and minimum factor

    Examples
    --------
    Need to add
    """

    df.loc[:, "year"] = df.loc[:, "datetime"].apply(lambda x: x.year)
    dfy = df.loc[df["discharge_cfs"] > 0.].groupby("year").max()
    aym = np.median(dfy.loc[:, "discharge_cfs"])
    
    df.loc[:, "mdflag"] = 0
    df.loc[df.apply(lambda x: all((x.discharge_cfs - aym > mindiff, x.discharge_cfs > minfact * aym)), axis=1), "mdflag"] = 1
    
    dfy = df.loc[(df["mdflag"] == 0) & (df["discharge_cfs"] > 0.)].groupby("year").max()
    aymstd = np.std(dfy.loc[:, "discharge_cfs"])
    
    df.loc[:, "sdflag"] = 0
    df.loc[df.apply(lambda x: x.discharge_cfs > (aym + sig * aymstd), axis=1), "sdflag"] = 1
    
    return df, aym


def gaplengths(vals):
    """
    Function to identify the length of consecutive null values surrounding each null value in a series of values.
    This function is used by fill_missing_diversion_values() to identify gaps to be filled that are less than 90 days.
    Gaps longer than 90 days are not filled to avoid interpolating values from irrigation season into non-irrigation season.

    Parameters
    ----------
    
    vals : list of float values
        (default is None)

    Returns
    ----------
    cts : list of int values

    See Also
    --------

    Notes
    -----

    Examples
    --------
    Need to add
    """
    i0 = 0
    x = 0
    cts = []

    for i,v in enumerate(vals):
        if np.isnan(v):
            x += 1
            cts.append(x)
        else:
            cts[i0: i] = [x] * (i - i0)
            i0 = i
            x = 0
            cts.append(x)
    return cts


def build_spdf(start="19791231", end="20220930", spfreq="M", tsfreq="D"):
    """
    Function to build a dataframe of stress periods and time steps for a given date range and intervals

    Parameters
    ----------
    
    start : str
        starting datetime in string format "%Y%m%d"
        (default is "19791231")

    end : str
        ending datetime in string format "%Y%m%d"
        (default is "20220930")

    spfreq : str
        stress period frequency. Must be "M"
        (default is "M")

    Returns
    ----------
    df : pandas dataframe
        includes dates, stress periods assigned by month, and MODFLOW totim values for each day, in 1-index

    See Also
    --------

    Notes
    -----

    Examples
    --------
    Need to add    
    """
    dts = pd.date_range(start=pd.to_datetime(start), end=pd.to_datetime(end), freq=tsfreq)
    df = pd.DataFrame(data={"year": dts.year, "month": dts.month, "day": dts.day}, index=dts)
    if spfreq == "M":
        df.loc[:, "sp"] = df.groupby(["year", "month"]).ngroup()
        gr = df.groupby('sp')
        df.loc[:, "ts"] = gr.cumcount()
    df.loc[:, "totim"] = range(1, len(df) + 1)
    
    return df
    

def format_sites_df(df=None):
    """
    Function to create point geometries from decimal lat/long and reproject to UTM 12N 

    Parameters
    ----------
    
    df : pandas dataframe
        function applies to diversion site dataframes created in each data pulling function
        (default is None)

    Returns
    ----------
    df : pandas dataframe
        original dataframe modified to include point geometries for each site in UTM 12N projection

    See Also
    --------

    Notes
    -----

    Examples
    --------
    Need to add    
    """
    df.loc[:, "geometry"] = df.apply(lambda x: Point(x.siteLong, x.siteLat), axis=1)
    df = gp.GeoDataFrame(df, geometry="geometry", crs="epsg:4269")
    df.to_crs(epsg=model_epsg, inplace=True)
    df.loc[:, "utmX"] = df.loc[:, "geometry"].apply(lambda xx: xx.x)
    df.loc[:, "utmY"] = df.loc[:, "geometry"].apply(lambda xx: xx.y)
    # df.drop(["siteLat", "siteLong"], axis=1, inplace=True)

    return df


def get_cdss_diversion_data(dst_dir=os.path.join("..", "output", "cdss_raw_data"),
                            sites_ifp=os.path.join("..", "input", "ucrb_diversion_master_table.csv"),
                            apiKey=None):
    """
    Function to pull diversion records from Colorado Decision Support Systems website 

    Parameters
    ----------
    
    dst_dir : str
        relative path location to directory to save downloaded data
        (default is "cdss_raw_data")

    sites_ifp : str
        relative path location to csv file containing all UCRB diversion sites.
        This function only attempts to pull data for records with "dataSource" attribute of "CDSS"
        (default is "ucrb_diversion_master_table.csv")

    apiKey : str
        CDSS web services API token, needed to exceed the maximum number of requests and volume of
        data allowed by CDSS without a token. To acquire a token, browse to
        https://dwr.state.co.us/rest/get/help and click the Help button on the top right corner,
        then select the link in the popup to “How to Guide – REST Telemetry POST”

    Exports
    ----------
    Microsoft Excel CSV file "cdss_diversion_sites.csv" containing site information of every site
    for which daily records were pulled

    1 additional CSV file for each site (e.g. "raw_5900584.csv") containing daily diversion records of that site


    Returns
    ----------
    None

    See Also
    --------

    Notes
    -----

    Examples
    --------
    Need to add    
    """

    print("downloading diversion record data to directory {0}".format(dst_dir))
    if os.path.exists(dst_dir):
        pass
        # shutil.rmtree(dst_dir)
        # print("existing diversion data directory found and will be replaced")
    else:
        os.mkdir(dst_dir)

    # organize info for lookup table
    siteIds = []
    siteNames = []
    utmX = []
    utmY = []
    siteSource = []
    siteFiles = []
    siteUse = []
    siteStart = []
    siteEnd = []
    noFillYears = []
    shortID = []
    siteLat = []
    siteLong = []
    destinationCode = []
    destinationFlag = []

    # load site info retrieved with "get_cdss_site_data()"
    sdf = pd.read_csv(sites_ifp)
    sdf = sdf.loc[sdf["dataSource"] == "CDSS"].copy()

    sdf.loc[:, "cdssID"] = sdf.loc[:, "cdssID"].astype(int).astype(str)

    # pull daily diversion data for each site in loop
    website = "https://dwr.state.co.us/Rest/GET/api/v2/structures/divrec/divrecday/"
    outputFormat = "csvforced"
    dateFormat = "dateOnly"
    minDate = "01-01-1980"
    maxDate = "10-01-2022"
    wcIdentifier = "*Total (Diversion)"

    params = {'format': outputFormat,
              'dateFormat': dateFormat,
              'min-dataMeasDate': minDate,
              'max-dataMeasDate': maxDate,
              'wcIdentifier': wcIdentifier}
    if apiKey:
        params["apiKey"] = apiKey

    #for i, r in sdf.iloc[:10, :].iterrows():
    for i, r in sdf.iterrows():
        site = r.cdssID
        # siteFile = "raw_{0}.csv".format(site)
        nm = r.siteName
        siteFile = "{0}.csv".format(r.siteName)
        use = r.siteUse
        params['wdid'] = site
        
        try:
            rr = requests.get(url=website, params=params)
            temp = StringIO(rr.text)
            df = pd.read_csv(temp,skiprows=2)
            
            # save copy of raw downloaded data to source directory
            # df.to_csv(os.path.join(dst_dir, siteFile))

            # modify record to be compatible with other states
            # convert dataMeasDate to datetime and set as index
            df.loc[:, "date"] = pd.DatetimeIndex(df.loc[:, "dataMeasDate"])
            df.rename(columns={"dataValue": "discharge_cfs"}, inplace=True)
            df = df.filter(["date", "discharge_cfs"])
            df.index = df.pop("date")
            # df.to_csv(os.path.join(dst_dir, siteFile.replace("raw", "mod")))
            df.to_csv(os.path.join(dst_dir, siteFile))
            print("downloaded data for site {0} {1}".format(site, nm))

            # append site info to list compatible with other states
            siteIds.append(site)
            siteNames.append(nm)
            siteUse.append(use)
            siteStart.append(r.startDate)
            siteEnd.append(r.endDate)
            noFillYears.append(r.no_fill_years)
            shortID.append(r.shortID)
            siteLat.append(r.decLat)
            siteLong.append(r.decLong)
            siteSource.append(r.dataSource)
            #siteFiles.append(siteFile.replace("raw", "mod"))
            siteFiles.append(siteFile)
            destinationCode.append(r.destinationCode)
            destinationFlag.append(r.destinationFlag)

        except:
            print("could not download or process data from site {0} {1}".format(site, nm))

    # build and export diversion site lookup table for use in build_diversion_tabfiles()
    df = pd.DataFrame(data={"siteID": siteIds, "siteName": siteNames, "siteUse": siteUse,
                            "siteLat": siteLat, "siteLong": siteLong,
                            "siteSource": siteSource, "siteFile": siteFiles,
                            "startDate": siteStart, "endDate": siteEnd,
                            "noFillYears": noFillYears, "shortID": shortID,
                            "destinationCode": destinationCode, "destinationFlag": destinationFlag})
    
    df.loc[:, "siteFolder"] = os.path.split(dst_dir)[-1]
    df_out = format_sites_df(df)
    df_out.to_csv(os.path.join(dst_dir, "..", "cdss_diversion_sites.csv"))


def get_nmose_diversion_data(dst_dir=os.path.join("..", "output", "nmose_raw_data"),
                             sites_ifp=os.path.join("..", "input", "ucrb_diversion_master_table.csv"),
                             sp_df=None,
                             hst_dir=os.path.join("..", "input", "nmose_historical_data"),
                             comb_dir=os.path.join("..", "output", "nmose_combined_data")):
    """
    Function to pull diversion records from New Mexico Office of the State Engineer website 

    Parameters
    ----------
    
    dst_dir : str
        relative path location to directory to save downloaded data
        (default is "nmose_raw_data")

    sites_ifp : str
        relative path location to csv file containing all UCRB diversion sites.
        This function only attempts to pull data for records with "dataSource" attribute of "NMOSE"
        (default is "ucrb_diversion_master_table.csv")

    sp_df : pandas dataframe
        dataframe containing one record per day within period of interest, used for combining automatically
        retrieved data with manually retrieved data located in hst_dir directory

    hst_dir : str
        relative path location to directory containing manually-retrieved historical records
        (default is "nmose_historical_data")

    comb_dir : str
        relative path location to directory where combined automatically-retrieved and manually-retrieved
        records will be saved
        (default is "nmose_combined_data")

    Exports
    ----------
    Microsoft Excel CSV file "nmose_diversion_sites.csv" containing site information of every site
    for which daily records were pulled

    1 additional CSV file for each site (e.g. "sjb_nm_aztec.csv") containing daily diversion records of that site


    Returns
    ----------
    None

    See Also
    --------

    Notes
    -----

    Examples
    --------
    Need to add    
    """

    print("downloading NMOSE diversion record data to directory {0}".format(dst_dir))
    if os.path.exists(dst_dir):
        pass
    else:
        os.mkdir(dst_dir)

    print("combining NMOSE diversion records into directory {0}".format(comb_dir))
    if os.path.exists(comb_dir):
        pass
    else:
        os.mkdir(comb_dir)

    # organize info for lookup table
    siteIds = []
    siteNames = []
    siteLat = []
    siteLong = []
    siteSource = []
    siteFiles = []
    siteUse = []
    siteStart = []
    siteEnd = []
    noFillYears = []
    shortID = []
    destinationCode = []
    destinationFlag = []

    # import table of NM diversion sites
    sites = pd.read_csv(sites_ifp)
    sites = sites.loc[sites["dataSource"] == "NMOSE"].copy()
    
    # retrieve NM OSE data
    minDate = "01/01/2011"
    maxDate = "10/01/2022"
    # endDate = date.today().strftime("%d/%m/%Y")

    website="http://meas.ose.state.nm.us/ReportProxy"
    params = {"type": "S",
              "sDate": minDate,
              "eDate": maxDate,
              "dischargeData": "davg",
              "rptFormat": "CSV",
              "sort": "asc"}

    for i, r in sites.loc[sites["nmoseID"].notnull()].iterrows():
        params["id"] = r.nmoseID
        siteFile = "{0}.csv".format(r.siteName)

        try:
            rr = requests.get(url=website, params=params)
            temp=StringIO(rr.text)
            df = pd.read_csv(temp, skiprows=1)
            num=df._get_numeric_data()
            num[num < 0] = 0
            df.rename(columns={'DischargeAvg (cfs)': 'discharge_cfs', "Day": "date"}, inplace=True)
            df.index=df.pop("date")
            df.to_csv(os.path.join(dst_dir, siteFile))
            print(r.siteName)
            siteIds.append(r.nmoseID)
            siteNames.append(r.siteName)
            siteUse.append(r.siteUse)
            siteLat.append(r.decLat)
            siteLong.append(r.decLong)
            siteSource.append(r.dataSource)
            siteFiles.append(siteFile)
            siteStart.append(r.startDate)
            siteEnd.append(r.endDate)
            noFillYears.append(r.no_fill_years)
            shortID.append(r.shortID)
            destinationCode.append(r.destinationCode)
            destinationFlag.append(r.destinationFlag)

        except:
            print("could not download or process data from NM OSE diversion site: {0}".format(r.siteName))
            pass

        if r.historicalRecord == "y":
            temp = pd.read_csv(os.path.join(hst_dir, "{0}.csv".format(r.siteName)))
            temp.loc[:,"date"] = pd.to_datetime(temp.loc[:,"date"])
            temp.index = temp.pop("date")
            temp = sp_df.join(temp,how="left")

            df.rename(columns={"discharge_cfs": "auto_cfs"}, inplace=True)

            df = temp.join(df, how="left")

            df.loc[:, 'discharge_cfs'] = df.loc[:, 'discharge_cfs'].fillna(df.loc[:, 'auto_cfs'])
            df.loc[:, "date"] = df.index.values
            df.index = df.pop("date")

            df.filter(['discharge_cfs']).to_csv(os.path.join(comb_dir, siteFile))
        else:
            df.to_csv(os.path.join(comb_dir, siteFile))
    
    # build and export diversion site lookup table for use in build_diversion_tabfiles()
    df = pd.DataFrame(data={"siteID": siteIds, "siteName": siteNames, "siteUse": siteUse,
                            "siteLat": siteLat, "siteLong": siteLong,
                            "siteSource": siteSource, "siteFile": siteFiles,
                            "startDate": siteStart, "endDate": siteEnd,
                            "noFillYears": noFillYears, "shortID": shortID,
                            "destinationCode": destinationCode, "destinationFlag": destinationFlag})
    
    df.loc[:, "siteFolder"] = os.path.split(dst_dir)[-1]
    df_out = format_sites_df(df)
    df_out.to_csv(os.path.join(dst_dir, "..", "nmose_diversion_sites.csv"))


def get_ut_diversion_data(dst_dir=os.path.join("..", "output", "utdwr_raw_data"),
                          sp_df=None,
                          sites_ifp=os.path.join("..", "input", "ucrb_diversion_master_table.csv"),
                          hst_dir=os.path.join("..", "input", "utdwr_historical_data"),
                          hst_dir1=os.path.join("..", "input", "cuwcd_historical_data"),
                          comb_dir=os.path.join("..", "output", "utdwr_combined_data")):
    """
    Function to pull diversion records from Utah Department of Water Resources website 

    Parameters
    ----------
    
    dst_dir : str
        relative path location to directory to save downloaded data
        (default is "utdwr_raw_data")

    sites_ifp : str
        relative path location to csv file containing all UCRB diversion sites.
        This function only attempts to pull data for records with "dataSource" attribute of "UTDWR"
        (default is "ucrb_diversion_master_table.csv")

    sp_df : pandas dataframe
        dataframe containing one record per day within period of interest, used for combining automatically
        retrieved data with manually retrieved data located in hst_dir directory

    hst_dir : str
        relative path location to directory containing manually-retrieved historical records
        (default is "utdwr_historical_data")

    comb_dir : str
        relative path location to directory where combined automatically-retrieved and manually-retrieved
        records will be saved
        (default is "utdwr_combined_data")

    Exports
    ----------
    Microsoft Excel CSV file "utdwr_diversion_sites.csv" containing site information of every site
    for which daily records were pulled

    1 additional CSV file for each site (e.g. "cms_ut_caineville_canal.csv") containing daily diversion records of that site


    Returns
    ----------
    None

    See Also
    --------

    Notes
    -----

    Examples
    --------
    Need to add    
    """    
    print("downloading UTDWR diversion record data to directory {0}".format(dst_dir))
    if os.path.exists(dst_dir):
        pass
        # shutil.rmtree(dst_dir)
        # print("existing diversion data directory found and will be replaced")
    else:
        os.mkdir(dst_dir)

    print("combining UTDWR diversion records into directory {0}".format(comb_dir))
    if os.path.exists(comb_dir):
        pass
        # shutil.rmtree(comb_dir)
        # print("existing directory for combined data found and will be replaced")
    else:
        os.mkdir(comb_dir)
    
    # organize info for lookup table
    siteIds = []
    siteNames = []
    siteLat = []
    siteLong = []
    siteSource = []
    siteFiles = []
    siteUse = []
    siteStart = []
    siteEnd = []
    noFillYears = []
    shortID = []
    destinationCode = []
    destinationFlag = []

    # import table of UTDWR diversion sites
    sites = pd.read_csv(sites_ifp)
    sites = sites.loc[sites["dataSource"] == "UTDWR"].copy()
    
    # retrieve UT DWR data
    for i, r in sites.loc[sites["utdwrID"].notnull()].iterrows():
        ID = r.utdwrID
        siteFile = "{0}.csv".format(r.siteName)
        
        today = date.today()
        Current_Date = today.strftime("%Y")
        URL = f"https://www.waterrights.utah.gov/cgi-bin/dvrtview.exe?STATION_ID={ID}&RECORD_YEAR={Current_Date}&Modinfo=Daily_Comma"
        
        try:
            rr = requests.get(URL)
            temp=StringIO(rr.text)
            temp1=temp.readlines()
            for line in temp1:
                if line.startswith("Daily comma delimited"):
                    URL_raw= re.findall('"([^"]*)"', line)
            URL_end="".join(map(str,URL_raw))
            URL_base = 'https://www.waterrights.utah.gov'
            URL_full = f"{URL_base}{URL_end}"
            rrr = requests.get(URL_full)
            temp_cd = StringIO(rrr.text)
            df = pd.read_csv(temp_cd)
            df.columns=["year", "month", "day", "discharge_cfs"]
            df.loc[:,"date"]=pd.to_datetime(df[['year','month', 'day']])
            df.drop(labels=df.columns.difference(["date", "discharge_cfs"]), axis=1, inplace=True)
            df.index=df.pop("date")

            df.to_csv(os.path.join(dst_dir, siteFile))

            if r.historicalRecord == "y":
                try:
                    temp = pd.read_csv(os.path.join(hst_dir, "{0}.csv".format(r.siteName)))
                except:
                    temp = pd.read_csv(os.path.join(hst_dir1, "{0}.csv".format(r.siteName)))

                temp.loc[:,"date"] = pd.to_datetime(temp.loc[:,"date"])
                temp.index = temp.pop("date")
                temp = sp_df.join(temp,how="left")

                for ii, rr in temp.loc[temp["monthly_cfsd"].notnull()].iterrows():
                    ix = temp.loc[(temp["year"] == rr.year) & (temp["month"] == rr.month)].index
                    temp.loc[ix, "discharge_cfs"] = rr.monthly_cfsd / monthrange(int(rr.year), int(rr.month))[1]

                df.rename(columns={"discharge_cfs": "auto_cfs"}, inplace=True)

                df = temp.join(df, how="left")

                df.loc[:, 'discharge_cfs'] = df.loc[:, 'discharge_cfs'].fillna(df.loc[:, 'auto_cfs'])
                df.loc[:, "date"] = df.index.values
                df.index = df.pop("date")

                df.filter(['discharge_cfs']).to_csv(os.path.join(comb_dir, siteFile))
            else:
                df.to_csv(os.path.join(comb_dir, siteFile))

            print(r.siteName)
            siteIds.append(r.utdwrID)
            siteNames.append(r.siteName)
            siteUse.append(r.siteUse)
            siteLat.append(r.decLat)
            siteLong.append(r.decLong)
            siteSource.append(r.dataSource)
            siteFiles.append(siteFile)
            siteStart.append(r.startDate)
            siteEnd.append(r.endDate)
            noFillYears.append(r.no_fill_years)
            shortID.append(r.shortID)
            destinationCode.append(r.destinationCode)
            destinationFlag.append(r.destinationFlag)

        except:
            print("could not download or process data from UTDWR diversion site: {0}".format(r.siteName))
            pass   
            
    # build and export diversion site lookup table for use in build_diversion_tabfiles()
    df = pd.DataFrame(data={"siteID": siteIds, "siteName": siteNames, "siteUse": siteUse,
                            "siteLat": siteLat, "siteLong": siteLong,
                            "siteSource": siteSource, "siteFile": siteFiles,
                            "startDate": siteStart, "endDate": siteEnd,
                            "noFillYears": noFillYears, "shortID": shortID,
                            "destinationCode": destinationCode, "destinationFlag": destinationFlag})
    
    df.loc[:, "siteFolder"] = os.path.split(comb_dir)[-1]
    df_out = format_sites_df(df)
    df_out.to_csv(os.path.join(dst_dir, "..", "utdwr_diversion_sites.csv"))


def get_wy_diversion_data(dst_dir=os.path.join("..", "output", "wyseo_raw_data"),
                          sp_df=None,
                          sites_ifp=os.path.join("..", "input", "ucrb_diversion_master_table.csv"),
                          hst_dir=os.path.join("..", "input", "wyseo_historical_data"),
                          comb_dir=os.path.join("..", "output", "wyseo_combined_data")):
    """
    Function to pull diversion records from Wyoming State Engineer's Office website 

    Parameters
    ----------
    
    dst_dir : str
        relative path location to directory to save downloaded data
        (default is "wyseo_raw_data")

    sites_ifp : str
        relative path location to csv file containing all UCRB diversion sites.
        This function only attempts to pull data for records with "dataSource" attribute of "WYSEO"
        (default is "ucrb_diversion_master_table.csv")

    sp_df : pandas dataframe
        dataframe containing one record per day within period of interest, used for combining automatically
        retrieved data with manually retrieved data located in hst_dir directory

    hst_dir : str
        relative path location to directory containing manually-retrieved historical records
        (default is "wyseo_historical_data")

    comb_dir : str
        relative path location to directory where combined automatically-retrieved and manually-retrieved
        records will be saved
        (default is "wyseo_combined_data")

    Exports
    ----------
    Microsoft Excel CSV file "wyseo_diversion_sites.csv" containing site information of every site
    for which daily records were pulled

    1 additional CSV file for each site (e.g. "raw_grb_wy_aurora_ditch.csv") containing daily diversion records of that site


    Returns
    ----------
    None

    See Also
    --------

    Notes
    -----

    Examples
    --------
    Need to add    
    """

    print("downloading WYSEO diversion record data to directory {0}".format(dst_dir))
    if os.path.exists(dst_dir):
        pass
        # shutil.rmtree(dst_dir)
        # print("existing diversion data directory found and will be replaced")
    else:
        os.mkdir(dst_dir)

    print("combining WYSEO diversion records into directory {0}".format(comb_dir))
    if os.path.exists(comb_dir):
        pass
        # shutil.rmtree(comb_dir)
        # print("existing directory for combined data found and will be replaced")
    else:
        os.mkdir(comb_dir)
    
    # organize info for lookup table
    siteIds = []
    siteNames = []
    siteLat = []
    siteLong = []
    siteSource = []
    siteFiles = []
    siteUse = []
    siteStart = []
    siteEnd = []
    noFillYears = []
    shortID = []
    destinationCode = []
    destinationFlag = []

    # import table of WYSEO diversion sites
    sites = pd.read_csv(sites_ifp)
    sites = sites.loc[sites["dataSource"] == "WYSEO"].copy()
    sites.loc[:, "wyseoID"] = sites.loc[:, "wyseoID"].astype(str)
    
    #retrieve streamgauge data and append to historical records
    for i, r in sites.iterrows():
        siteFile = "{0}.csv".format(r.siteName)
        if len(r.wyseoID) > 4:
            try:
                url = ("https://seoflow.wyo.gov/Export/BulkExport?DateRange=EntirePeriodOfRecord"
                       "&TimeZone=0&Calendar=CALENDARYEAR&Interval=Daily&Step=1&ExportFormat=csv"
                       "&TimeAligned=True&RoundData=False&IncludeGradeCodes=False&IncludeApprovalLevels=False"
                       "&IncludeInterpolationTypes=False&Datasets[0].DatasetName=Discharge.Discharge%40{0}"
                       "&Datasets[0].Calculation=Aggregate&Datasets[0].UnitId=208&_=".format(r.wyseoID))
                rr = requests.get(url)
                temp = StringIO(rr.text)
                df = pd.read_csv(temp, skiprows = 4)

                if len(df) == 0:
                    print("no data found for site {0} {1}".format(r.siteName, r.wyseoID))
                    print("attempting to pull data with 'Discharge (cfs)' instead of 'Discharge'")
                    rr = requests.get(url.replace("Discharge.Discharge", "Discharge.Discharge (cfs)"))
                    temp = StringIO(rr.text)
                    df = pd.read_csv(temp, skiprows = 4)
                    
                df.drop("End of Interval (UTC)", axis=1, inplace=True)
                df['date'] = pd.to_datetime(df['Start of Interval (UTC)']).dt.date
                df.drop("Start of Interval (UTC)", axis=1, inplace=True)
                df.rename(columns={'Average (ft^3/s)': 'discharge_cfs'}, inplace=True)
                df.index=df.pop("date")
                
                df.to_csv(os.path.join(dst_dir, "raw_{0}".format(siteFile)))

                if r.historicalRecord == "y":
                    temp = pd.read_csv(os.path.join(hst_dir, "{0}.csv".format(r.siteName)))
                    temp.loc[:,"date"] = pd.to_datetime(temp.loc[:,"date"])
                    temp.index = temp.pop("date")
                    temp = sp_df.join(temp,how="left")

                    df.rename(columns={"discharge_cfs": "auto_cfs"}, inplace=True)

                    df = temp.join(df, how="left")

                    df.loc[:, 'discharge_cfs'] = df.loc[:, 'discharge_cfs'].fillna(df.loc[:, 'auto_cfs'])
                    df.loc[:, "date"] = df.index.values
                    df.index = df.pop("date")

                    df.filter(['discharge_cfs']).to_csv(os.path.join(comb_dir, siteFile))
                else:
                    df.to_csv(os.path.join(comb_dir, siteFile))


                print("downloaded {0} diversion records for site {1} {2}".format(len(df), r.siteName, r.wyseoID))

            except:
                print("could not download or process diversion data for site {0} {1}".format(r.siteName, r.wyseoID))

        else:
            print("no online data available for {0}.{1}Copying data from {2} to {3}".format(r.siteName, newln, hst_dir, comb_dir))
            shutil.copy2(os.path.join(hst_dir, siteFile), os.path.join(comb_dir, siteFile))
            
        siteIds.append(r.wyseoID)
        siteNames.append(r.siteName)
        siteUse.append(r.siteUse)
        siteLat.append(r.decLat)
        siteLong.append(r.decLong)
        siteSource.append(r.dataSource)
        siteFiles.append(siteFile)
        siteStart.append(r.startDate)
        siteEnd.append(r.endDate)
        noFillYears.append(r.no_fill_years)
        shortID.append(r.shortID)
        destinationCode.append(r.destinationCode)
        destinationFlag.append(r.destinationFlag)

    # build and export diversion site lookup table for use in build_diversion_tabfiles()
    df = pd.DataFrame(data={"siteID": siteIds, "siteName": siteNames, "siteUse": siteUse,
                            "siteLat": siteLat, "siteLong": siteLong,
                            "siteSource": siteSource, "siteFile": siteFiles,
                            "startDate": siteStart, "endDate": siteEnd,
                            "noFillYears": noFillYears, "shortID": shortID,
                            "destinationCode": destinationCode, "destinationFlag": destinationFlag})
    
    df.loc[:, "siteFolder"] = os.path.split(comb_dir)[-1]
    df_out = format_sites_df(df)
    df_out.to_csv(os.path.join(dst_dir, "..", "wyseo_diversion_sites.csv"))


def get_usbr_diversion_data(dst_dir=os.path.join("..", "output", "usbr_raw_data"),
                            sites_ifp=os.path.join("..", "input", "ucrb_diversion_master_table.csv")):
    """
    Function to pull diversion records from U.S. Bureau of Reclamation website 

    Parameters
    ----------
    
    dst_dir : str
        relative path location to directory to save downloaded data
        (default is "usbr_raw_data")

    sites_ifp : str
        relative path location to csv file containing all UCRB diversion sites.
        This function only attempts to pull data for records with "dataSource" attribute of "USBR"
        (default is "ucrb_diversion_master_table.csv")

    Exports
    ----------
    Microsoft Excel CSV file "usbr_diversion_sites.csv" containing site information of every site
    for which daily records were pulled

    1 additional CSV file for each site (e.g. "grb_ut_tyzack_diversion.csv") containing daily diversion records of that site


    Returns
    ----------
    None

    See Also
    --------

    Notes
    -----

    Examples
    --------
    Need to add    
    """

    print("downloading USBR diversion record data to directory {0}".format(dst_dir))
    if os.path.exists(dst_dir):
        pass
        # shutil.rmtree(dst_dir)
        # print("existing diversion data directory found and will be replaced")
    else:
        os.mkdir(dst_dir)

    # import table of WYSEO diversion sites
    sites = pd.read_csv(sites_ifp)
    sites = sites.loc[sites["dataSource"] == "USBR"].copy()

    # organize info for lookup table
    siteIds = []
    siteNames = []
    siteLat = []
    siteLong = []
    siteSource = []
    siteFiles = []
    siteUse = []
    siteStart = []
    siteEnd = []
    noFillYears = []
    shortID = []
    destinationCode = []
    destinationFlag = []

    # retrieve USBR data
    for i, r in sites.loc[sites["usbrID"].notnull()].iterrows():
        url = "https://www.usbr.gov/uc/water/hydrodata/gage_data/{0}/csv/19.csv".format(int(r.usbrID))
        siteFile = "{0}.csv".format(r.siteName)
        try:
            rr = requests.get(url)
            temp = StringIO(rr.text)
            df = pd.read_csv(temp)
            df.columns = ["date", "discharge_cfs"]
            df.index=df.pop("date")
            df.to_csv(os.path.join(dst_dir, siteFile))
            print(r.siteName)
            siteIds.append(int(r.usbrID))
            siteNames.append(r.siteName)
            siteUse.append(r.siteUse)
            siteLat.append(r.decLat)
            siteLong.append(r.decLong)
            siteSource.append(r.dataSource)
            siteFiles.append(siteFile)
            siteStart.append(r.startDate)
            siteEnd.append(r.endDate)
            noFillYears.append(r.no_fill_years)
            shortID.append(r.shortID)
            destinationCode.append(r.destinationCode)
            destinationFlag.append(r.destinationFlag)

        except:
            print("could not download or process data from USBR diversion site: {0} {1}".format(r.siteName, int(r.usbrID)))
            pass

    # build and export diversion site lookup table for use in build_diversion_tabfiles()
    df = pd.DataFrame(data={"siteID": siteIds, "siteName": siteNames, "siteUse": siteUse,
                            "siteLat": siteLat, "siteLong": siteLong,
                            "siteSource": siteSource, "siteFile": siteFiles,
                            "startDate": siteStart, "endDate": siteEnd,
                            "noFillYears": noFillYears, "shortID": shortID,
                            "destinationCode": destinationCode, "destinationFlag": destinationFlag})

    df.loc[:, "siteFolder"] = os.path.split(dst_dir)[-1]
    df_out = format_sites_df(df)    
    df_out.to_csv(os.path.join(dst_dir, "..", "usbr_diversion_sites.csv"))


def get_usgs_diversion_data(dst_dir=os.path.join("..", "output", "usgs_raw_data"),
                            sites_ifp=os.path.join("..", "input", "ucrb_diversion_master_table.csv")):
    """
    Function to pull diversion records from U.S. Geological Survey website 

    Parameters
    ----------
    
    dst_dir : str
        relative path location to directory to save downloaded data
        (default is "usgs_raw_data")

    sites_ifp : str
        relative path location to csv file containing all UCRB diversion sites.
        This function only attempts to pull data for records with "dataSource" attribute of "USGS"
        (default is "ucrb_diversion_master_table.csv")

    Exports
    ----------
    Microsoft Excel CSV file "usgs_diversion_sites.csv" containing site information of every site
    for which daily records were pulled

    1 additional CSV file for each site (e.g. "grb_ut_ephraim_tunnel.csv") containing daily diversion records of that site


    Returns
    ----------
    None

    See Also
    --------

    Notes
    -----

    Examples
    --------
    Need to add    
    """

    print("downloading USGS diversion record data to directory {0}".format(dst_dir))
    if os.path.exists(dst_dir):
        pass
        # shutil.rmtree(dst_dir)
        # print("existing diversion data directory found and will be replaced")
    else:
        os.mkdir(dst_dir)

    # import table of USGS diversion sites
    sites = pd.read_csv(sites_ifp)
    sites = sites.loc[sites["dataSource"] == "USGS"].copy()

    # organize info for lookup table
    siteIds = []
    siteNames = []
    siteLat = []
    siteLong = []
    siteSource = []
    siteFiles = []
    siteUse = []
    siteStart = []
    siteEnd = []
    noFillYears = []
    shortID = []
    destinationCode = []
    destinationFlag = []

    # retrieve USGS data
    minDate = "1900-01-01"
    maxDate = "2022-10-01"

    website = "https://waterservices.usgs.gov/nwis/dv"
    params = {"format": "rdb",
              "startDT": minDate,
              "endDT": maxDate,
              "statCd": "00003",
              "parameterCd": "00060",
              "siteStatus": "all"}
    
    for i, r in sites.loc[sites["usgsID"].notnull()].iterrows():
        usgsID_numeric = int(r.usgsID.replace("USGS_", ""))
        params["sites"] = "{0:08d}".format(usgsID_numeric)
        siteFile = "{0}.csv".format(r.siteName)
        
        try:
            rr = requests.get(url=website, params=params)
            temp = StringIO(rr.text)
            line = '#'
            i = -1
            while line[0] == '#':
                line = temp.readline()
                i+=1
            temp = StringIO(rr.text) #reset text file for pandas to import
            df = pd.read_csv(temp,sep='\t',header=[i,i+1])
            cols = [0, 1, 4]
            df.drop(df.columns[cols], axis=1, inplace=True)
            df.columns = ["date","discharge_cfs"]
            df.index=df.pop("date")

            ix = df.loc[df["discharge_cfs"]=="Ssn"].index
            df.drop(ix, inplace=True)
            
            df.to_csv(os.path.join(dst_dir, siteFile))
            print("downloaded diversion data from USGS diversion site: {0}".format(r.siteName))

            siteIds.append(usgsID_numeric)
            siteNames.append(r.siteName)
            siteUse.append(r.siteUse)
            siteLat.append(r.decLat)
            siteLong.append(r.decLong)
            siteSource.append(r.dataSource)
            siteFiles.append(siteFile)
            siteStart.append(r.startDate)
            siteEnd.append(r.endDate)
            noFillYears.append(r.no_fill_years)
            shortID.append(r.shortID)
            destinationCode.append(r.destinationCode)
            destinationFlag.append(r.destinationFlag)

        except:
            print("could not download or process data from USGS diversion site: {0}".format(r.siteName, usgsID_numeric))
            pass

    # build and export diversion site lookup table for use in build_diversion_tabfiles()
    df = pd.DataFrame(data={"siteID": siteIds, "siteName": siteNames, "siteUse": siteUse,
                            "siteLat": siteLat, "siteLong": siteLong,
                            "siteSource": siteSource, "siteFile": siteFiles,
                            "startDate": siteStart, "endDate": siteEnd,
                            "noFillYears": noFillYears, "shortID": shortID,
                            "destinationCode": destinationCode, "destinationFlag": destinationFlag})

    df.loc[:, "siteFolder"] = os.path.split(dst_dir)[-1]
    df_out = format_sites_df(df)
    df_out.to_csv(os.path.join(dst_dir, "..", "usgs_diversion_sites.csv"))


def get_az_diversion_data(dst_dir=os.path.join("..", "output", "adwr_raw_data"),
                          sp_df=None,
                          sites_ifp=os.path.join("..", "input", "ucrb_diversion_master_table.csv")):
                          
    """
    Function to create estimated daily records based on water rights provided by Arizona Department of Water Resources 

    Parameters
    ----------
    
    dst_dir : str
        relative path location to directory to save downloaded data
        (default is "adwr_raw_data")

    sp_df : pandas dataframe
        dataframe containing one record per day within period of interest, used for creating
        a timeseries of estimated diversion records from an annual water right volume

    sites_ifp : str
        relative path location to csv file containing all UCRB diversion sites.
        This function only attempts to create time series for records with "dataSource" attribute of "ADWR"
        (default is "ucrb_diversion_master_table.csv")

    Exports
    ----------
    Microsoft Excel CSV file "adwr_diversion_sites.csv" containing site information of every site
    for which daily records were estimated

    1 additional CSV file for each site (e.g. "lcr_az_black_streamside.csv") containing daily diversion records of that site


    Returns
    ----------
    None

    See Also
    --------

    Notes
    -----

    Examples
    --------
    Need to add    
    """

    print("writing AZ diversion record data to directory {0}".format(dst_dir))
    if os.path.exists(dst_dir):
        pass
        # shutil.rmtree(dst_dir)
        # print("existing diversion data directory found and will be replaced")
    else:
        os.mkdir(dst_dir)

    # organize info for lookup table
    siteIds = []
    siteNames = []
    siteLat = []
    siteLong = []
    siteSource = []
    siteFiles = []
    siteUse = []
    siteStart = []
    siteEnd = []
    noFillYears = []
    shortID = []
    destinationCode = []
    destinationFlag = []

    # import table of ADWR diversion sites
    sites = pd.read_csv(sites_ifp)
    sites = sites.loc[sites["dataSource"] == "ADWR"].copy()

    for i, r in sites.iterrows():
        df = sp_df.copy()
        df.loc[:, "discharge_cfs"] = 0.0
        df.loc[df.loc[:, "month"].apply(lambda x: x in [6, 7, 8]), "discharge_cfs"] = r.adwrCFS
        df = df.filter(["discharge_cfs"])
        df.loc[:, "date"] = df.index.values
        df.index = df.pop("date")

        siteFile = "{0}.csv".format(r.siteName)
        df.to_csv(os.path.join(dst_dir, siteFile))
        print(r.siteName)

        siteIds.append(r.adwrID)
        siteNames.append(r.siteName)
        siteUse.append(r.siteUse)
        siteLat.append(r.decLat)
        siteLong.append(r.decLong)
        siteSource.append(r.dataSource)
        siteFiles.append(siteFile)
        siteStart.append(r.startDate)
        siteEnd.append(r.endDate)
        noFillYears.append(r.no_fill_years)
        shortID.append(r.shortID)
        destinationCode.append(r.destinationCode)
        destinationFlag.append(r.destinationFlag)
    
    # build and export diversion site lookup table for use in build_diversion_tabfiles()
    df = pd.DataFrame(data={"siteID": siteIds, "siteName": siteNames, "siteUse": siteUse,
                            "siteLat": siteLat, "siteLong": siteLong,
                            "siteSource": siteSource, "siteFile": siteFiles,
                            "startDate": siteStart, "endDate": siteEnd,
                            "noFillYears": noFillYears, "shortID": shortID,
                            "destinationCode": destinationCode, "destinationFlag": destinationFlag})

    df.loc[:, "siteFolder"] = os.path.split(dst_dir)[-1]
    df_out = format_sites_df(df)
    df_out.to_csv(os.path.join(dst_dir, "..", "adwr_diversion_sites.csv"))


def build_diversion_dataframe(src_dir=os.path.join("..", "output"),
                              dst_dir=os.path.join("..", "processed"),
                              mod_shp_ifp=os.path.join("..", "template_data", "ucrb_grid_shp", "ucrb.shp"),
                              str_shp_ifp=os.path.join("..", "template_data", "CP_streamsegs_shp", "CP_streamsegs.shp")):
    """
    Function to create a combined dataframe of all diversion sites with aquired or estimated daily records in UCRB

    Parameters
    ----------

    src_dir : str
        relative path location to directory containing site information tables for each data source
        (default is "../output")
    
    dst_dir : str
        relative path location to directory to save combined dataframe
        (default is "../processed")

    mod_shp_ifp : str
        relative path location to ESRI shapefile of structured model grid
        (default is None)

    str_shp_ifp : str
        relative path location to ESRI shapefile of stream network segments
        (default is None)

    Exports
    ----------
    Microsoft Excel CSV file "combined_diversion_sites.csv" containing site information of every site
    for which daily records were aquired or estimated


    Returns
    ----------
    None

    See Also
    --------

    Notes
    -----

    Examples
    --------
    Need to add    
    """

    print("{0}building diversion dataframe in directory {1}{0}".format(newln, dst_dir))
    assert os.path.exists(dst_dir)

    # combine diversion sites files into single dataframe
    ifps = ["cdss_diversion_sites.csv",
            "nmose_diversion_sites.csv",
            "usbr_diversion_sites.csv",
            "usgs_diversion_sites.csv",
            "utdwr_diversion_sites.csv",
            "wyseo_diversion_sites.csv"]

    print("using diversion sites files:{0}....{1}".format(newln, "{0}....".format(newln).join(ifps)))
    sdf = pd.concat([pd.read_csv(os.path.join(src_dir, ifp), index_col=0) for ifp in ifps], ignore_index=True)
    sdf.sort_values("shortID", inplace=True)
    sdf.reset_index(drop=True, inplace=True)

    # filter sites to active model cells:
    sdf.loc[:, "geometry"] = sdf.apply(lambda x: Point(x.utmX, x.utmY), axis=1)
    sdf = gp.GeoDataFrame(sdf, geometry="geometry", crs="epsg:{0}".format(model_epsg))

    # drop sites without location info and export to track issue
    temp = sdf.loc[sdf["utmY"].isnull()].copy()
    temp.to_csv(os.path.join(dst_dir, "diversion_sites_missing_loc.csv")) 
    # sdf = sdf.loc[~sdf["utmY"].isnull()].copy()
    sdf.drop(temp.index, inplace=True)

    sdf.loc[:, "tabFile"] = sdf.apply(lambda xx: "div_{0:04d}_{1}.tab".format(xx.shortID, xx.siteUse), axis=1)

    sdf.to_csv(os.path.join(dst_dir, "combined_diversion_sites.csv"))


def filter_diversion_records(src_dir=os.path.join("..", "output"),
                             sp_df=None,
                             dst_dir=os.path.join("..", "processed")):
    """
    Function to identify and reduce high-outlier values in daily time series of each diversion site

    Parameters
    ----------

    src_dir : str
        relative path location to directory containing sub-directories of raw or combined records for each data source
        (default is "../output")

    sp_df : pandas dataframe
        dataframe containing one record per day within period of interest
    
    dst_dir : str
        relative path location to directory containing combined dataframe, to which combined time series table and plots
        will be saved
        (default is "../processed")

    Exports
    ----------
    Adobe PDF file "diversion_plots_filtered_cfs.pdf"
        Plots of raw time series values for each site with potential outliers denoted

    Microsoft Excel CSV file "combined_diversion_records_raw_cfs.csv"
        Table includes raw pre-filtered time series values for each diversion site
        Table contains 1 row for each day in the period of interest, and 1 column for each diversion site.
        Empty cells indicate missing data 

    Microsoft Excel CSV file "combined_diversion_records_filtered_cfs.csv"
        Table includes filtered time series values for each diversion site
        Table contains 1 row for each day in the period of interest, and 1 column for each diversion site.
        Empty cells indicate missing data 

    Microsoft Excel CSV file "nodatafound_diversion_sites.csv"
        Table includes site information for diversion sites that are missing time series data

    Returns
    ----------
    None

    See Also
    --------

    Notes
    -----

    Examples
    --------
    Need to add    
    """

    print("{0}filtering diversion records{0}".format(newln))

    sdf = pd.read_csv(os.path.join(src_dir, "combined_diversion_sites.csv"))
    sdf['startDate'] = sdf['startDate'].replace('', pd.NaT)
    sdf['endDate'] = sdf['endDate'].replace('', pd.NaT)
    sdf['startDate'].fillna(pd.Timestamp('1900-01-01'), inplace=True)
    sdf['endDate'].fillna(pd.Timestamp('2100-01-01'), inplace=True)
    sdf.loc[:, "startDate"] = pd.to_datetime(sdf.loc[:, "startDate"])
    sdf.loc[:, "endDate"] = pd.to_datetime(sdf.loc[:, "endDate"])

    with PdfPages(os.path.join(dst_dir, "diversion_plots_filtered_cfs.pdf")) as pdf:
            ax_per_page = 8
            ncols = 2
            fig, axes = plt.subplots(int(ax_per_page / ncols), ncols, figsize=(8.5, 11), dpi=100)
            ax_count = 0
            pg_count = 0
            plt_count = 0

            ct = 0

            #mdf = sp_df.filter(["totim"]).copy()
            #mdf.loc[:, "totim"] = mdf.loc[:, "totim"].astype(int)
            #mdf.index = mdf.pop("totim")
            mdf = sp_df.filter([]).copy()

            odf = mdf.copy()

            problem_sites = []

            for i, r in sdf.iterrows():
                site = r.siteID
                sid = r.shortID
                nm = r.siteName
                use = r.siteUse
                # iseg = r.iseg
                ct += 1
                if ct % 100 == 0:
                    print("filtering site {0}/{1}".format(ct, len(sdf)))

                df = pd.read_csv(os.path.join(src_dir, r.siteFolder, r.siteFile))
                # workaround for currant creek bad values
                df.loc[:, "discharge_cfs"] = pd.to_numeric(df.loc[:, "discharge_cfs"], errors='coerce')
                # df.loc[:, "discharge_cfs"].fillna(0., inplace=True)
                df.loc[df["discharge_cfs"] < 0., "discharge_cfs"] = np.nan
                df = df.loc[df["discharge_cfs"].notnull()].copy()
                # df.loc[df["discharge_cfs"] < 0., "discharge_cfs"] = 0.

                if len(df.loc[df["discharge_cfs"] > 0.]) > 0:
                    ax = axes.flat[ax_count]

                    df.loc[:, "date"] = pd.DatetimeIndex(df.loc[:, "date"])
                    df.loc[:, "datetime"] = df.loc[:, "date"].copy()
                    # changed 9/14/2023
                    #df = df.groupby("date").first()
                    df = df.groupby("date").max()

                    # USE divfilter() FUNCTION TO IDENTIFY OUTLIERS
                    # df, aym, aymstd, aym_lim, aymstd_lim = divfilter(df)
                    df, aym = divfilter(df)

                    df.loc[:, "discharge_cfs_orig"] = df.loc[:, "discharge_cfs"].copy()
                    
                    # DECREASE OUTLIER VALUES TO THE MEDIAN OF THE ANNUAL MAXIMUM VALUES
                    df.loc[(df["mdflag"] == 1) & (df["sdflag"] == 1), "discharge_cfs"] = aym

                    ax.plot(df.loc[:, "datetime"], df.loc[:, "discharge_cfs_orig"], lw=0.5, c="k")
                    ax.plot(df.loc[:, "datetime"], df.loc[:, "discharge_cfs_orig"], marker=".", lw=0, markersize=3, rasterized=True)

                    ax.scatter(df.loc[df["mdflag"]==1, "datetime"], df.loc[df["mdflag"]==1, "discharge_cfs_orig"],
                               marker="^", facecolors="None", edgecolors="b", label="mdflag", alpha=0.5)
                    ax.scatter(df.loc[df["sdflag"]==1, "datetime"], df.loc[df["sdflag"]==1, "discharge_cfs_orig"],
                               marker=".", facecolors="None", edgecolors="r", label="sdflag", alpha=0.5)
                    # ax.hlines([aym, aym_lim, aymstd_lim], pd.to_datetime("1980-01-01"), pd.to_datetime("2025-01-01"),
                    #           colors=["b", "b", "r"], linewidths=0.4, linestyles=["solid", ":", ":"])
                    # ax.plot(df.loc[:, "datetime"], df.loc[:, "discharge_cfs"], lw=0.5, c="m")

                    ax.set_title("{0} shortID {1:04d}".format(nm, sid), loc="left")
                    ax.set_xlim([pd.to_datetime("1980-01-01"), pd.to_datetime("2025-01-01")])

                    # ymin = max([50 * np.ceil(max(grp.lev_va) / 50), 150])
                    # ymax = max([0, ymin - 150])
                    # ax.set_ylim([ymin, ymax])

                    ax.xaxis.set_major_locator(years10)
                    ax.xaxis.set_minor_locator(years1)

                    ax.xaxis.set_major_formatter(years_fmt)

                    # # enforce zero values before diversion start date and after diversion end date
                    # df.loc[(df["datetime"] < r.startDate) | (df["datetime"] > r.endDate), "discharge_cfs"] = 0.
                    # df.loc[(df["datetime"] < r.startDate) | (df["datetime"] > r.endDate), "discharge_cfs_orig"] = 0.

                    # join SP info
                    df.drop(["datetime", "year"], axis=1, inplace=True)
                    df = sp_df.join(df, how="outer")
                    df.loc[:, "datetime_copy"] = pd.to_datetime(df.index.values)

                    # enforce zero values before diversion start date and after diversion end date
                    df.loc[(df["datetime_copy"] < r.startDate) | (df["datetime_copy"] > r.endDate), "discharge_cfs"] = 0.
                    df.loc[(df["datetime_copy"] < r.startDate) | (df["datetime_copy"] > r.endDate), "discharge_cfs_orig"] = 0.

                    
                    # filter to days within model simulation (totim not empty)
                    df_out = df.loc[df["totim"].notnull()].filter(["totim", "discharge_cfs", "discharge_cfs_orig"]).copy()
                    # df_out.loc[:, "totim"] = df_out.loc[:, "totim"].astype(int)
                    # df_out.index = df_out.pop("totim")
                    colname = "div_{0:04d}_{1}".format(r.shortID, r.siteUse)
                    
                    df_out1 = df_out.filter(["discharge_cfs"]).copy()
                    df_out1.rename(columns={"discharge_cfs": colname}, inplace=True)
                    mdf = mdf.join(df_out1, how="left")

                    df_out2 = df_out.filter(["discharge_cfs_orig"]).copy()
                    df_out2.rename(columns={"discharge_cfs_orig": colname}, inplace=True)
                    odf = odf.join(df_out2, how="left")

                else:
                    print("no data for site {0} {1}".format(nm, site))
                    problem_sites.append(i)

                ax_count += 1
                plt_count += 1

                if ax_count >= ax_per_page:
                    if plt_count < len(sdf):

                        plt.tight_layout()
                        pdf.savefig()
                        plt.close(fig)

                        fig, axes = plt.subplots(int(ax_per_page / ncols), ncols, figsize=(8.5, 11), dpi=100)
                        ax_count = 0
                        pg_count += 1

                    else:
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close(fig)

                elif plt_count >= len(sdf):
                    for rem_ax in range(ax_count, ax_per_page):
                        axes.flat[rem_ax].set_xticks([])
                        axes.flat[rem_ax].set_yticks([])
                        axes.flat[rem_ax].axis("off")
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close(fig)

    odf.to_csv(os.path.join(dst_dir, "combined_diversion_records_raw_cfs.csv"))
    mdf.to_csv(os.path.join(dst_dir, "combined_diversion_records_filtered_cfs.csv"))

    if len(problem_sites) > 0:
        psdf = sdf.loc[problem_sites].copy()
        psdf.to_csv(os.path.join(src_dir, "nodatafound_diversion_sites.csv"))


def fill_missing_diversion_records(sp_df=None,
                                   src_dir=os.path.join("..", "output"),
                                   dst_dir=os.path.join("..", "processed"),
                                   fill_all_missing=False):
    """
    Function to interpolate and fill missing values in daily time series of each diversion site

    Parameters
    ----------

    sp_df : pandas dataframe
        dataframe containing one record per day within period of interest
        (default is None)
    
    dst_dir : str
        relative path location to directory containing combined dataframe, to which combined time series table and plots
        will be saved
        (default is "../processed")

    fill_all_missing : Bool
        Flag indicating whether to fill all missing data according to the scripted logic, or to limit filling to only
        years for which a given site has zero non-null records
        (default is False)

    Exports
    ----------
    Adobe PDF file "diversion_plots_filtered_filled_cfs_fill_years.pdf"
        Plots of filtered and filled time series values for each site

    Microsoft Excel CSV file "combined_diversion_records_filtered_filled_cfs_fill_years.csv"
        Table includes filtered and filled time series values for each diversion site
        Table contains 1 row for each day in the period of interest, and 1 column for each diversion site.

    Returns
    ----------
    None

    See Also
    --------

    Notes
    -----

    Examples
    --------
    Need to add    
    """
    print("{0}interpolating and filling missing diversion records{0}".format(newln))
    # load timestepping info
    # load diversion records and join to time stepping info
    df = pd.read_csv(os.path.join(dst_dir, "combined_diversion_records_filtered_cfs.csv"), index_col=0)
    df.loc[:, "datetime"] = pd.to_datetime(df.index.values)
    df.index = df.pop("datetime")
    divcols = df.columns

    df = sp_df.join(df, how="outer", on="datetime")
    df.index = df.pop("datetime")
    
    # load diversion site information
    sdf = pd.read_csv(os.path.join(src_dir, "combined_diversion_sites.csv"), index_col=0)
    sdf.index = sdf.loc[:, "shortID"].apply(lambda x: "{0:04d}".format(x))
    sdf.loc[:, "divcol"] = sdf.loc[:, "tabFile"].apply(lambda x: x.replace(".tab", ""))

    # fill nan values with zero in years designated no-fill
    temp = sdf.loc[sdf["noFillYears"].notnull()].copy()
    for i,r in temp.iterrows():
        yearlist = [int(x) for x in r.noFillYears.split(";")]
        for yr in yearlist:
            df.loc[(df["year"] == yr) & (df[r.divcol].isnull()), r.divcol] = 0.
    
    # identify years for each site with no records
    ydf = df.groupby("year").max()
    ydf = ydf.filter(divcols).copy()
    ydf = sp_df.join(ydf, how="left", on="year")

    # calculate monthly mean diversions
    mdf = df.groupby("month").mean()
    mdf.fillna(0., inplace=True)
    mdf = mdf.filter(divcols).copy()
    mdf = sp_df.join(mdf, how="left", on="month")

    # interpolate linearly between records no more than 90 days apart
    df1 = df.copy()
    mf = df.copy()
    
    print("interpolating over data gaps")
    i = 0
    for div in divcols:
        i += 1
        df1.loc[:, div] = df1.loc[:, div].interpolate(method="linear", axis=0, on="datetime",
                                                      limit=90, limit_direction="backward", limit_area="inside")
        mf.loc[:, div] = gaplengths(df.loc[:, div])
        if i % 100 == 0:
            print("interpolating site {0}/{1}".format(i, len(divcols)))
    
    print("masking interpolated output to gaps less than 90 days in length")
    a = np.array(df1[divcols].values.tolist())
    m = np.array(mf[divcols].values.tolist())

    df1[divcols] = np.where(m >= 90., np.nan, a).tolist()

    # fill missing daily values with monthly mean values in years with zero daily values
    a = np.array(df1[divcols].values.tolist())
    b = np.array(mdf[divcols].values.tolist())

    m = np.array(ydf[divcols].values.tolist())

    df1[divcols] = np.where(np.isnan(m), b, a).tolist()

    if fill_all_missing:
        # fill all remaining missing daily values with monthly mean values
        df1.fillna(mdf, inplace=True)
        ftag = "fill_all"
    else:
        ftag = "fill_years"
    
    # fill remaining nan values with 0.
    df1.fillna(0., inplace=True)

    df1.filter(divcols).to_csv(os.path.join(dst_dir, "combined_diversion_records_filtered_filled_cfs_{0}.csv".format(ftag)))
    
    # plot filled timeseries
    with PdfPages(os.path.join(dst_dir, "diversion_plots_filtered_filled_cfs_{0}.pdf".format(ftag))) as pdf:
            ax_per_page = 8
            ncols = 2
            fig, axes = plt.subplots(int(ax_per_page / ncols), ncols, figsize=(8.5, 11), dpi=100)
            ax_count = 0
            pg_count = 0
            plt_count = 0

            ct = 0

            for div in divcols:
                ax = axes.flat[ax_count]

                sid = div.split("_")[-2]
                site = sdf.loc[sid,"siteID"]
                nm = sdf.loc[sid,"siteName"]
                use = sdf.loc[sid,"siteUse"]
                # iseg = sdf.loc[sid,"iseg"]
                
                ct += 1
                if ct % 100 == 0:
                    print("plotting site {0} of {1}".format(ct, len(divcols)))

                
                # ax.plot(df.loc[:, "datetime"], df.loc[:, div], marker=".", lw=0, markersize=3, rasterized=True)
                # ax.plot(df1.loc[:, "datetime"], df1.loc[:, div], c="m", lw=0.25)
                ax.plot(df.index, df.loc[:, div], marker=".", lw=0, markersize=3, rasterized=True)
                ax.plot(df1.index, df1.loc[:, div], c="m", lw=0.25)

                ax.set_title("{0} shortID {1}".format(nm, sid), loc="left")
                ax.set_xlim([pd.to_datetime("1980-01-01"), pd.to_datetime("2025-01-01")])
                #ax.set_xlim([pd.to_datetime("1980-01-01"), pd.to_datetime("1990-01-01")])

                ax.xaxis.set_major_locator(years10)
                ax.xaxis.set_minor_locator(years1)

                ax.xaxis.set_major_formatter(years_fmt)

                ax_count += 1
                plt_count += 1

                if ax_count >= ax_per_page:             
                    if plt_count < len(sdf):

                        plt.tight_layout()
                        pdf.savefig()
                        plt.close(fig)

                        fig, axes = plt.subplots(int(ax_per_page / ncols), ncols, figsize=(8.5, 11), dpi=100)
                        ax_count = 0
                        pg_count += 1

                    else:
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close(fig)

                elif plt_count >= len(sdf):
                    for rem_ax in range(ax_count, ax_per_page):
                        axes.flat[rem_ax].set_xticks([])
                        axes.flat[rem_ax].set_yticks([])
                        axes.flat[rem_ax].axis("off")
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close(fig)


if __name__ == "__main__":

    div_in = os.path.join("..", "input")
    div_out = os.path.join("..", "intermediate_script_output")
    div_proc = os.path.join("..", "processed")

    for idir in [div_out, div_proc]:
        if os.path.exists(idir):
            pass
        else:
            os.mkdir(idir)
    try:
        api_token_file = os.path.join(div_in, "api_token.txt")
        with open(api_token_file) as f:
            data = f.read()
        userdict = json.loads(data)
        apiKey = userdict["apiKey"]
        
        if apiKey == "copy_API_token_string_within_these_quotes":
            print("{0}WARNING: you must enter a valid CDSS API token in order to download all of the CDSS diversion data".format(newln))
            print("{0}request an API token at https://dwr.state.co.us/rest/get/help".format(newln))
            print("{0}Copy 'api_token_TEMPLATE.txt' to 'api_token.txt' in directory 'input'".format(newln))
            print("{0}Enter API token string into 'api_token.txt' in directory 'input'".format(newln))
            print("{0}Delete commented lines at beginning of 'api_token.txt'".format(newln))
            print("{0}'api_token.txt' in directory 'input' is not synced to project git repository".format(newln))
            apiKey = None
    except:
        apiKey = None

    diversion_master_table = os.path.join(div_in, "ucrb_diversion_master_table.csv")
    
    #spdf = build_spdf(start="19791231", end="20220930", spfreq="M", tsfreq="D")
    spdf = build_spdf(start="19800101", end="20220930", spfreq="M", tsfreq="D")
    spdf1 = spdf.copy()
    spdf1.loc[:, "datetime"] = spdf1.index.values.copy()
    spdf1.loc[:, "datetime"] = pd.to_datetime(spdf1.loc[:, "datetime"])
    spdf1.index = spdf1.pop("totim")

    # specify EPSG code for projection from LatLong
    model_epsg = 26912
    
    # PULL AVAILABLE RECORDS FROM INTERNET AND COMBINE WITH HISTORICAL RECORDS AS NEEDED
    
    # CDSS IMPOSES A MAXIMUM DAILY REQUEST ALLOWANCE. MUST HAVE API KEY WITH SUFFICIENT PRIVILEGE TO PULL CDSS DATA
    get_cdss_diversion_data(dst_dir=os.path.join(div_out, "cdss_raw_data"),
                            sites_ifp=diversion_master_table,
                            apiKey=apiKey)
    
    #Occasionally, the New Mexico request fails on the first try. The script will try to re-pull NM diversion data for a specified number of times.
    nm_try = 1
    nm_max_try = 5
    while nm_try <= nm_max_try:
        try:
            get_nmose_diversion_data(dst_dir=os.path.join(div_out, "nmose_raw_data"),
                                    sp_df=spdf,
                                    sites_ifp=diversion_master_table,
                                    hst_dir=os.path.join(div_in, "nmose_historical_data"),
                                    comb_dir=os.path.join(div_out, "nmose_combined_data"))
            nm_try = nm_max_try + 1
        except:
            nm_try += 1
            print("\n....NMOSE retrieval failed. Attempting again ({0}/{1})\n".format(nm_try, nm_max_try))

    get_ut_diversion_data(dst_dir=os.path.join(div_out, "utdwr_raw_data"),
                          sp_df=spdf,
                          sites_ifp=diversion_master_table,
                          hst_dir=os.path.join(div_in, "utdwr_historical_data"),
                          hst_dir1=os.path.join(div_in, "cuwcd_historical_data"),
                          comb_dir=os.path.join(div_out, "utdwr_combined_data"))

    get_wy_diversion_data(dst_dir=os.path.join(div_out, "wyseo_raw_data"),
                          sp_df=spdf,
                          sites_ifp=diversion_master_table,
                          hst_dir=os.path.join(div_in, "wyseo_historical_data"),
                          comb_dir=os.path.join(div_out, "wyseo_combined_data"))

    get_usbr_diversion_data(dst_dir=os.path.join(div_out, "usbr_raw_data"),
                            sites_ifp=diversion_master_table)

    get_usgs_diversion_data(dst_dir=os.path.join(div_out, "usgs_raw_data"),
                            sites_ifp=diversion_master_table)

    #get_az_diversion_data(dst_dir=os.path.join(div_out, "adwr_raw_data"),
                          #sp_df=spdf,
                          #sites_ifp=diversion_master_table)

    # COMBINE DIVERSION SITES INFO
    build_diversion_dataframe(src_dir=div_out,
                             dst_dir=div_out,
                             mod_shp_ifp=None,
                             str_shp_ifp=None)

    # FILTER OUTLIER RECORDS FROM TIME SERIES DATA
    filter_diversion_records(src_dir=div_out,
                             sp_df=spdf,
                             dst_dir=div_proc)

    # FILL MISSING RECORDS
    fill_missing_diversion_records(sp_df=spdf1,
                                   src_dir=div_out,
                                   dst_dir=div_proc,
                                   fill_all_missing=False)