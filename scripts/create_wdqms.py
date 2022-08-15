import sys
import os
import argparse
import pandas as pd
import numpy as np
from netCDF4 import Dataset
from pathlib import Path


def temp_2_saturation_specific_humidity(pres, tsen):
    """
    Uses pressure and temperature arrays to calculate saturation
    specific humidity. 
    Args:
        pres: (array) array of pressure obs
        tsen: (array) array of temperature obs in Celsius
    Returns:
        qsat_array: (array) corresponding calculated sat. spec. humidity
    """
    
    ttp   = 2.7316e2      # temperature at h2o triple point (K) 
    psat  = 6.1078e1      # pressure at h2o triple point  (Pa)
    cvap  = 1.8460e3      # specific heat of h2o vapor (J/kg/K)
    csol  = 2.1060e3      # specific heat of solid h2o (ice)(J/kg/K)
    hvap  = 2.5000e6      # latent heat of h2o condensation (J/kg)
    hfus  = 3.3358e5      # latent heat of h2o fusion (J/kg)
    rd    = 2.8705e2       
    rv    = 4.6150e2
    cv    = 7.1760e2
    cliq  = 4.1855e3

    dldt  = cvap-cliq
    dldti = cvap-csol
    hsub  = hvap+hfus
    tmix  = ttp-20.
    xa    = -(dldt/rv)
    xai   = -(dldti/rv)
    xb    = xa+hvap/(rv*ttp)
    xbi   = xai+hsub/(rv*ttp)
    eps   = rd/rv
    omeps = 1.0-eps

    tdry = tsen+ttp
    tdry = np.array([1.0e-8 if np.abs(t) < 1.0e-8 else t for t in tdry])

    tr = ttp/tdry

    qsat_array = []

    # Loop through temperatures and appropriate indexes to solve qsat
    for idx, t in enumerate(tdry):
        # Get correct estmax and es values based on conditions
        if t >= ttp:
            estmax = psat * (tr[idx]**xa) * np.exp(xb*(1.0-tr[idx]))
            es = estmax
        elif t < tmix:
            estmax = psat * (tr[idx]**xa) * np.exp(xbi*(1.0-tr[idx]))
            es = estmax
        else:
            w = (t-tmix) / (ttp-tmix)
            estmax = w * psat * (tr[idx]**xa) * np.exp(xb*(1.0-tr[idx])) \
                     + (1.0-w) * psat * (tr[idx]**xai) * np.exp(xbi*(1.0-tr[idx]))

            es  = w * psat * (tr[idx]**xa) * np.exp(xb*(1.0-tr[idx])) \
                  + (1.0-w) * psat * (tr[idx]**xai) * np.exp(xbi*(1.0-tr[idx]))

        pw = pres[idx]
        esmax = pw

        esmax = np.min([esmax, estmax])
        es2 = np.min([es, esmax])

        qsat = eps * es2 / ((pw*10.0) - (omeps*es2))
        qsat2 = qsat*1e6

        qsat_array.append(qsat2)
    
    return np.array(qsat_array)


def grab_netcdf_data(file, var):
    """
    Opens and grabs data based on column name.
    Args:
        file : (str) netCDF GSI file
        var  : (str) the variable to be extracted
    Returns:
        data : (array) values from the specified variable
    """
    
    with Dataset(file, mode='r') as f:
        # Station_ID and Observation_Class variables need
        # to be converted from byte string to string
        if var == 'Datetime':
            data = f.date_time
        
        elif var in ['Station_ID', 'Observation_Class']:
            data = f.variables[var][:]
            data = [i.tobytes(fill_value='        ', order='C')
                    for i in data]
            data = np.array(
                [''.join(i.decode('UTF-8', 'ignore').split())
                 for i in data])
            
        # Grab variables with only 'nobs' dimension
        elif len(f.variables[var].shape) == 1:
            data = f.variables[var][:]
            
    return data

def read_gsi_diag(file):
    """
    Reads the data from the conventional diagnostic file during
    initialization into a pandas dataframe.
    Args:
        file : (str) netCDF GSI diagnostic file
    Returns:
        df : (dataframe) pandas dataframe populated with data from
             netCDF GSI diagnostic file
    """
    filename = os.path.splitext(Path(file).stem)[0]
    obs_type = filename.split('_')[1]
    variable = filename.split('_')[2]
    ftype = filename.split('_')[-1]
    
    print(variable)
    
    variable_ids = {
        'ps': 110,
        'q': 29,
        't': 2,
        'u': 3,
        'v': 4
    }
    
    df_dict = {}

    column_list = ['Station_ID', 'Observation_Class', 'Observation_Type',
                   'Latitude', 'Longitude', 'Pressure', 'Time', 'Prep_QC_Mark', 'Prep_Use_Flag',
                   'Analysis_Use_Flag', 'Observation', 'Obs_Minus_Forecast_adjusted']
    
    #Grab datetime from file
    datetime = grab_netcdf_data(file, 'Datetime')
    
    if variable == 'uv':
        for wtype in ['u','v']:
            df_dict[wtype] = {}
            for col in column_list:
                col = f'{wtype}_'+col if col == 'Observation' else col
                col = f'{wtype}_'+col if col == 'Obs_Minus_Forecast_adjusted' else col
                data = grab_netcdf_data(file, col)
                df_dict[wtype][col] = data
    
        # Need to separate the u and v dataframes to concatenate them
        u_df = pd.DataFrame(df_dict['u'])
        u_df = u_df.rename({'Observation_Class': 'var_id',
                            'u_Observation': 'Observation',
                            'u_Obs_Minus_Forecast_adjusted': 'Obs_Minus_Forecast_adjusted'},
                           axis=1)
        u_df['var_id'] = variable_ids['u']

        v_df = pd.DataFrame(df_dict['v'])
        v_df = v_df.rename({'Observation_Class': 'var_id',
                            'v_Observation': 'Observation',
                            'v_Obs_Minus_Forecast_adjusted': 'Obs_Minus_Forecast_adjusted'},
                           axis=1)
        v_df['var_id'] = variable_ids['v']

        df = pd.concat([u_df, v_df])
        

    else:
        for col in column_list:
            data = grab_netcdf_data(file, col)
            df_dict[col] = data
        
        df = pd.DataFrame(df_dict)
        df = df.rename({'Observation_Class': 'var_id'}, axis=1)
        df['var_id'] = variable_ids[variable]
    
    # Add datetime column to dataframe
    df['Datetime'] = datetime
    
    # Subtract longitudes > 180 by 360 to be negative
    df.loc[df['Longitude'] > 180, 'Longitude'] -= 360

    return df

def create_status_flag(df):
    """
    Create Status Flag based on the values from Prep_QC_Mark,
    Prep_Use_Flag, and Analysis_Use_Flag.
    Args:
        df: (df) pandas dataframe populated with data from GSI
            diagnostic files
    Returns:
        df: (df) the same dataframe read in with a new column: 'StatusFlag'
    """
    # Create 'StatusFlag' column and fill with nans
    df['StatusFlag'] = np.nan

    # Obs used by GSI, Status_Flag=0
    df.loc[(df['Prep_QC_Mark'] <= 8) & (df['Analysis_Use_Flag'] == 1), 'StatusFlag'] = 0

    # Obs rejected by GSI, Status_Flag=0
    df.loc[(df['Prep_QC_Mark'] <= 8) & (df['Analysis_Use_Flag'] == -1), 'StatusFlag'] = 2

    # Obs never used by GSI, Status_Flag=3
    df.loc[(df['Prep_QC_Mark'] > 8) & (df['Prep_Use_Flag'] >= 100), 'StatusFlag'] = 3

    # Obs is flagged for non-use by the analysis, Status_Flag=3
    df.loc[df['Prep_QC_Mark'] >= 15, 'StatusFlag'] = 3

    # Obs rejected by SDM or CQCPROF, Status_Flag=7
    df.loc[(df['Prep_QC_Mark'] >= 12) & (df['Prep_QC_Mark'] <= 14), 'StatusFlag'] = 7

    # Fill those that do not fit a condition with -999
    df.loc[df['StatusFlag'].isnull(), 'StatusFlag'] = -999

    return df


def get_datetimes(df):
    """
    Use 'Datetime' and 'Time' columns to create new datetime and
    separate into new columns: 'YYYYMMDD' and 'HHMMSS'
    Args:
        df : (df) pandas dataframe populated with data from GSI
             diagnostic files
    Returns:
        df: (df) the same dataframe read in with new columns:
            'YYYYMMDD' and 'HHMMSS'
    """
    # Convert 'Datetime' column from str to datetime
    dates = pd.to_datetime(df_total['Datetime'], format='%Y%m%d%H')
    # Converts 'Time' column to time delta in hours
    hrs = pd.to_timedelta(df_total['Time'], unit='hours')
    # Actual datetime of ob adding datetime and timedelta in hours
    new_dt = dates+hrs

    df['yyyymmdd'] = new_dt.dt.strftime('%Y%m%d')
    df['HHMMSS'] = new_dt.dt.strftime('%H%M%S')

    return df


def genqsat(df):
    """
    Calculates new background departure values for specific humidity (q)
    by calculating saturation specific humidity from corresponding temperature
    and pressure values. 
    
    bg_dep = (q_obs/qsat_obs)-(q_ges/qsat_ges)
    
    q_obs : measured q obs
    qsat_obs : calculated saturation q
    q_ges : q_obs minus q background error from GSI diagnostic file
    qsat_ges : calculated saturation q using temperature obs minus
               temperature background error from GSI diagnostic file
    
    Args:
        df : (df) pandas dataframe populated with data from GSI
             diagnostic files
    Returns:
        df: (df) the same dataframe read in with new background
            departure values
    """
    # Create two dataframes, one for q vales and one for t values
    q_df = df_total.loc[(df_total['var_id'] == 58.)]
    t_df = df_total.loc[(df_total['var_id'] == 39.)]
    
    # Find where stations are the same
    stn_ids = np.intersect1d(t_df.Station_ID, q_df.Station_ID)

    # loop through stations, calculate saturation specific humidity,
    # and replace background departure values from NetCDF file with 
    # new ones calculated using the temperature obs and best temperature guess
    for stn in stn_ids:
        t_tmp = t_df.loc[(t_df['Station_ID'] == stn)]
        q_tmp = q_df.loc[(q_df['Station_ID'] == stn)]

        t_tmp = t_tmp.loc[(np.in1d(t_tmp['Time'], q_tmp['Time']) ) &
                          (np.in1d(t_tmp['Pressure'], q_tmp['Pressure'])) &
                          (np.in1d(t_tmp['Latitude'], q_tmp['Latitude'])) &
                          (np.in1d(t_tmp['Longitude'], q_tmp['Longitude']))]
        
        q_obs = q_tmp['Observation'].to_numpy() * 1.0e6
        q_ges = (q_tmp['Observation'].to_numpy() - 
                 q_tmp['Obs_Minus_Forecast_adjusted'].to_numpy()) * 1.0e6
        t_obs = t_tmp['Observation'].to_numpy() - 273.16
        t_ges = (t_tmp['Observation'].to_numpy() - 
                 t_tmp['Obs_Minus_Forecast_adjusted'].to_numpy()) -273.16
        pressure = q_tmp['Pressure'].to_numpy()

        qsat_obs = temp_2_saturation_specific_humidity(pressure, t_obs)
        qsat_ges = temp_2_saturation_specific_humidity(pressure, t_ges)
        
        bg_dep = (q_obs/qsat_obs)-(q_ges/qsat_ges)
        
        # Replace the current background departure with the new calculated one
        df.loc[(df['var_id'] == 58.) & (df['Station_ID'] == stn),
               'Obs_Minus_Forecast_adjusted'] = bg_dep
        
    return df

def create_sondes_df(df):
    """
    Create dataframe for sondes.
    """
    stn_ids = df['Station_ID'].unique()

    df_list = []
    
    # Loop through stations and create individual dataframes
    # that grabs average stats from surface, troposphere, and
    # stratosphere
    for stn in stn_ids:
        d = {
            'var_id': [],
            'Mean_Bg_dep': [],
            'Std_Bg_dep': [],
            'Levels': [],
            'LastRepLevel': []
        }

        surf_lat = None
        surf_lon = None

        # Temporary dataframe of specific station data
        tmp = df.loc[df['Station_ID'] == stn]

        # Add pressure info if available
        if 110 in tmp['var_id'].unique():
            d['var_id'].append(110)
            d['Mean_Bg_dep'].append(tmp['Obs_Minus_Forecast_adjusted'].loc[tmp['var_id'] == 110].values[0])
            d['Std_Bg_dep'].append(0) # cannot compute std w/ one value so set to 0
            d['Levels'].append('Surf')
            d['LastRepLevel'].append(-999.99)
            d['StatusFlag'].append(tmp['StatusFlag'].loc[tmp['var_id'] == 110].values[0])

            #surface lat and lon if exists
            surf_lat = tmp['Latitude'].loc[tmp['var_id'] == 110].values[0]
            surf_lon = tmp['Longitude'].loc[tmp['var_id'] == 110].values[0]

        # Get unique variable ID's and remove 110 (surface pressure)
        var_ids = sorted(tmp['var_id'].unique())
        var_ids.remove(110) if 110 in var_ids else var_ids

        for var in var_ids:
            # Surface
            if (110 in tmp['var_id'].unique() and
                var in tmp['var_id'].loc[tmp['Pressure'] == tmp['Pressure'].max()].unique()):

                surf_tmp = tmp.loc[(tmp['Pressure'] == tmp['Pressure'].max()) &
                                   (tmp['var_id'] == var)]

                surf_omf = surf_tmp['Obs_Minus_Forecast_adjusted'].values[0]
                surf_std = 0 # cannot compute std w/ one value so set to 0

                # If at least one ob is used, we report the lowest Status Flag.
                # Although it does not represent the whole column, it is what is
                # required by the WDQMS team.
                status_flag = surf_tmp['StatusFlag'].min()

                d['var_id'].append(var)
                d['Mean_Bg_dep'].append(round(surf_omf, 4))
                d['Std_Bg_dep'].append(round(surf_std, 4))
                d['Levels'].append('Surf')
                d['LastRepLevel'].append(-999.99)
                d['StatusFlag'].append(status_flag)

            # Troposphere
            trop_tmp = tmp.loc[(tmp['var_id'] == var) &
                               (tmp['Pressure'] >= 100)]

            if len(trop_tmp) > 0:
                trop_avg_omf = trop_tmp['Obs_Minus_Forecast_adjusted'].mean()
                trop_std_omf = trop_tmp['Obs_Minus_Forecast_adjusted'].std()
                # Get lowest p for entire atmosphere
                last_ps_rep = tmp['Pressure'].min()

                # If at least one ob is used, we report the lowest Status Flag.
                # Although it does not represent the whole column, it is what is
                # required by the WDQMS team.
                status_flag = trop_tmp['StatusFlag'].min()

                d['var_id'].append(var)
                d['Mean_Bg_dep'].append(round(trop_avg_omf, 4))
                d['Std_Bg_dep'].append(round(trop_std_omf, 4))
                d['Levels'].append('Trop')
                d['LastRepLevel'].append(last_ps_rep)
                d['StatusFlag'].append(status_flag)

            # Stratosphere
            stra_tmp = tmp.loc[(tmp['var_id'] == var) & 
                               (tmp['Pressure'] < 100)]

            if len(stra_tmp) > 0:
                stra_avg_omf = stra_tmp['Obs_Minus_Forecast_adjusted'].mean()
                stra_std_omf = stra_tmp['Obs_Minus_Forecast_adjusted'].std()
                # Get lowest p for entire atmosphere
                last_ps_rep = tmp['Pressure'].min()

                # If at least one ob is used, we report the lowest Status Flag.
                # Although it does not represent the whole column, it is what is
                # required by the WDQMS team.
                status_flag = stra_tmp['StatusFlag'].min()

                d['var_id'].append(var)
                d['Mean_Bg_dep'].append(round(stra_avg_omf, 4))
                d['Std_Bg_dep'].append(round(stra_std_omf, 4))
                d['Levels'].append('Stra')
                d['LastRepLevel'].append(last_ps_rep)
                d['StatusFlag'].append(status_flag)

        sub_df = pd.DataFrame.from_dict(d)
        sub_df['Station_id'] = stn
        # Add lats and lons
        lat = surf_lat if surf_lat else tmp['Latitude'].value_counts().index[0]
        lon = surf_lon if surf_lon else tmp['Longitude'].value_counts().index[0]
        sub_df['latitude'] = lat
        sub_df['Longitude'] = lon
        # add datetime
        str_datetime = str(tmp['Datetime'].values[0])
        sub_df['yyyymmdd'] = str_datetime[:-2]
        sub_df['HHMMSS'] =  str_datetime[-2:] + '0000'

        df_list.append(sub_df)

    df = pd.concat(df_list)
    df['Centre_id'] = 'NCEP'
    df['CodeType'] = 999
    
    # Ordered columns
    cols = ['Station_id', 'yyyymmdd', 'HHMMSS', 'latitude', 'Longitude',
            'StatusFlag', 'Centre_id', 'var_id', 'Mean_Bg_dep', 'Std_Bg_dep',
            'Levels', 'LastRepLevel', 'CodeType']
    
    df = df[cols]
    df = df.reset_index(drop=True)
    
    return df


def create_conv_df(df):
    """
    Create dataframe for conventional data.
    """
    # Add center_id
    df['Centre_id'] = 'NCEP'
    df['CodeType'] = 999

    # Remove unnecessary columns
    df.drop(['Observation_Type', 'Pressure', 'Time', 'Prep_QC_Mark',
             'Prep_Use_Flag', 'Analysis_Use_Flag', 'Datetime'],
             axis=1, inplace=True)

    #Rename columns
    df = df.rename({'Obs_Minus_Forecast_adjusted': 'Bg_dep',
                    'Latitude': 'latitude',
                    'Station_ID': 'Station_id'}, axis=1)

    #ordered columns
    cols = ['Station_id', 'yyyymmdd', 'HHMMSS', 'latitude', 'Longitude',
            'StatusFlag', 'Centre_id', 'var_id', 'Bg_dep', 'CodeType']

    df = df[cols]
    df = df.reset_index(drop=True)

    return df


def df_to_csv(df, wdqms_type, datetime, outdir):
    """
    Produce output .csv file from dataframe.
    """
    
    # Write dataframe to .csv
    date = datetime[:-2]
    cycle = datetime[-2:] 
    
    hr_range = {
        '00': ['21', '03'],
        '06': ['03', '09'],
        '12': ['09', '15'],
        '18': ['15', '21']
    }
    
    filename = f'{outdir}/NCEP_{wdqms_type}_{date}_{cycle}.csv'
    
    f = open(filename, 'a')
    f.write("# TYPE=TEMP\n")
    f.write(f"#An_Date= {date}\n")
    f.write(f"#An_time= {cycle}\n")
    f.write(f"#An_range=[ {hr_range[cycle][0]} to {hr_range[cycle][-1]} )\n")
    f.write("#StatusFlag: 0(Used);1(Not Used);2(Rejected by DA);" \
            "3(Never Used by DA);4(Data Thinned);5(Rejected before DA);" \
            "6(Alternative Used);7(Quality Issue);8(Other Reason);9(No content)\n")
    df.to_csv(f, index=False)
    f.close()
    
    return filename
    
def wdqms(inputfiles, wdqms_type, outdir):
    """
    Main driver function to produce WDQMS output .csv files.
    """
    # Create dataframes from GSI diag files
    df_list = []

    for file in inputfiles:
        df = read_gsi_diag(file)
        df_list.append(df)

    df_total = pd.concat(df_list)

    # Grab actual datetimes from datetime + timedelta
    df_total = get_datetimes(df_total)

    # Adjust relative humidity data
    df_total = genqsat(df_total)

    # Add Status Flag column
    df_total = create_status_flag(df_total)

    # Sort by Station ID
    df_total = df_total.sort_values('Station_ID')

    if wdqms_type == 'SYNOP':
        output_df = create_conv_df(total_df)
    elif wdqms_type == 'TEMP':
        output_df = create_sondes_df(total_df)
    else:
        print("Please enter valid WDQMS type. Exiting ...")
        sys.exit()

    # Get str datetime
    datetime = inputfile[0].split('/')[-1].split('.')[-2]

    out_filename = df_to_csv(output_df, wdqms_type, datetime, outdir)

    print("Success! Output file saved to: {out_filename}")
    print("Exiting ...")
    sys.exit()


if __name__ == "__main__":

    # Parse command line
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_list", nargs='+', default=[],
                    help="List of input GSI diagnostic files")
    ap.add_argument("-t", "--type",
                    help="WDQMS file type (SYNOP or TEMP)")
    ap.add_argument("-o", "--outdir",
                    help="Out directory where files will be saved")

    myargs = ap.parse_args()

    wdqms(myargs.input_list, myargs.type, myargs.outdir)
