import os
import requests
import shutil

import numpy as np
import pandas as pd

from tqdm import tqdm
from maad import sound, util

# ------------------------------- DEFINE PATHS AND GLOBAL VARIABLES ------------------------------

TIME_PERIODS = {
    'YYYYMMDD_HHMMSS YYYYMMDD_HHMMSS': '<path to folder>',
    'YYYYMMDD_HHMMSS YYYYMMDD_HHMMSS': '<path to folder>',
    'YYYYMMDD_HHMMSS YYYYMMDD_HHMMSS': '<path to folder>'
}

# August 10th, 12:20 until August 26th, 08:02 (UNINSTALL) 
# '20250810_122000 20250826_070200': r'/Volumes/X9 Pro Ness/Raw Audio/Ness/Ness_A_Aug',
# August 31st, 8:54 (INSTALL) until September 25th, 18:00 (start of low-battery degraded audio)
#'20250831_095400 20250925_180000': r'/Volumes/X9 Pro Ness/Raw Audio/Ness/Ness_A_Sep',
# September 30th, 12:26 (INSTALL) until October 12th, 00:00 (start of low-battery degraded audio)
# '20250930_132600 20251012_000000': r'/Volumes/X9 Pro Ness/Raw Audio/Ness/Ness_A_Oct',

# Rain stations names
all_relevant_stations = ['Inshes','Flichity','Corrimony','Urquhart'] 

# Date and time of the start of the recording period 
SEPA_START = "2025-08-10T00:00:00"

# Length of the recording period
SEPA_PERIOD = "70D"

# Water Level station name
STATION_NAME = "Ness-side" 

# Folder weather data should be saved to
WEATHER_FOLDER = '<path>'

RAIN_PATH = os.path.join(WEATHER_FOLDER, 'df_rain.csv')
WATER_LEVEL_PATH = os.path.join(WEATHER_FOLDER, 'df_water_level.csv')
FLOW_PATH = os.path.join(WEATHER_FOLDER, 'df_flow.csv')

accessKey = "<SEPA API Access Key>"

# Path were signal-to-noise database should be written to
DATABASE_PATH = r'<path>.csv'

# Path were the snr database and environmental data should be merged
DATABASE_ENV_PATH = r'<path>.csv'

# List of stations where nonzero rain levels will result in a soundfile not being considered
station_list = ['Inshes','Urquhart','Flichity']

# Downsampling variable (if set to 100, one file every 100 will be considered. 
# Useful for testing, but should systematically be set to 1 for analysis)
ONE_EVERY_MANY = 1

# Frequency filters (in Hz)
LOW = 100
HIGH = 20000 

# Folders where the selected files should be stored
DEST_PATH_ABOVE_THRESH = r'<path>/Above_Thresh'
DEST_PATH_BELOW_THRESH = r'<path>/Below_Thresh'

# Name of the hydrophone, used to discriminate between identical filenames
HYDROPHONE_ID = 'HYDRO_A'

# Which variable should be chosen to select the top files. 'snr' for signal-to-noise ratio, 'enr' for total energy (deprecated)
criterion = 'snr'

# If True, will not consider files that match with ,nearby rain records
no_rain = True

# Fraction of top snr files which should be computed. In the associated paper, only 
# sound files below threshold (1.75m water level) were analysed, and the selection
# above threshold was only computed as a sanity check
fraction_top_above_thresh = 0.05
fraction_top_below_thresh = 0.03

water_level_thresh = 1.75

# Number of files to include in each batch (for manual labelling in Raven for example)
selection_size = 50

# ------------------------------- HELPER FUNCTIONS -----------------------------------------------

# Evaluate whether the timestamp of a file name following the expected YYYYMMDD_HHMMSS.wav 
# format is between two defined time bounds
def isFilenameInBounds(filename, bounds):
    start, finish = bounds.split()

    time_start_date, time_start_time = start.split("_")
    time_start = int(time_start_date) * 1000000 + int(time_start_time)

    time_finish_date, time_finish_time = finish.split("_")
    time_finish = int(time_finish_date) * 1000000 + int(time_finish_time)

    filename_date, filename_time = filename.split(".")[0].split("_")
    time_filename = int(filename_date) * 1000000 + int(filename_time)

    return (time_filename >= time_start and time_filename <= time_finish)

# Evaluate whether the timestamp of a file name following the expected YYYYMMDD_HHMMSS.wav 
# format is comprised in at least one of a list of time periods
def isFilenameInPeriods(filename, time_periods):
    for bounds in time_periods:
        if isFilenameInBounds(filename, bounds):
            return True, time_periods[bounds]
    return False, None

def getDateTimeFromFilename(filename):
    return pd.to_datetime(filename, format='%Y%m%d_%H%M%S.WAV', utc = True)

# Convert a datetime into its nearest future quarter_of_hour equivalent
def findClosestQuarter(datetime):

    minute = datetime.minute
    quarter = 0
    for i in [45, 30, 15]:
        if minute < i:
            quarter = i

    datetime = datetime.replace(minute = quarter)
    datetime = datetime.replace(second = 0)

    if quarter == 0:
        datetime = datetime + pd.DateOffset(hours=1)

    return datetime

# return whether there is rain at one of the specified rain stations at a certain time
def isThereRain(rain_df, effective_station_list, filename):
    datetime = pd.to_datetime(filename, format='%Y%m%d_%H%M%S.WAV', utc = True)
    row = rain_df.loc[rain_df['datetime'] == findClosestQuarter(datetime)]
    for station in effective_station_list:
        if row['rain_'+station].item():
            return True
    return False

def filtered_spectral_snr(Sxx_power, fn, low, high):

    # Define low and high filter frequencies indices for ablation of the spectrograms
    idx_last_low = np.searchsorted(fn, low, side='left')
    idx_first_high = np.searchsorted(fn, high, side='right')

    # Ablate the spectrogram below and above the provided low and high frequencies
    Sxx_power = Sxx_power[idx_last_low:idx_first_high,:]

    # Compute total soundfile energy (dB)
    ENRf_per_bin = sound.avg_power_spectro(Sxx_power)
    ENRf = util.power2dB(sum(ENRf_per_bin))

    # Compute the background noise spectrum (median power per frequency, averaged over time)
    _, noise_profile = sound.remove_background_along_axis(Sxx_power, mode='median',axis=1) 
    noise_profile = util.running_mean(noise_profile,N=5)
    BGNf = util.power2dB(sum(noise_profile))

    # Define snr as the difference between total and background power
    SNRf = ENRf - BGNf 

    return ENRf, BGNf, SNRf, Sxx_power, fn[idx_last_low:idx_first_high]

def getTimeFromDateTime(datetime):
    datetime = datetime.replace(year = 2000)
    datetime = datetime.replace(month = 1)
    datetime = datetime.replace(day = 1)
    return datetime

# Between-dataframe linear interpolation (useful for minute-wise estimation
# of water levels between 15-min discrete timestamps)
def interpolateValue(target_time, df, field_name):
    df = df.sort_values('datetime').reset_index(drop=True)

    before = df[df['datetime'] < target_time].tail(1)
    after = df[df['datetime'] >= target_time].head(1)

    if before.empty or after.empty:
        raise ValueError("Target datetime is outside the range of the dataframe.")

    t0, b0 = before.iloc[0]['datetime'], before.iloc[0][field_name]
    t1, b1 = after.iloc[0]['datetime'], after.iloc[0][field_name]

    t = target_time.timestamp()
    t0_ts = t0.timestamp()
    t1_ts = t1.timestamp()

    value_interp = b0 + (b1 - b0) * ((t - t0_ts) / (t1_ts - t0_ts))

    return value_interp

# Get the signal-to-noise ratio of a list of filenames 
def computeFullRow(time_periods, filename, rain_df, station_list):
    _, input_dir = isFilenameInPeriods(filename, time_periods)
    path = os.path.join(input_dir, filename)
    s, fs = sound.load(path)

    Sxx_power, tn, fn,_ = sound.spectrogram(s, fs)
    enr, bgn, snr, filtered_Sxx_power, filtered_fn = filtered_spectral_snr(Sxx_power, fn, LOW, HIGH)

    datetime = getDateTimeFromFilename(filename)

    is_there_rain = isThereRain(rain_df, station_list, filename)

    return [input_dir, filename, datetime, enr, bgn, snr, is_there_rain]


# ------------------------------- MAIN SCRIPT ----------------------------------------------------

def createEnvironmentalDatasets():

    is_rain_available = True
    is_waterlevel_available = True
    is_flow_available = True

    # specify target, key, header - N.B. set accessKey to value issued to you
    tokenURL = 'https://timeseries.sepa.org.uk/KiWebPortal/rest/auth/oidcServer/token'
    authHeaders = { 'Authorization' : 'Basic ' + accessKey }

    # POST token request to return response object
    responseToken= requests.post(tokenURL, headers = authHeaders, data = 'grant_type=client_credentials')

    # retrieve access token from response object access
    accessToken = responseToken.json()['access_token']

    # specify request string, and header
    headDict = {'Authorization':'Bearer ' + accessToken}


    ## RAIN

    try:

        # Obtain the IDs of the requested series 
        ID_s = {}

        for station_name in all_relevant_stations:
            requestURL = f'https://timeseries.sepa.org.uk/KiWIS/KiWIS?service=kisters&type=queryServices&datasource=0&request=getTimeseriesList&station_name={station_name}&format=json'

            # GET data request as response object
            responseData = requests.get(requestURL, headers = headDict)
            data = responseData.json()

            # format responseData as a pandas DataFrame with the first row as header
            data = pd.DataFrame(data)
            data.columns = data.iloc[0]
            data = data[1:]
            
            for i, row in data.iterrows():
                if row['ts_name'] == '15minute.Total':
                    ID_s[station_name] = str(row['ts_id'])

        print(ID_s)

        rainfall_all_stations = []

        # Obtain and save the time series
        for name, id in ID_s.items():
            requestURL = f'https://timeseries.sepa.org.uk/KiWIS/KiWIS?service=kisters&type=queryServices&datasource=0&request=getTimeseriesValues&ts_id={id}&from={SEPA_START}&period=P{SEPA_PERIOD}&format=json'

            responseData = requests.get(requestURL, headers = headDict)
            data = responseData.json()

            values = data[0]['data']
            columns = data[0]['columns']

            datetimes = np.array([row[0] for row in values])
            rain = np.array([row[1] for row in values])

            column_name = f'rain_{name}'

            df_rain = pd.DataFrame({'datetime': datetimes, column_name: rain})
            rainfall_all_stations.append(df_rain)

        df_rainfall_all_stations = rainfall_all_stations[0]

        for df in rainfall_all_stations[1:]:
            df_rainfall_all_stations = df_rainfall_all_stations.merge(df, on = 'datetime')
        
    except:
        is_rain_available = False


    ## WATER LEVEL

    try:
        # specify request string, and header
        requestURL = f'https://timeseries.sepa.org.uk/KiWIS/KiWIS?service=kisters&type=queryServices&datasource=0&request=getTimeseriesList&station_name={STATION_NAME}&format=json'

        # GET data request as response object
        responseData = requests.get(requestURL, headers = headDict)
        data = responseData.json()

        # format responseData as a pandas DataFrame with the first row as header
        data = pd.DataFrame(data)
        data.columns = data.iloc[0]
        data = data[1:]

        water_level_id = ''
        for i, row in data.iterrows():
            if row['ts_name'] == '15minute' and row['parametertype_name']=='S':
                water_level_id = str(row['ts_id'])

        requestURL = f'https://timeseries.sepa.org.uk/KiWIS/KiWIS?service=kisters&type=queryServices&datasource=0&request=getTimeseriesValues&ts_id={water_level_id}&from={SEPA_START}&period=P{SEPA_PERIOD}&format=json'

        # GET data request as response object
        responseData = requests.get(requestURL, headers = headDict)
        data = responseData.json()

        values = data[0]['data']
        columns = data[0]['columns']

        # put the datetimes into a numpy array
        datetimes = np.array([row[0] for row in values])
        df_water = np.array([row[1] for row in values])

        df_water = pd.DataFrame({'datetime': datetimes, 'water_level': df_water})

        df_water['datetime'] = pd.to_datetime(df_water['datetime'], utc=True)  # result is tz-aware (UTC)
        df_water = df_water.set_index('datetime')
        df_water.sort_index(inplace=True)
    except:
        is_waterlevel_available = False

    ## FLOW LEVEL

    try:
        # specify request string, and header
        requestURL = f'https://timeseries.sepa.org.uk/KiWIS/KiWIS?service=kisters&type=queryServices&datasource=0&request=getTimeseriesList&station_name={STATION_NAME}&format=json'

        # GET data request as response object
        responseData = requests.get(requestURL, headers = headDict)
        data = responseData.json()

        # format responseData as a pandas DataFrame with the first row as header
        data = pd.DataFrame(data)
        data.columns = data.iloc[0]
        data = data[1:]

        flow_level_id = ''
        for i, row in data.iterrows():
            if row['ts_name'] == '15minute' and row['parametertype_name']=='Q':
                flow_level_id = str(row['ts_id'])

        requestURL = f'https://timeseries.sepa.org.uk/KiWIS/KiWIS?service=kisters&type=queryServices&datasource=0&request=getTimeseriesValues&ts_id={flow_level_id}&from={SEPA_START}&period=P{SEPA_PERIOD}&format=json'

        # GET data request as response object
        responseData = requests.get(requestURL, headers = headDict)
        data = responseData.json()

        values = data[0]['data']
        columns = data[0]['columns']

        # put the datetimes into a numpy array
        datetimes = np.array([row[0] for row in values])
        df_flow = np.array([row[1] for row in values])

        df_flow = pd.DataFrame({'datetime': datetimes, 'flow': df_flow})

        df_flow['datetime'] = pd.to_datetime(df_flow['datetime'], utc=True)  # result is tz-aware (UTC)
        df_flow = df_flow.set_index('datetime')
        df_flow.sort_index(inplace=True)
    except:
        is_flow_available = False

    ## CREATE CSV FILES 

    if is_rain_available:
        df_rainfall_all_stations.to_csv(RAIN_PATH)

    if is_waterlevel_available:
        df_water.to_csv(WATER_LEVEL_PATH)

    if is_flow_available:
        df_flow.to_csv(FLOW_PATH)

def makeSNRDatabase():

    df_stations_rainfall = pd.read_csv(RAIN_PATH)
    df_stations_rainfall['datetime'] = pd.to_datetime(df_stations_rainfall['datetime'], utc=True)

    filelist = []

    for input_dir in TIME_PERIODS.values():

        dir_list = [f for f in os.listdir(input_dir) if (f.endswith('.WAV') and f.startswith('20'))]

        for filename in tqdm(dir_list, total = len(dir_list)):
            in_bounds, _ = isFilenameInPeriods(filename, TIME_PERIODS)
            if in_bounds:
                filelist.append(filename)

    database = []
    exception_dict = {} 

    sorted_filelist = sorted(filelist)

    print('Computing per-file SNR')
    for filename in tqdm(sorted_filelist[::ONE_EVERY_MANY]):
        try:
            full_row = computeFullRow(TIME_PERIODS, filename, df_stations_rainfall, station_list)
            database.append(full_row)
        except Exception as e:
            exception_dict[filename] = e

    print('Finished computing per-file SNR')

    df = pd.DataFrame(database, columns=['dir', 'filename', 'datetime', 'enr', 'bgn', 'snr', 'is_there_rain'])
    df.to_csv(DATABASE_PATH)

    waterlevel_df = pd.read_csv(WATER_LEVEL_PATH) 
    waterlevel_df["datetime"] = pd.to_datetime(waterlevel_df['datetime'], utc=True)

    flow_df = pd.read_csv(FLOW_PATH) 
    flow_df["datetime"] = pd.to_datetime(flow_df['datetime'], utc=True)

    rain_df = pd.read_csv(RAIN_PATH) 
    rain_df["datetime"] = pd.to_datetime(rain_df['datetime'], utc=True)

    snr_dataset = pd.read_csv(DATABASE_PATH) 
    snr_dataset["datetime"] = pd.to_datetime(snr_dataset['datetime'], utc=True)
    snr_dataset["time_of_day"] = snr_dataset["datetime"].apply(getTimeFromDateTime)

    print('Successfully processed all dataframes — Starting Interpolation')

    dataset_list = [waterlevel_df, rain_df, flow_df]

    for index, df in enumerate(dataset_list):
        for col in df.columns.values.tolist():
            if ('datetime' in col) or ('unnamed' in col.lower()):
                continue
            snr_dataset[col] = snr_dataset["datetime"].apply(interpolateValue, args = (df, col))
            print('Processed column ',col,' of dataset ',index+1,' out of ',len(dataset_list))
    print('Finished computing merged database')

    snr_dataset.to_csv(DATABASE_ENV_PATH)
    print('Saved merged database')

    database_df = pd.read_csv(DATABASE_ENV_PATH)

    if no_rain:
        database_df_usable = database_df[~database_df['is_there_rain']].copy()
    else:
        database_df_usable = database_df.copy()

    # The original code had a strict '>' sign instead of '>='. It was fixed here.
    # Only the fraction below threshold was used in the ICUA paper, this fix has no incidence on the results.
    database_above_thresh = database_df_usable[database_df_usable['water_level'] >= water_level_thresh] 
    database_above_thresh_usable = database_above_thresh.sort_values(by='snr', ascending=False)

    database_below_thresh = database_df_usable[database_df_usable['water_level'] < water_level_thresh]
    database_below_thresh_usable = database_below_thresh.sort_values(by='snr', ascending=False)

    # Select top x files and randomize them
    database_above_head = database_above_thresh_usable.head(int(fraction_top_above_thresh * len(database_above_thresh_usable)))
    database_above_sample = database_above_head.sample(frac=1).reset_index()

    database_below_head = database_below_thresh_usable.head(int(fraction_top_below_thresh * len(database_below_thresh_usable)))
    database_below_sample = database_below_head.sample(frac=1).reset_index()

    file_total = 0
    for folder in TIME_PERIODS.values():
        file_total += len([f for f in os.listdir(folder) if f.startswith('20') and f.lower().endswith('.wav')])

    print('Total files: ', file_total)
    print('After deployment-related removal: ',len(sorted_filelist),'. Nb of lost files',file_total - len(sorted_filelist))
    print('After downsampling by a factor ',ONE_EVERY_MANY,': ',len(database_df),'. Nb of lost files: ', len(sorted_filelist) - len(database_df))
    print('Without rain: ', len(database_df_usable), '. Nb of lost files',len(database_df) - len(database_df_usable))
    print('Below 1.75m: ', len(database_below_thresh), '. Nb of lost files',len(database_df_usable) - len(database_below_thresh))
    print('Sampling the top ',np.round(100 * fraction_top_below_thresh, 2),'% .Files remaining: ',int(fraction_top_below_thresh * len(database_below_thresh)))

    above_filename_dict = {}
    below_filename_dict = {}

    for i, row in database_above_sample.iterrows():

        dict_num = i//selection_size
        if dict_num in above_filename_dict:
            above_filename_dict[dict_num].append(row['filename'])
        else:
            above_filename_dict[dict_num] = [row['filename']]

    for i, row in database_below_sample.iterrows():

        dict_num = i//selection_size
        if dict_num in below_filename_dict:
            below_filename_dict[dict_num].append(row['filename'])
        else:
            below_filename_dict[dict_num] = [row['filename']]

    # Creating a different folder for each batch, above and below threshold
    for N in tqdm(range(len(above_filename_dict))):

        path = DEST_PATH_ABOVE_THRESH + "/Split_"+str(N)
        os.mkdir(path)
        sound_files = above_filename_dict[N]

        for i, filename in enumerate(sound_files):
            random_indexed_filename = str(N * selection_size + i)+HYDROPHONE_ID+'.'+filename
            try:
                _, input_dir = isFilenameInPeriods(filename, TIME_PERIODS)
                shutil.copyfile(os.path.join(input_dir, filename), os.path.join(path, random_indexed_filename))
            except Exception as e:
                print(e)

    for N in tqdm(range(len(below_filename_dict))):

        path = DEST_PATH_BELOW_THRESH + "/Split_"+str(N)
        os.mkdir(path)
        sound_files = below_filename_dict[N]

        for i, filename in enumerate(sound_files):
            random_indexed_filename = str(N * selection_size + i)+HYDROPHONE_ID+'.'+filename
            try:
                _, input_dir = isFilenameInPeriods(filename, TIME_PERIODS)
                shutil.copyfile(os.path.join(input_dir, filename), os.path.join(path, random_indexed_filename))
            except Exception as e:
                print(e)


if __name__ == '__main__':
    createEnvironmentalDatasets()
    makeSNRDatabase()