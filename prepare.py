import pandas as pd
from datetime import datetime

def prepare(df):
    df = df.drop(columns=['Crash ID', 'Average Daily Traffic Amount', 'Average Daily Traffic Year', 'Highway Number',
                     'Surface Condition', 'Surface Type', 'Vehicle Hit and Run Flag',
                     'Person Blood Alcohol Content Test Result', 'Person Drug Test Result',
                     'Crash Month', 'Crash Year', 'Number of Lanes', 'License Plate State',
                     'Driver License Type', 'Crash Severity', 'Unit Death Count', 'Unit Not Injured Count',
                     'Person Death Count', 'Person Injury Severity', 'Person Not Injured Count', 'Person Total Injury Count'])
    
    df = df.rename(str.lower, axis='columns')

    df = df.rename(columns = {'crash date':'date', 'day of week':'day',
                         'weather condition':'weather', 'vehicle color':'color', 'vehicle make':'make',
                         'person age':'age', 'person ethnicity':'ethnicity', 'crash death count':'deceased', 'crash time':'time',
                         'crash total injury count':'injured', 'driver license state':'dl_state', 'person gender':'gender',
                         'person helmet':'helmet', 'person type':'driver'})
    
    df = df.replace(to_replace = {'5 - DRIVER OF MOTORCYCLE TYPE VEHICLE', '6 - PASSENGER/OCCUPANT ON MOTORCYCLE TYPE VEHICLE'},
                value = {'driver', 'passenger'})

    df = df.replace(to_replace = {'1 - NOT WORN', '99 - UNKNOWN IF WORN'}, value = 'not worn')
    
    df = df.replace(to_replace = {'2 - WORN, DAMAGED', '3 - WORN, NOT DAMAGED', '4 - WORN, UNK DAMAGE'}, value = 'worn')

    df = df.replace(to_replace = {'2 - FEMALE'}, value = 'female')

    df = df.replace(to_replace = {'1 - MALE'}, value = 'male')

    df['gender'] = df['gender'].replace({'99 - UNKNOWN':'male', 'No Data':'male'})
    df['ethnicity'] = df['ethnicity'].replace({'No Data':'98 - OTHER'})
    df['age'] = df['age'].replace({'No Data':37})
    df['make'] = df['make'].replace({'No Data':'unknown', 'UNKNOWN':'unknown'})
    df['color'] = df['color'].replace({'No Data':'99 - UNKNOWN'})
    df['dl_state'] = df['dl_state'].replace({'No Data':'UN - UNKNOWN'})
    df['injured'] = df['injured'].replace({2:1, 3:1, 4:1, 5:1, 6:1, 7:1})
    df['deceased'] = df['deceased'].replace({2:0})
    df['latitude'] = df['latitude'].replace({'No Data':0})
    df['longitude'] = df['longitude'].replace({'No Data':0})
    df['weather'] = df['weather'].str[4:]
    df['weather'] = df['weather'].str.strip()
    df['dl_state'] = df['dl_state'].str[5:]
    df['color'] = df['color'].str[6:]
    df['ethnicity'] = df['ethnicity'].str[4:]
    df['ethnicity'] = df['ethnicity'].str.strip()
    df.time = df.time.astype(str)
    df['time'] = df['time'].apply(lambda x: x.zfill(4))
    df.time = df.time.str[:2] + ':' + df.time.str[-2:]
    df['crash_date'] = df['date'] +' '+ df['time']
    df['crash_date'] = pd.to_datetime(df['crash_date'])
    df = df.set_index('crash_date').sort_index()
    convert_dict_int = {'age': int, 'deceased':int, 'injured':int, 'latitude':float, 'longitude':float}
    df = df.astype(convert_dict_int)
#get_dummies creates a seperate df of booleans for the identified columns below. Cleaning for the decission tree.
    dummy_df = pd.get_dummies(df[['driver', 'helmet', 'gender']], dummy_na=False, drop_first=[True, True])
    df = df.drop(columns=['driver', 'helmet', 'gender', 'date', 'time'])
    df = pd.concat([df, dummy_df], axis=1)
    df = df.rename(columns = {'driver_passenger':'driver', 'helmet_worn':'helmet', 'gender_male':'male'})
    convert_dict_int = {'driver': int, 'helmet':int, 'male':int}
    df = df.astype(convert_dict_int)

    return df

