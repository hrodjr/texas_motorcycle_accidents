import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from scipy import stats

##########ACQUIRE########################
#acquires csv from local file
def get_data():
    df = pd.read_csv('texasma.csv')
    return df

##########PREPARE########################
#prepares data for analysis
def cleaned(df):
#Dropped columns on initial inspection of the dataset
    df = df.drop(columns=['Crash ID', 'Average Daily Traffic Amount', 'Average Daily Traffic Year', 'Highway Number',
                     'Surface Condition', 'Surface Type', 'Vehicle Hit and Run Flag',
                     'Person Blood Alcohol Content Test Result', 'Person Drug Test Result',
                     'Crash Month', 'Crash Year', 'Number of Lanes', 'License Plate State',
                     'Driver License Type', 'Crash Severity', 'Unit Death Count', 'Unit Not Injured Count',
                     'Person Death Count', 'Person Injury Severity', 'Person Not Injured Count', 'Person Total Injury Count'])
#sets all column nemase to lowercas    
    df = df.rename(str.lower, axis='columns')
#column rename
    df = df.rename(columns = {'crash date':'date', 'day of week':'day',
                         'weather condition':'weather', 'vehicle color':'color', 'vehicle make':'make',
                         'person age':'age', 'person ethnicity':'ethnicity', 'crash death count':'deceased', 'crash time':'time',
                         'crash total injury count':'injured', 'driver license state':'dl_state', 'person gender':'gender',
                         'person helmet':'helmet', 'person type':'driver'})
#replaces values in preparation for encoding    
    df = df.replace(to_replace = {'5 - DRIVER OF MOTORCYCLE TYPE VEHICLE', '6 - PASSENGER/OCCUPANT ON MOTORCYCLE TYPE VEHICLE'},
                value = {'driver', 'passenger'})
    df = df.replace(to_replace = {'1 - NOT WORN', '99 - UNKNOWN IF WORN'}, value = 'not worn')
    df = df.replace(to_replace = {'2 - WORN, DAMAGED', '3 - WORN, NOT DAMAGED', '4 - WORN, UNK DAMAGE'}, value = 'worn')
    df = df.replace(to_replace = {'2 - FEMALE'}, value = 'female')
    df = df.replace(to_replace = {'1 - MALE'}, value = 'male')
#replaces values by column
    df['gender'] = df['gender'].replace({'99 - UNKNOWN':'male', 'No Data':'male'})
    df['ethnicity'] = df['ethnicity'].replace({'No Data':'98 - OTHER'})
    df['age'] = df['age'].replace({'No Data':37})#average age
    df['make'] = df['make'].replace({'No Data':'unknown', 'UNKNOWN':'unknown'})
    df['color'] = df['color'].replace({'No Data':'99 - UNKNOWN'})
    df['dl_state'] = df['dl_state'].replace({'No Data':'UN - UNKNOWN'})
    df['injured'] = df['injured'].replace({2:1, 3:1, 4:1, 5:1, 6:1, 7:1})
    df['deceased'] = df['deceased'].replace({2:0})
    df['latitude'] = df['latitude'].replace({'No Data':0})#assigned 0 to 'unknown' values in latitude/longitude columns
    df['longitude'] = df['longitude'].replace({'No Data':0})
    df = df[df.latitude != 0]#Decided to drop rows based on o values of latitude values.
 #dropped unneeded characters from respective columns   
    df['weather'] = df['weather'].str[4:]
    df['weather'] = df['weather'].str.strip()
    df['dl_state'] = df['dl_state'].str[5:]
    df['color'] = df['color'].str[6:]
    df['ethnicity'] = df['ethnicity'].str[4:]
    df['ethnicity'] = df['ethnicity'].str.strip()
 #time and date were in two seperate columns. Time was in an object dtype then concat with crash date.
    df.time = df.time.astype(str)
    df['time'] = df['time'].apply(lambda x: x.zfill(4))#zfill
    df.time = df.time.str[:2] + ':' + df.time.str[-2:]#time object was in 24 hour format, but not in datetime format. hadd to format and convert.
    df['crash_date'] = df['date'] +' '+ df['time']#concat time and date
 #datetime conversion and set index   
    df['crash_date'] = pd.to_datetime(df['crash_date'])
    df = df.set_index('crash_date').sort_index()
    convert_dict_int = {'age': int, 'deceased':int, 'injured':int, 'latitude':float, 'longitude':float}
    df = df.astype(convert_dict_int)
#get_dummies creates a seperate df of booleans for the identified columns below. Cleaning for the decission tree.
    dummy_df = pd.get_dummies(df[['driver', 'helmet', 'gender']], dummy_na=False, drop_first=[True, True])
    df = df.drop(columns=['driver', 'helmet', 'gender', 'date', 'time', 'make', 'color', 'weather', 'dl_state', 'ethnicity'])
    df = pd.concat([df, dummy_df], axis=1)
    df = df.rename(columns = {'driver_passenger':'driver', 'helmet_worn':'helmet', 'gender_male':'male'})
    convert_dict_int = {'driver': int, 'helmet':int, 'male':int}
    df = df.astype(convert_dict_int)

    return df

#new df summarization
def summarize(df):
    print(df.shape)
    print(f'___________________________')
    print(df.info())
    print(f'___________________________')      
    print(df.isnull().sum())

#train, validate, test 
def target_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    
    train_validate, test = train_test_split(df, test_size=0.2, random_state=seed, stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=seed, stratify=train_validate[target])
    return train, validate, test

def split(df):
#set split variables
    train_size = int(len(df) * .5)
    validate_size = int(len(df) * .3)
    test_size = int(len(df) - train_size - validate_size)
    validate_end_index = train_size + validate_size
# split into train, validation, test
    train = df[: train_size]
    alidate = df[train_size : validate_end_index]
    test = df[validate_end_index : ]
    return trani, validate, test

######MODEL###############
#prediction, accuracy and class report evaluation function used for the above functions
def get_metrics_bin(clf, X, y):
    '''
    get_metrics_bin will take in a sklearn classifier model, an X and a y variable and utilize
    the model to make a prediction and then gather accuracy, class report evaluations
    Credit to @madeleine-capper
    return:  a classification report as a pandas DataFrame
    '''
    y_pred = clf.predict(X)
    accuracy = clf.score(X, y)
    conf = confusion_matrix(y, y_pred)
    class_report = pd.DataFrame(classification_report(y, y_pred, output_dict=True)).T
    tpr = conf[1][1] / conf[1].sum()
    fpr = conf[0][1] / conf[0].sum()
    tnr = conf[0][0] / conf[0].sum()
    fnr = conf[1][0] / conf[1].sum()
    print(f'''
    The accuracy for our model is {accuracy:.4}
    The True Positive Rate is {tpr:.3}, The False Positive Rate is {fpr:.3},
    The True Negative Rate is {tnr:.3}, and the False Negative Rate is {fnr:.3}
    ''')
    return class_report 