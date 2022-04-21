from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn import metrics, model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from tqdm import tqdm

feat_to_drop = ['PuaMode', 
                'Census_ProcessorClass', 
                'EngineVersion', 
                'AppVersion', 
                'AvSigVersion', 
                'Census_OSVersion', 
                'OsVer', 
                'OsBuildLab']

prep_pt1 = False
prep_pt2 = False

if prep_pt1:

    db = pd.read_csv('train.csv', low_memory=False)
    print('wdawd')
    print(db.describe())

    rnd = np.random.randint(len(db))
    rnd2 = np.random.randint(len(db))
    rnd3 = np.random.randint(len(db))

    for i in db.columns:
        print(i, ": ", type(db[i][rnd]), " ------> ", db[i][rnd], ", ", db[i][rnd2], ", ", db[i][rnd3], '\n', db[i].value_counts())

    print(len(db.columns))

    #Verificando valores nulos
    missing_data = (db.isnull().sum()/db.isnull().count()*100).sort_values(ascending = False)

    print(missing_data.head(10))

    #Discrepancias en SmartScreen
    db.replace({'SmartScreen': {'Enabled':'on', 'RequiredAdmin':'requireadmin', 'of':'off', 'Promt':'prompt', 'Promprt':'prompt'}})

    #Minusculas
    db['SmartScreen'] = db['SmartScreen'].str.lower()
    db['Census_ChassisTypeName'] = db['Census_ChassisTypeName'].str.lower()
    db['Census_OSEdition'] = db['Census_OSEdition'].str.lower()
    db['Census_PowerPlatformRoleName'] = db['Census_PowerPlatformRoleName'].str.lower()
    db['Census_GenuineStateName'] = db['Census_GenuineStateName'].str.lower()

    #Caracteres extraños
    db['SmartScreen'] = db['SmartScreen'].str.replace(r'&#x(\d\d);', '\1', regex=True)
    db['SmartScreen'] = db['SmartScreen'].str.replace(r'[\x00-\x1f]', '', regex=True)

    db['Census_InternalBatteryType'] = db['Census_InternalBatteryType'].str.replace(r'&#x(\d\d);', '\1', regex=True)
    db['Census_InternalBatteryType'] = db['Census_InternalBatteryType'].str.replace(r'[\x00-\x1f]', '', regex=True)

    #Variables categoricas
    db['SmartScreen'] = db['SmartScreen'].astype('category')
    db['Census_InternalBatteryType'] = db['Census_InternalBatteryType'].astype('category')
    db['Census_ChassisTypeName'] = db['Census_ChassisTypeName'].astype('category')
    db['Census_OSEdition'] = db['Census_OSEdition'].astype('category')
    db['Census_PowerPlatformRoleName'] = db['Census_PowerPlatformRoleName'].astype('category')

    #Añadiendo categoria unknown a clases categoricas
    try:
        if 'unknown' not in db['SmartScreen'].cat.categories:
            db['SmartScreen'].cat.add_categories(['unknown'], inplace=True)
        db['SmartScreen'].fillna('unknown', inplace=True)
    except:
        pass

    try:
        if 'unknown' not in db['Census_PrimaryDiskTypeName'].cat.categories:
            db['Census_PrimaryDiskTypeName'].cat.add_categories(['unknown'], inplace=True)
        db['Census_PrimaryDiskTypeName'].fillna('unknown', inplace=True)
    except:
        pass

    try:
        if 'unknown' not in db['Census_InternalBatteryType'].cat.categories:
            db['Census_InternalBatteryType'].cat.add_categories(['unknown'], inplace=True)
        db['Census_InternalBatteryType'].fillna('unknown', inplace=True)
    except:
        pass

    try:
        if 'unknown' not in db['Census_OSEdition'].cat.categories:
            db['Census_OSEdition'].cat.add_categories(['unknown'], inplace=True)
        db['Census_OSEdition'].fillna('unknown', inplace=True)
    except:
        pass

    try:
        if 'unknown' not in db['Census_PowerPlatformRoleName'].cat.categories:
            db['Census_PowerPlatformRoleName'].cat.add_categories(['unknown'], inplace=True)
        db['Census_PowerPlatformRoleName'].fillna('unknown', inplace=True)
    except:
        pass

    try:
        if 'unknown' not in db['Census_GenuineStateName'].cat.categories:
            db['Census_GenuineStateName'].cat.add_categories(['unknown'], inplace=True)
        db['Census_GenuineStateName'].fillna('unknown', inplace=True)
    except:
        pass

    try:
        if 'unknown' not in db['Census_ChassisTypeName'].cat.categories:
            db['Census_ChassisTypeName'].cat.add_categories(['unknown'], inplace=True)
        db['Census_ChassisTypeName'].fillna('unknown', inplace=True)
    except:
        pass


    db.to_csv('preprocess_pt1.csv')

if prep_pt2:
    db = pd.read_csv('preprocess_pt1.csv', low_memory=False)
    print('wdawd')

    #Removiendo columnas innecesarias
    db = db.drop(columns=feat_to_drop)


    #convirtiendo nans en datos
    nans = ["RtpStateBitfield",
            "DefaultBrowsersIdentifier",
            "AVProductStatesIdentifier",
            "AVProductsInstalled",
            "AVProductsEnabled",
            "CityIdentifier",
            "OrganizationIdentifier",
            "GeoNameIdentifier",
            "IsProtected",
            "SMode",
            "IeVerIdentifier",
            "Firewall",
            "UacLuaenable",
            "Census_OEMNameIdentifier",
            "Census_OEMModelIdentifier",
            "Census_ProcessorCoreCount",
            "Census_ProcessorManufacturerIdentifier",
            "Census_ProcessorModelIdentifier",
            "Census_PrimaryDiskTotalCapacity",
            "Census_SystemVolumeTotalCapacity",
            "Census_TotalPhysicalRAM",
            "Census_InternalPrimaryDiagonalDisplaySizeInInches",
            "Census_InternalPrimaryDisplayResolutionHorizontal",
            "Census_InternalPrimaryDisplayResolutionVertical",
            "Census_InternalBatteryNumberOfCharges",
            "Census_OSInstallLanguageIdentifier",
            "Census_IsFlightingInternal",
            "Census_IsFlightsDisabled",
            "Census_ThresholdOptIn",
            "Census_FirmwareManufacturerIdentifier",
            "Census_IsWIMBootEnabled",
            "Census_IsVirtualDevice",
            "Census_IsAlwaysOnAlwaysConnectedCapable",
            "Wdft_IsGamer",
            "Wdft_RegionIdentifier",
            "Census_FirmwareVersionIdentifier"]


    '''for col in nans:
        for i in tqdm(range(len(db))):
            if pd.isnull(db[col][i]) or pd.isna(db[col][i]) or db[col][i] == 'nan' or db[col][i] == 'Nan' or db[col][i] == 'NaN' or db[col][i] == 'NAN':
                db[col][i] = 0'''
    
    db.fillna(0, inplace=True)

    db.to_csv('preprocess_pt2.csv')


    cols = db.columns

    '''for i in cols:
        for j in range(len(db)):
            if db[i][j] == 'nan' or isna:
    '''



    print('nans: ')
    print('len: ', len(db))
    print('porcentaje de nans: ', db.isnull().sum().sum()/len(db))



        