# -*- coding: utf-8 -*-
import json
import os
# from pyspark import *
# import databricks.koalas as ks
import pandas as pd
import functools
from operator import add
import collections
import datetime
from more_itertools import pairwise
import numpy as np
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist
from matplotlib.image import NonUniformImage
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.cluster import DBSCAN


def json_reader(path):
    """Loads the json content in a Pandas Dataframe

    Args:
        path ('str'): path to directory of json files

    Returns:
        full_data ('DataFrame'): Dataframe with the wanted content of
            all json files
    """
    try:
        # Reads all json files in directory and pass content in a dataframe
        json_files = [pos_json for pos_json in os.listdir(
            path) if pos_json.endswith('.json')]

        columns = ['id', 'date', 'timestamp', 'latitude', 'longtitude']

        full_data = pd.DataFrame(columns=columns)

        for index, js in enumerate(json_files):
            with open(path+'/{}'.format(js)) as json_file:

                data = json.load(json_file)
                main_data = json.loads(data['payload'])

                full_data.loc[index] = [data['serialnumber'], main_data['EndTime'][:10],
                                        main_data['timestamp'], main_data['latitude'], main_data['longtitude']]

    except Exception as e:
        print(index, e)

    full_data = full_data.to_json()
    with open("payload.json", "w", encoding="utf-8") as f:
        json.dump(full_data, f, ensure_ascii=False, indent=4)


def meta_data_reader():
    """Reads the meta data with more information for every user
    and stores them to a new file in json form
    """
    # path to jsons metadata
    path_to_json_metadata = '/home/charis/Documents/Διπλωματική/All_7/'
    json_files = [pos_json for pos_json in os.listdir(
        path_to_json_metadata) if pos_json.endswith('.json')]

    # Initializes dataframe with columns
    meta_df = pd.DataFrame(columns=['id', 'age', 'health_id',
                                    'gender_id', 'education_id',
                                    'usability_id', 'language_code'])

    # Reads all jsons from path
    for index, js in enumerate(json_files):
        with open(os.path.join(path_to_json_metadata, js)) as json_file:
            try:
                # Reads json data and loads the data
                json_data = json_file.read()
                data = json.loads(json_data)

                # Takes user's id
                user_id = data['id']
                user_id = user_id[:-3]

                # Calculates user's age
                age = data['usermetadata_age']
                age = date.today().year - int(age)

                # Takes the rest info I want to keep in dataframe
                health = data['usermetadata_healthstatus_id']
                gender = data['usermetadata_gender_id']
                education = data['usermetadata_education_id']
                usability = data['usermetadata_usability_id']
                country = data['usermetadata_languagecode']

                # Forms the dataframe line by line
                meta_df.loc[index] = [user_id, age, health,
                                      gender, education, usability, country]

            except Exception as e:
                print(e)

    meta_df.dropna(inplace=True)
    # Make dataframe json serializable and save it in a file
    meta_df = meta_df.to_json()
    with open("meta_data2.json", "w") as json_file:
        json.dump(meta_df, json_file, indent=2)


def group_by_id():
    """Groups all data  on their 'id' column and writes the result
    to a new json file where their is all the information->[latitude,longtitude,timestamp]
    in the line with the user's unique id

    Args:
        (DataFrame): data
    """

    # Open the json file with payload to form
    # a table with all info about [lat,long] of a user
    with open('payload.json', 'r') as json_file:
        data = json.load(json_file)

    data = json.loads(data)
    data = pd.DataFrame(data)
    data = data.reset_index()
    data.drop(columns='index', inplace=True)

    if not isinstance(data, pd.DataFrame):
        print("function's Argument should be a pandas Dataframe")
        return -1

    def func(tmp):

        temp_df = functools.reduce(lambda x, y: x+y, tmp)
        return temp_df

    # Pass all the information for every user in a big table
    data = data.groupby('id').aggregate(
        {'latitude': func, 'longtitude': func, 'timestamp': func})
    data = data.reset_index()

    data = data.to_json()

    # home_finder has all payloads for every user

    with open('home_finder.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def cluster_home(x, y):
    """Clusters points on map based on DBSCAN clustering algorithm

    Args:
        x (float): array with values of latitide
        y (array): array with values of longtitude
    Returns:
        num_of_points (int): Number of total places
        on_the_road (float): Percentage of time spent on the road
        be_at_home (float): Percentage of time spent at home
    """
    samples = int(0.2*len(x))
    dist = 5/11100
    new_array = np.stack((x, y), axis=-1)

    clustering = DBSCAN(eps=dist, min_samples=samples).fit(new_array)

    num_of_points = len(np.unique(clustering.labels_))-1

    counter = collections.Counter(clustering)

    # find on the road
    try:
        on_the_road = counter[-1]
        del counter[-1]

        on_the_road = on_the_road/sum(counter.values())

    except Exception as e:
        print(e)
        on_the_road = 0

    # find be_at_home parameter
    try:
        max_key = max(counter, key=counter.get)
        be_at_home = counter[max_key]
        del counter[max_key]

        counter[-1] = on_the_road

        be_at_home = be_at_home/sum(counter.values())

    except Exception as e:
        print(e)
        be_at_home = 0

    return num_of_points, on_the_road, be_at_home


def antena_index(identity, latitude, longtitude):
    """Reads the json file with all [lat] [long] for every
    user, counts the most common pairs of (lat,long) and if
    a pair appears more than 3 times consideres it as an andena
    and removes it from the home finder

    Args:
        panda (Pandas DataFrame): A DataFrame with the info of a user
    """

    home_index = {}

    real_lat = []
    real_long = []

    try:

        tmp_lat = ["{:+.9f}".format(i) for i in latitude]
        tmp_long = ["{:+.9f}".format(i) for i in longtitude]
        antena_finder = list(map(add, tmp_lat, tmp_long))
        common_co = collections.Counter(antena_finder).most_common()

    except Exception as e:
        print(e)
        return "gtp"
    # Find lat,long pairs that exist more than 3 times in payload
    # and delete it from the table
    for table in common_co:

        if (table[1] > 3):
            antena_finder = list(filter((table[0]).__ne__, antena_finder))
        else:
            continue

    for values in antena_finder:
        try:
            try:
                ant_lat = values[0:12]
                ant_long = values[12:-1]
                ant_lat = float(ant_lat)
                ant_long = float(ant_long)
                real_lat.append(ant_lat)
                real_long.append(ant_long)

            except:
                ant_lat = values[0:13]
                ant_long = values[13:-1]
                ant_lat = float(ant_lat)
                ant_long = float(ant_long)
                real_lat.append(ant_lat)
                real_long.append(ant_long)

        except Exception as e:
            print(e)

    # antennas['lat'].clear()
    # antennas['long'].clear()
    if len(real_lat) < 10 or len(real_long) < 10:
        print('idia')
        return "too little info to cluster"

    # x, y = home(id=l, x=real_lat, y=real_long)

    points_of_interest, on_the_road, be_at_home = cluster_home(
        x=real_lat, y=real_long)

    # Make all numbers integers in order to be json serializable
    # x = int(x)
    # y = int(y)

    home_index = {"points of interest": points_of_interest,
                  "on the road": on_the_road, "be at home": be_at_home}

    antena_finder.clear()
    real_lat.clear()
    real_long.clear()
    return home_index


def clusterer():
    """ Reads all data and groups them based on id and date appling a
    function to add all elements from every list in one list with all 
    coordinates info for a user's day
    """
    stats_from_cluster = []
    dict_from_cluster = {}

    with open('corona_data.json', 'r') as json_file:
        total_data = json.load(json_file)

    total_data = json.loads(total_data)
    total_data = pd.DataFrame(total_data)

    def func(tmp):
        temp_df = functools.reduce(lambda x, y: x+y, tmp)
        return temp_df

    total_data = total_data.groupby(['id', 'date']).aggregate(
        {'latitude': func, 'longtitude': func, 'timestamp': func})

    total_data.reset_index(inplace=True)
    total_data.insert(len(total_data.columns), 'userDay', 1)

    total_data['date'] = pd.to_datetime(total_data['date'])

    # total_data['week'] = total_data['date'].dt.week
    # total_data['year'] = total_data['date'].dt.year

    total_data = total_data.groupby('id').aggregate(
        {'latitude': func, 'longtitude': func, 'timestamp': func, 'userDay': 'sum'})

    total_data.reset_index(inplace=True)
    unique_id = np.unique(total_data['id'])

    total_data.set_index('id', inplace=True)

    for i in unique_id:

        tmp = total_data.loc[i]

        if tmp['userDay'] < 5:
            continue
        # else:

        #     # tmp_lat = functools.reduce(lambda x, y: x+y, tmp['latitude'])
        #     # tmp_long = functools.reduce(lambda x, y: x+y, tmp['longtitude'])
        #     x = antena_index(
        #         identity=i, latitude=tmp['latitude'], longtitude=tmp['longtitude'])

        else:
            num_of_calls = len(tmp['date'])//30
            print(num_of_calls)
            for j in range(int(num_of_calls)):

                top = j*30
                buttom = (j+1)*30
                new_tmp = tmp[top:buttom]

                tmp_lat = functools.reduce(
                    lambda x, y: x+y, new_tmp['latitude'])
                tmp_long = functools.reduce(
                    lambda x, y: x+y, new_tmp['longtitude'])
                x = antena_index(identity=i, latitude=tmp_lat,
                                 longtitude=tmp_long)

        stats_from_cluster.append(x)

        dict_from_cluster[i] = x.copy()

        with open('map_info.json', 'w') as f:
            json.dump(dict_from_cluster, f, ensure_ascii=False, indent=4)

        # stats_from_cluster.clear()


def spd_dist_calc(path):
    """Takes the json with all lat,long and timestamps of users and
    and calulates the speed and the covered distance of every user

    Args:
        path (str): path to directory with json files 

    """

    # reads json
    with open(path, 'r') as json_file:
        data = json.load(json_file)
    data = json.loads(data)
    data = pd.DataFrame(data)
    data = data.reset_index()
    data = data.drop(columns='index')

    # initialize dataframe for statistic results
    statistics = pd.DataFrame(columns=['id', 'mean_spd',
                                       'median_spd', 'walking_dist', 'total_dist', 'num_of_dates'])

    # radius of the earth in km
    R = 6371.0
    # initialize tables for later use
    speed = []
    total_dist = []
    walking_dist = []

    # takes every user in dataframe
    for index, i in enumerate(data['id']):

        tmp = data.loc[data['id'] == i]

        timer = [datetime.datetime.utcfromtimestamp(
            x/1000.0) for x in tmp['timestamp'][index]]

        # sort all lists based on datetime
        timer, tmp['timestamp'][index], tmp['latitude'][index], tmp['longtitude'][index] = (list(t) for t in zip(
            *sorted(zip(timer, tmp['timestamp'][index], tmp['latitude'][index], tmp['longtitude'][index]))))

        date_timer = [x.date() for x in timer]
        date_timer = np.unique(date_timer)
        for pos in range(len(data['latitude'][index])-1):
            try:
                lat1 = np.deg2rad(tmp['latitude'][index][pos])
                lat2 = np.deg2rad(tmp['latitude'][index][pos+1])
                dLat = lat2 - lat1

                long1 = tmp['longtitude'][index][pos]
                long2 = tmp['longtitude'][index][pos+1]
                dLong = np.deg2rad(long2 - long1)

                time_dif = timer[pos+1]-timer[pos]
                # time difference in sec
                time_dif = time_dif.total_seconds()

                # --------math type for distance-------
                a = np.sin(dLat/2)**2 + np.cos(lat1) * \
                    np.cos(lat2)*np.sin(dLong/2)**2
                c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
                # distance in km
                tmp_dist = R*c

                # speed in km/h
                tmp_spd = (tmp_dist/time_dif)*3600.0

                # speed check
                if(tmp_spd < 3):
                    total_dist.append(tmp_dist)
                elif(tmp_spd >= 3 and tmp_spd <= 10):
                    speed.append(tmp_spd)
                    walking_dist.append(tmp_dist)
                elif (tmp_spd > 10.0 and tmp_spd <= 100.0):
                    total_dist.append(tmp_dist)
                else:
                    continue
            except Exception as e:
                print(index, i, pos, e)
                continue

        # pass results to DataFrame
        statistics.loc[index] = [i, np.mean(speed), np.median(speed), np.sum(
            walking_dist), np.sum(walking_dist) + np.sum(total_dist), len(date_timer)]

        # clear the lists for the next loop
        speed.clear()
        walking_dist.clear()
        total_dist.clear()

    statistics.to_csv(
        r'/home/charis/Desktop/statistics_data_regular_speed.csv', index=None, header=True)


def json_read_for_stats(path):

    json_files = [pos_json for pos_json in os.listdir(
        path) if pos_json.endswith('.json')]

    columns = ['id', 'date', 'mean_speed',
               'median_speed', 'walking_dist', 'total_dist', 'jsons']

    stat_data = pd.DataFrame(columns=columns)

    prev_data = {}
    counter = 0
    R = 6371.0
    speed = []
    walking_dist = []
    no_walking_dist = []

    for index, js in enumerate(json_files):
        with open(path+'/{}'.format(js)) as json_file:
            try:
                side_data = json.load(json_file)
                data = json.loads(side_data['payload'])

                if (data == prev_data):
                    counter += 1
                    prev_data = data
                    continue
                else:
                    prev_data = data

                timer = [datetime.datetime.fromtimestamp(
                    x/1000.0) for x in data['timestamp']]

                for pos in range(len(data['latitude'])-1):

                    lat1 = np.deg2rad(data['latitude'][pos])
                    lat2 = np.deg2rad(data['latitude'][pos+1])
                    dLat = lat2 - lat1

                    long1 = data['longtitude'][pos]
                    long2 = data['longtitude'][pos+1]
                    dLong = np.deg2rad(long2 - long1)

                    time_dif = timer[pos+1]-timer[pos]
                    # time difference in sec
                    time_dif = time_dif.total_seconds()

                    # --------math type for distance-------
                    a = np.sin(dLat/2)**2 + np.cos(lat1) * \
                        np.cos(lat2)*np.sin(dLong/2)**2
                    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
                    # distance in km
                    tmp_dist = R*c

                    # speed in km/h
                    tmp_spd = (tmp_dist/time_dif)*3600.0

                    # speed check
                    if(tmp_spd < 1.5):
                        no_walking_dist.append(tmp_dist)
                    elif(tmp_spd >= 1.5 and tmp_spd <= 10.0):
                        speed.append(tmp_spd)
                        walking_dist.append(tmp_dist)
                    elif(tmp_spd > 10.0 and tmp_spd <= 100.0):
                        no_walking_dist.append(tmp_dist)
                    else:
                        continue

                stat_data.loc[index] = [side_data['serialnumber'], data['EndTime'][:10], np.mean(
                    speed), np.median(speed), np.sum(walking_dist), np.sum(walking_dist)+np.sum(no_walking_dist), 1]

                speed.clear()
                walking_dist.clear()
                no_walking_dist.clear()
            except Exception as e:
                print(e)

    stat_data = stat_data.to_json()
    with open('stat_per_json.json', 'w', encoding='utf-8') as f:
        json.dump(stat_data, f, ensure_ascii=False, indent=4)


def stats_meta():
    """Combines the statistic results of every user with his meta
    data -> age, health status, gender, education, usability
    """

    meta_data = pd.read_csv("/home/charis/Desktop/meta_data.csv")
    statistics_data = pd.read_csv(
        "/home/charis/Desktop/statistics_data_regular_speed.csv")
    # clinic_id = pd.read_csv("/home/charis/Desktop/SDATA_SEPTEMBER_new.csv")

    # clinic_id['id'] = clinic_id['Secondary ID']

    # print(clinic_id)

    statistics_data.dropna(inplace=True)
    statistics_data.reset_index(inplace=True)
    statistics_data.drop(columns='index', inplace=True)
    statistics_data['mean_walking_dist'] = statistics_data['walking_dist'] / \
        statistics_data['num_of_dates']

    meta_data.drop_duplicates(inplace=True)
    meta_data.reset_index(inplace=True)
    meta_data.drop(columns='index')

    full_data = pd.merge(statistics_data, meta_data, on='id', how='inner')
    # full_data = pd.merge(full_data, clinic_id, on='id', how='inner')
    for numOfDays in [50, 60, 70, 80, 90, 100]:
        full_data = full_data[full_data['num_of_dates'] >= numOfDays]

        full_data = full_data[full_data['age'] >= 45]
        full_data = full_data[full_data['age'] < 89]

        # full_data = full_data[full_data['mean_spd'] >= 3.5]
        # full_data = full_data[full_data['median_spd'] >= 3.5]

        full_data_park = full_data[full_data['health_id'] == 0]
        full_data_health = full_data[full_data['health_id'] != 0]
        full_data_hist = full_data[full_data['health_id'] == 1]

        print(full_data_park.describe())
        print(full_data_health.describe())

        hist(full_data_park['mean_spd'], bins=10,
             color='red', alpha=0.5, normed=True)
        hist(full_data_health['mean_spd'], bins=10,
             color='blue', alpha=0.5, normed=True)
        plt.show()


def haus_manipulate():
    """Calculates statistics of huasdorff distance for every user
    """

    with open("hausdorff_dist.json", "r") as json_file:
        data = json.load(json_file)

    rand_dict = {}
    for key, value in data.items():

        std = float(np.std(value))
        min_val = float(np.min(value))
        max_val = float(np.max(value))
        rand_dict[key] = {"std": std, "min": min_val, "max": max_val}

        with open("haus_metrics.json", "w") as f:
            json.dump(rand_dict, f, ensure_ascii=False, indent=4)


def reader(name):
    with open(name, 'r') as json_file:
        data = json.load(json_file)

    data = json.loads(data)
    data = pd.DataFrame(data)
    return data


def corona_walking():
    """Find the walking distance of people in Greece
    during the first three months of 2020 
    """

    data = reader(name='corona_data.json')
    data_2 = reader(name='corona_data_2.json')
    data_2 = data_2.loc[data_2['date'] < '2020-03-01']

    home_data = reader('corona_data_home_closer.json')
    home_data = home_data.loc[home_data['date'] > '2020-03-23']

    data = pd.concat([data, data_2, home_data], ignore_index=True)

    data = data.groupby(['id', 'date']).aggregate(
        {'walking_dist': np.sum, 'total_dist': np.sum})

    data = data.loc[data['walking_dist'] <= 15]
    data.reset_index(inplace=True)

    meta_data = reader(name='meta_data2.json')
    meta_data.drop_duplicates(subset=['id'], inplace=True)

    data = pd.merge(data, meta_data, on='id')
    data.reset_index(inplace=True)

    data = data.loc[data['language_code'] == 'el']

    print(len(np.unique(data['id'])))

    data = data.groupby('date').aggregate({'walking_dist': np.mean})
    data.reset_index(inplace=True)
    data.date = pd.to_datetime(data.date)
    data = data.loc[data['date'] > '2020-01-01']
    data.set_index('date', inplace=True)
    data = data.sort_index()
    data.reset_index(inplace=True)

    # for identity in np.unique(data['id']):
    #     try:
    #         tmp_data = data.loc[data['id'] == identity]
    decomposition = sm.tsa.seasonal_decompose(
        data['walking_dist'], model="additive", period=7)

    plt.figure(figsize=(8.4, 6.4))
    # plt.title(str(identity))

    # seaborn.regplot(data.index, data['walking_dist'], color='lightgreen')

    plt.scatter(data.index,
                data['walking_dist'], alpha=0.2, c='green')
    plt.plot(decomposition.trend.index,
             decomposition.trend, c='blue', linewidth=2)
    plt.xticks(rotation=30)

    plt.show()
