import pandas as pd
import os
import xarray as xr
import numpy as np
from functools import reduce


def get_time_stamp(input_filename):
    return int(input_filename.split('.')[0])


data_folder = '../2019AIOps_data_(20190131update)/'
filter_valid = True
header = None
max_num = None

counter = 0
data_dict = {}
df_columns = []
xr_attr_dict = {}
filename_list = os.listdir(data_folder)
filename_list.sort()
time_interval = get_time_stamp(filename_list[1]) - get_time_stamp(filename_list[0])

print(time_interval)

for filename in filename_list:
    if max_num is not None and max_num > len(filename_list):
        raise RuntimeError('没有这么多的文件')

    if max_num is not None and counter == max_num:
        break

    df = pd.read_csv(filepath_or_buffer=data_folder + filename, header=header)

    if counter == 0:
        df_columns = df.columns
        for df_column in df_columns:
            if df_column != df_columns[len(df_columns) - 1]:
                xr_attr_dict[str(df_column)] = set()

    if filter_valid:
        # delete unknown line
        bool_list = [df[attr].isin(['unknown']) for attr in df_columns]
        df = df[~reduce(lambda x, y: x | y, bool_list)]

    time_stamp_val = get_time_stamp(filename)
    data_dict[time_stamp_val] = df
    # xr_attr_dict['timestamp'].add(time_stamp_val)

    for df_column in df_columns:
        if df_column != df_columns[len(df_columns) - 1]:
            # 批量刷入
            xr_attr_dict[str(df_column)].update(df[df_column])
    counter += 1
    if counter % 10 == 0 or counter == max_num or counter == len(filename_list):
        print(counter)
    else:
        print(counter, end=',')

# {a:1, b:2}   [(a,1),(b,2)]
xr_attr_dict = {key: list(value) for key, value in xr_attr_dict.items()}
for _, value in xr_attr_dict.items():
    value.sort()
dim_len_list = [len(value) for _, value in xr_attr_dict.items()]

print(xr_attr_dict)
# TODO To slow
for key, value in data_dict.items():
    data_array = xr.DataArray(np.zeros(dim_len_list), coords=xr_attr_dict.values(), dims=xr_attr_dict.keys())
    print(data_array)
    for index, row in value.iterrows():
        locate_dict = {}
        for df_column in df_columns:
            if df_column != len(df_columns) - 1:
                locate_dict[str(df_column)] = row[df_column]
        # print(locate_dict)
        data_array.loc[locate_dict] = row[df_columns[len(df_columns) - 1]]
        # print(int(data_array.loc[locate_dict]))
    data_dict[key] = data_array
