import argparse
from collections import OrderedDict

import pandas as pd
import os
import xarray as xr
import numpy as np
from functools import reduce
from sklearn.externals import joblib


def get_time_stamp(input_filename):
    return int(input_filename.split('.')[0])


data_folder = '../2019AIOps_data_(20190131update)/'
pickle_folder = '../pickle_folder/'
data_dict_path = pickle_folder + 'data_dict.pkl'
xr_attr_dict_path = pickle_folder + 'xr_attr_dict.pkl'
df_columns_path = pickle_folder + 'df_columns.pkl'
filter_valid = True
header = None
max_num = None
save_pickle_data = True
load_from_pickle = True

# data that can be stored
data_dict = {}
df_columns = []
xr_attr_dict = OrderedDict()

filename_list = os.listdir(data_folder)
filename_list.sort()
filename_list = filename_list[:max_num]

start_time_stamp = get_time_stamp(filename_list[0])
end_time_stamp = get_time_stamp(filename_list[-1])
time_interval = get_time_stamp(filename_list[1]) - start_time_stamp
total_time = end_time_stamp - start_time_stamp
one_day_in_mills = 24 * 60 * 60 * 1000
one_week_in_mills = 7 * one_day_in_mills

period_time = one_week_in_mills

n_period_before = total_time // period_time
last_period_start = start_time_stamp + n_period_before * time_interval


def pre_print():
    print('时间间隔为' + str(time_interval) + '个单位')
    print('总时长为' + str(total_time) + '个单位')
    print('一天时间为' + str(one_day_in_mills) + '个单位')
    print('一周时间为' + str(one_week_in_mills) + '个单位')
    print('最后一周的开始时间为时间戳' + str(last_period_start))


pre_print()


def read_origin_data():
    global df_columns
    global xr_attr_dict

    print('开始读入文件，以下显示的是读入了第几个文件：')
    counter = 0
    for filename in filename_list:

        df = pd.read_csv(filepath_or_buffer=data_folder + filename, header=header)

        if counter == 0:
            df_columns = df.columns
            for df_column in df_columns:
                if df_column != df_columns[len(df_columns) - 1]:
                    xr_attr_dict[df_column] = set()

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
                xr_attr_dict[df_column].update(df[df_column])
        counter += 1
        if counter % 10 == 0 or counter == max_num or counter == len(filename_list):
            print(counter)
        else:
            print(counter, end=',')

    # {a:1, b:2}   [(a,1),(b,2)]
    xr_attr_dict = {key: list(value) for key, value in xr_attr_dict.items()}
    for _, value in xr_attr_dict.items():
        value.sort()

    if save_pickle_data:
        joblib.dump(data_dict, data_dict_path)
        joblib.dump(xr_attr_dict, xr_attr_dict_path)
        joblib.dump(df_columns, df_columns_path)


def read_pickle_data():
    global data_dict
    global xr_attr_dict
    global df_columns
    data_dict = joblib.load(data_dict_path)
    xr_attr_dict = joblib.load(xr_attr_dict_path)
    df_columns = joblib.load(df_columns_path)


if load_from_pickle:
    read_pickle_data()
else:
    read_origin_data()
print(xr_attr_dict)


# # 这里想把数据变成data_cube，但是太慢了
# dim_len_list = [len(value) for _, value in xr_attr_dict.items()]
# xr_attr_dict = {str(key): value for key, value in xr_attr_dict.items()}
# for key, value in data_dict.items():
#     data_array = xr.DataArray(np.zeros(dim_len_list), coords=xr_attr_dict.values(), dims=xr_attr_dict.keys())
#     print(data_array)
#     for index, row in value.iterrows():
#         locate_dict = {}
#         for df_column in df_columns:
#             if df_column != len(df_columns) - 1:
#                 locate_dict[str(df_column)] = row[df_column]
#         # print(locate_dict)
#         data_array.loc[locate_dict] = row[df_columns[- 1]]
#         # print(int(data_array.loc[locate_dict]))
#     data_dict[key] = data_array

def get_v(timestamp=start_time_stamp, attr_dict=None):
    if attr_dict is None:
        attr_dict = OrderedDict()
    df_select = data_dict[timestamp]
    attr_counter = 0
    bool_select = None
    for key, value in attr_dict.items():
        if key in xr_attr_dict.keys() and value != '*':
            if attr_counter == 0:
                bool_select = (df_select[key] == value)
            else:
                bool_select &= (df_select[key] == value)
            attr_counter += 1
    if attr_counter != 0:
        df_select = df_select[bool_select]
    # print(df_select)
    result = df_select[df_columns[-1]].sum()
    # print("v值为" + str(result))
    return result


def get_f(timestamp=last_period_start, attr_dict=None):
    if attr_dict is None:
        attr_dict = OrderedDict()
    list_val = []
    for seek_timestamp in range(start_time_stamp, timestamp, one_week_in_mills):
        list_val.append(get_v(timestamp=seek_timestamp, attr_dict=attr_dict))
    result = np.mean(list_val)
    # print(result)
    return result


def get_next_key(ordered_dict=None, key=object()):
    if ordered_dict is None:
        ordered_dict = OrderedDict()
    list_key = list(ordered_dict.keys())
    target_index = list_key.index(key) + 1
    if target_index < len(list_key):
        return list_key[list_key.index(key) + 1]
    else:
        return None


# len(list(xr_attr_dict.keys()))太大可能会爆内存
def generate_combination_save(container=None, all_dict=None, attr_dict=None, key=object()):
    if container is None:
        container = []
    if all_dict is None:
        all_dict = OrderedDict()
    if attr_dict is None:
        attr_dict = OrderedDict()
    if key is not None:
        for item in all_dict[key]:
            arr_copy = attr_dict.copy()
            arr_copy[key] = item
            next_key = get_next_key(all_dict, key)
            generate_combination_save(container, all_dict, arr_copy, next_key)
    else:
        container.append(attr_dict)


# # WARNING 这里想错了，按照这样写，vector_v很稀疏要算很久
# leaf_vector_list = []
# generate_combination_save(leaf_vector_list, xr_attr_dict, OrderedDict(), list(xr_attr_dict.keys())[0])
#
#
# def get_all_vector_v(timestamp=start_time_stamp):
#     result_vector = []
#     for leaf_vector in leaf_vector_list:
#         v_num = get_v(timestamp, leaf_vector)
#         result_vector.append(v_num)
#     return result_vector

def get_vector_v(timestamp=start_time_stamp):
    df_select = data_dict[timestamp]
    return list(df_select[df_columns[-1]])


def get_vector_f(timestamp=last_period_start):
    result_vector = []
    df_select = data_dict[timestamp]
    for attr_dict in df_select.iloc[:, :-1].to_dict('records'):
        result_vector.append(get_f(timestamp, attr_dict))
    return result_vector


def a_ripples_b(attr_dict_a=None, attr_dict_b=None):
    if attr_dict_b is None:
        attr_dict_b = {}
    if attr_dict_a is None:
        attr_dict_a = {}
    a_items = attr_dict_a.items()
    b_items = attr_dict_b.items()
    for item in a_items:
        if item[1] != '*' and item not in b_items:
            return False
    return True


def is_leaf_element(attr_dict=None):
    if attr_dict is None:
        attr_dict = {}
    return set(attr_dict.keys()) == set(xr_attr_dict.keys())


def get_vector_a(timestamp=last_period_start, cause_set=None, input_vector_v=None):
    if cause_set is None:
        cause_set = set()
    # TODO 利用index 先拿vector_v
    if input_vector_v is not None:
        temp_vector_v = input_vector_v.copy()
    elif 'vector_v' in globals().keys():
        temp_vector_v = globals()['vector_v'].copy()
    else:
        temp_vector_v = get_vector_v(timestamp).copy()
    df_select = data_dict[timestamp]
    attr_dict_leaf_list = df_select.iloc[:, :-1].to_dict('records')
    for attr_dict_cause in cause_set:
        index_counter = 0
        for attr_dict_leaf in attr_dict_leaf_list:
            if a_ripples_b(attr_dict_cause, attr_dict_leaf):
                # print(str(index_counter)+' do a ripples b')
                if not is_leaf_element(attr_dict_cause):
                    # print(str(index_counter)+' is not leaf element')
                    f_value = get_f(timestamp, attr_dict_cause)
                    v_value = get_v(timestamp, attr_dict_cause)
                    temp_vector_v[index_counter] *= (v_value / f_value)
            else:
                f_value = get_f(timestamp, attr_dict_leaf)
                temp_vector_v[index_counter] = f_value
            index_counter += 1
    return temp_vector_v


def get_ed(np_array_1, np_array_2):
    return np.linalg.norm(np_array_1 - np_array_2)


def get_ps(timestamp=last_period_start, cause_set=None):
    if cause_set is None:
        cause_set = set()
    v = np.array(get_vector_v(timestamp))
    f = np.array(get_vector_f(timestamp))
    a = np.array(get_vector_a(timestamp, cause_set, v))
    ed_va = get_ed(v, a)
    ed_vf = get_ed(v, f)
    return max(1 - ed_va / ed_vf, 0)


def do1():
    print(get_vector_a())


def do2():
    print(list(np.round(get_vector_a(cause_set=[{1: 'e08'}]))))


def do3():
    print(get_ps(cause_set=[{1: 'e08'}]))


def args_parse():
    # construct the argument parse and parse the arguments

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--method_name", required=False,
                    help="the method you want to call")
    return vars(ap.parse_args())


if __name__ == '__main__':
    args = args_parse()
    if 'method_name' in args.keys() and args['method_name'] in globals().keys():
        method_name = args['method_name']
        globals()[method_name]()
    while True:
        method_name = input('请输入方法名')
        try:
            if method_name in globals().keys():
                globals()[method_name]()
        except RuntimeError:
            print('调用错误')
