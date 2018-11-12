import pickle
import os
from functools import reduce
from multiprocessing import Pool
import time
import re
import pandas as pd
import pickle


def load_data(file_name='full_data.pkl'):
    file_dir = os.path.join('./data', file_name)
    return pickle.load(open(file_dir, 'rb'))


def process_data(data):
    info = float(str(data['query_info']).strip().split('Query_time:')[1].split('Lock_time:')[0].strip())
    if 30 <= info < 200 and str(data['user_info']).__contains__('program[program]'):
        sql_content = data['sql_content']
        re_patten = "\\'.*\\'"

        sql_key = [re.sub(re_patten, ' ', sql.strip()) for sql in sql_content]
        sql_key = reduce(lambda x, y: x + ' ' + y + ' ', sql_key)

        sql_content = reduce(lambda x, y: x + y, sql_content)

        data['sql_content'] = sql_content
        data['sql_key'] = sql_key
        return data


def remove_repeat_data(total_data):
    res_data = []
    sql_set = set()
    for data in total_data:
        if not sql_set.__contains__(data['sql_key']):
            sql_set.add(data['sql_key'])
            res_data.append(data)
    return res_data


def remove_repeat_timestamp(total_data):
    res_data = []
    sql_set = set()
    for data in total_data:
        key = data['sql_key']
        if str(key).__contains__('timestamp'):
            key = str(key).split('; ')[1]
            pass
        if str(key).__contains__('order by'):
            key = str(key).split('order by')[0]
            pass
        if str(key).__contains__(';'):
            key = str(key).split(';')[0].strip()
            pass

        if not sql_set.__contains__(key):
            sql_set.add(key)
            data['sql_key'] = key
            res_data.append(data)
    return res_data
    pass


def main():
    data = load_data()
    data = [process_data(batch_data) for batch_data in data]
    data = list(filter(lambda x: x is not None, data))
    print('Total Data', len(data))
    res_data = remove_repeat_data(data)
    res_data = remove_repeat_timestamp(res_data)
    print(len(res_data))
    pickle.dump(res_data, open('./data/res_data_30_200.pkl', 'wb'))


if __name__ == '__main__':
    main()
