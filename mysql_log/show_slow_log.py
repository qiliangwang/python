import os
import time
import pickle


def read_data(file_dir):
    time_patten = '# Time:'
    user_patten = '# User@Host:'
    query_patten = '# Query_time:'
    data = []
    time_info = ''
    user_info = ''
    query_info = ''
    new_data = False
    sql_content = []

    def check_valid():
        return time_info != '' and user_info != '' and query_info != ''

    with open(file_dir, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if line:
                if line.startswith(time_patten):
                    if time_info != '':
                        new_data = True

                    time_info = line

                elif line.startswith(user_patten):
                    if user_info != '':
                        new_data = True

                    user_info = line

                elif line.startswith(query_patten):
                    if query_info != '':
                        new_data = True

                    query_info = line

                elif check_valid():
                    if new_data:
                        small_batch_data = {'time_info': time_info,
                                            'user_info': user_info,
                                            'query_info': query_info,
                                            'sql_content': sql_content
                                            }
                        data.append(small_batch_data)
                        sql_content = []
                        new_data = False
                    else:
                        sql_content.append(line)
                    pass
            if not line:
                break
    return data


def main():
    full_data = []
    slow_log_fir = '/home/vaderwang/Desktop/logs'
    files = os.listdir(slow_log_fir)
    for file in files:
        print(file)
        data = read_data(os.path.join(slow_log_fir, file))
        full_data = full_data + data
        # print(data)
        # print('Len :data', len(data))
        # for line in data:
        #     print(line)
        # break
    pickle.dump(full_data, open('./data/full_data.pkl', 'wb'))
    pass


if __name__ == '__main__':
    main()
