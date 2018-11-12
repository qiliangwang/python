import os
import pickle


def load_data(file_name='res_data_200.pkl'):
    file_dir = os.path.join('./data', file_name)
    return pickle.load(open(file_dir, 'rb'))


def main():
    data = load_data()
    with open('result_200.txt', 'w') as f:
        for info in data:
            file_data = info['time_info'] + info['user_info'] + info['query_info'] + info['sql_content']
            f.write(file_data)
    pass


if __name__ == '__main__':
    main()
