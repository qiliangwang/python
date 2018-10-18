from multiprocessing import Pool
import pickle
import os
from jieba import posseg


def read_data():
    file_dir = "./data/corpus.txt"
    data = []
    with open(file_dir, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if line:
                content = line.strip('<content>').strip('</content>').strip()
                if len(content) > 0:
                    sentences = content.split('。')
                    data.extend(sentences)
            if not line:
                break
    return data


def part_of_speech(sentence):
    return [flag for word, flag in posseg.cut(sentence)]


def load_data(file_name='part_of_speech_list.pkl'):
    file_dir = os.path.join('./data', file_name)
    return pickle.load(open(file_dir, 'rb'))


def start_end(part_of_speech_list):
    return ['START'] + part_of_speech_list + ['END']


def unique_node(part_of_speech_result):
    nodes = set()
    for part_of_speech_list in part_of_speech_result:
        for node in part_of_speech_list:
            if node in nodes:
                pass
            else:
                nodes.add(node)
    return nodes


def weight_of_edge(part_of_speech_result):
    weights = {}
    for part_of_speech_list in part_of_speech_result:
        for i in range(len(part_of_speech_list) - 1):
            edge = part_of_speech_list[i] + '->' + part_of_speech_list[i + 1]
            if edge in weights:
                weights[edge] = weights[edge] + 1
            else:
                weights[edge] = 1
    return weights


def main():
    # data = read_data()

    # pool = Pool(32)

    # part_of_speech_result = pool.map(part_of_speech, data)
    # # 对每一句话，加上start和end
    # part_of_speech_result = pool.map(start_end, part_of_speech_result)
    #
    # pickle.dump(part_of_speech_result, open('./data/part_of_speech_list.pkl', 'wb'))

    # part_of_speech_result = load_data('part_of_speech_result.pkl')
    #
    # weights = weight_of_edge(part_of_speech_result)
    # pickle.dump(weights, open('./data/weights.pkl', 'wb'))
    #
    # nodes = unique_node(part_of_speech_result)
    # pickle.dump(nodes, open('./data/nodes.pkl', 'wb'))
    # print(data[:10])
    weights = load_data('weights.pkl')
    print(weights)

    nodes = load_data('nodes.pkl')
    print(nodes)
    pass


if __name__ == '__main__':
    main()
