import pickle

with open('result-1.pkl', 'rb') as f:
    data_list = pickle.load(f)
    data_list.sort(key=lambda x : x[0])
    print(data_list[:10])
    print(data_list[-10:])
