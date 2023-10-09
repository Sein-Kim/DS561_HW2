import numpy as np
import pandas as pd
import torch

class Data_Loader_Day:
    def __init__(self,args):
        data_file = args.datapath        
        data = self.get_data(data_file)
        self.data = data

    
    def get_data(self, path):
        df = pd.read_csv(path)
        speed_data = df.drop(columns=['Link_ID_1'	,'Link_ID_2',	'Center_Point_1',	'Center_Point_2',	'Limit'	,'Length',	'Direction'])
        a = set([c[:5] for c in list(speed_data.columns)])
        dic_day = {}
        day = []
        for a_ in a:
            day.append(a_[2:])
            dic_day[a_[2:]] = a_[:2]
        day.sort()
        col_s = [dic_day[d] + d for d in day]
        datas = []
        for c in col_s:
            filtered_columns = [col for col in speed_data.columns if c in col]
            filtered_df = speed_data[filtered_columns[:287]].to_numpy()
            datas.append(filtered_df)
        return torch.FloatTensor(datas)
    
class Data_Loader:
    def __init__(self,args):
        super(Data_Loader, self).__init__()
        data_file = args.datapath        
        train_set, labels, valid_set = self.get_data(data_file,args)
        self.train_set = train_set
        self.labels = labels
        self.valid_set = valid_set

    def make_batch(self,data,args):
        batch_data, batch_labels = [], []
        t_data = data[0]
        labels = data[1]
        num_batch = len(t_data)//args.batch_size
        
        for i in range(num_batch):
            batch_data.append(torch.FloatTensor(t_data[i*args.batch_size:(i+1)*args.batch_size]))
            batch_labels.append(torch.FloatTensor(labels[i*args.batch_size:(i+1)*args.batch_size]))
        batch_data.append(torch.FloatTensor(t_data[(i+1)*args.batch_size:]))
        batch_labels.append(torch.FloatTensor(labels[(i+1)*args.batch_size:]))
        
        return batch_data, batch_labels
    
    def get_data(self, path, args):
        df = pd.read_csv(path)
        speed_data = df.drop(columns=['Link_ID_1'	,'Link_ID_2',	'Center_Point_1',	'Center_Point_2',	'Limit'	,'Length',	'Direction']).to_numpy()
        dev_sample_index = -1 * int(args.split_percentage * float(speed_data.shape[1]))
        train_set, valid_set = speed_data[:,:dev_sample_index], speed_data[:,dev_sample_index:]
        data = []
        labels = []
        for i in range(train_set.shape[1]-args.time_window):
            data.append(train_set[:,i:i+args.time_window])
            labels.append(train_set[:,i+args.time_window])
        batch_data, batch_labels = self.make_batch([data,labels],args)
        
        return batch_data, batch_labels, torch.FloatTensor(valid_set)