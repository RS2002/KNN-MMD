from torch.utils.data import Dataset
import numpy as np

class CSI_dataset(Dataset):
    def __init__(self, magnitudes, label_action=None, label_people=None, task="action"):
        super().__init__()
        self.magnitudes = magnitudes
        self.label_action = label_action
        self.label_people = label_people
        self.num=self.magnitudes.shape[0]
        if self.label_action is None:
            self.label_action = [-1] * self.num
        if self.label_people is None:
            self.label_people = [-1] * self.num
        self.task=task

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        # return self.magnitudes[index], self.label_action[index], self.label_people[index]
        if self.task=="action":
            return self.magnitudes[index], self.label_action[index]
        elif self.task=="people":
            return self.magnitudes[index], self.label_people[index]
        else:
            return None

def load_data(data_path="./data",train_prop=None,valid_prop=None,task="action"):
    magnitude = np.load(data_path+"/magnitude_linear.npy").astype(np.float32)
    phase = np.load(data_path+"/phase_linear.npy").astype(np.float32)
    magnitude=np.concatenate([np.expand_dims(magnitude, axis=1) ,np.expand_dims(phase, axis=1)],axis=1)
    people=np.load(data_path+"/people.npy").astype(np.int64)
    action=np.load(data_path+"/action.npy").astype(np.int64)
    if train_prop is None:
        return CSI_dataset(magnitude, action, people,task)
    else:
        a = np.zeros_like(people)
        num=[]
        current_num=0
        current_action=None
        for i in range(action.shape[0]):
            if action[i]==current_action:
                current_num+=1
            else:
                current_action = action[i]
                if current_action is None:
                    current_num+=1
                else:
                    num.append(current_num)
                    current_num=0
        num.append(current_num)
        if valid_prop is None:
            current_num=0
            for i in range(len(num)):
                a[current_num:current_num+int(num[i]*train_prop)]=1
                current_num+=num[i]
            b=1-a
            a = a.astype(bool)
            b = b.astype(bool)
            return CSI_dataset(magnitude[a], action[a], people[a],task), CSI_dataset(magnitude[b], action[b], people[b],task)
        else:
            current_num=0
            b = np.zeros_like(people)
            for i in range(len(num)):
                a[current_num:current_num+int(num[i]*train_prop)]=1
                b[current_num+int(num[i]*train_prop):current_num+int(num[i]*(train_prop+valid_prop))]=1
                current_num+=num[i]
            c=1-a-b
            a = a.astype(bool)
            b = b.astype(bool)
            c = c.astype(bool)
            return CSI_dataset(magnitude[a], action[a], people[a],task), CSI_dataset(magnitude[b], action[b], people[b],task), CSI_dataset(magnitude[c], action[c], people[c],task)

def load_zero_people(test_people_list, data_path="./data",task="action"):
    magnitude = np.load(data_path+"/magnitude_linear.npy").astype(np.float32)
    phase = np.load(data_path+"/phase_linear.npy").astype(np.float32)
    magnitude=np.concatenate([np.expand_dims(magnitude, axis=1) ,np.expand_dims(phase, axis=1)],axis=1)
    people=np.load(data_path+"/people.npy").astype(np.int64)
    action=np.load(data_path+"/action.npy").astype(np.int64)
    a = np.zeros_like(people)
    b = np.zeros_like(people)
    for i in range(people.shape[0]):
        a[i]=(people[i] not in test_people_list)
        b[i]=not a[i]
    a=a.astype(bool)
    b=b.astype(bool)
    return CSI_dataset(magnitude[a], action[a], people[a],task),CSI_dataset(magnitude[b], action[b], people[b],task)

def load_zero_shot(test_people_list=None, test_action_list=None, data_path="./data", func="and",task="action"):
    # magnitude = np.load(data_path+"/magnitude_linear.npy").astype(np.float32)
    # phase = np.load(data_path+"/phase_linear.npy").astype(np.float32)
    # magnitude=np.concatenate([np.expand_dims(magnitude, axis=1) ,np.expand_dims(phase, axis=1)],axis=1)
    magnitude = np.load(data_path + "/magnitude_linear.npy").astype(np.float32)
    magnitude = np.expand_dims(magnitude, axis=1)
    people=np.load(data_path+"/people.npy").astype(np.int64)
    action=np.load(data_path+"/action.npy").astype(np.int64)
    a = np.zeros_like(people)
    b = np.zeros_like(people)
    if test_action_list is None and test_people_list is None:
        return CSI_dataset(magnitude, action, people,task)
    elif test_action_list is None:
        for i in range(people.shape[0]):
            a[i]=(people[i] not in test_people_list)
            b[i]=not a[i]
    elif test_people_list is None:
        for i in range(people.shape[0]):
            a[i]=(action[i] not in test_action_list)
            b[i]=not a[i]
    else:
        if func=="and":
            for i in range(people.shape[0]):
                a[i]=(action[i] not in test_action_list or people[i] not in test_people_list)
                b[i]=not a[i]
        elif func=="or":
            for i in range(people.shape[0]):
                a[i]=(action[i] not in test_action_list and people[i] not in test_people_list)
                b[i]=not a[i]
        else:
            print("ERROR")
            exit(-1)
    a=a.astype(bool)
    b=b.astype(bool)
    return CSI_dataset(magnitude[a], action[a], people[a],task),CSI_dataset(magnitude[b], action[b], people[b],task)
