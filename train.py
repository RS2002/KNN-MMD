import torch
from dataset import load_zero_shot,CSI_dataset
from torch.utils.data import DataLoader
import tqdm
import umap
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
import torch.nn as nn
import numpy as np
import argparse
from model import Resnet,Linear
from sklearn.model_selection import train_test_split
from func import mk_mmd_loss

support_mmd=False
global_mmd=True

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument("--data_path",type=str,default="./data")
    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda", type=str, default='0')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument("--test_list", type=int, nargs='+', default=[0])
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--task', type=str, default="action")

    parser.add_argument('--k', type=int, default=5) # shot num
    parser.add_argument('--n', type=int, default=0) # neighbor num
    parser.add_argument('--p', type=float, default=0.5) # select top p data for MK-MMD
    parser.add_argument('--d', type=int, default=128) # data reducation dim
    parser.add_argument('--mode', type=int, default=0)

    parser.add_argument("--norm", action="store_false",default=True)
    args = parser.parse_args()
    return args

def k_shot(origin_data,embedding,label,k=5,task="action"):
    if task=="action":
        class_num=6
    elif task=="people":
        class_num=8
    else:
        print("ERROR")
        exit(-1)
    template=torch.zeros_like(origin_data,dtype=torch.float32)
    template=template[:k*class_num]
    template_emb=torch.zeros_like(embedding,dtype=torch.float32)
    template_emb=template_emb[:k*class_num]
    template_label=torch.zeros([k*class_num],dtype=torch.int64)
    num=torch.zeros([class_num])
    index=0

    remove_index=[]
    for i in range(embedding.shape[0]):
        if num[label[i]] < k:
            template[index] = origin_data[i]
            template_emb[index] = embedding[i]
            template_label[index] = label[i]
            index += 1
            remove_index.append(i)
            if index >= k * 6:
                break

    mask = torch.ones(embedding.shape[0], dtype=torch.bool)
    mask[remove_index] = False
    origin_data = origin_data[mask]
    embedding = embedding[mask]
    label = label[mask]

    return template, template_emb, template_label, origin_data, embedding, label

def dimension_reducation(data_loader,input_dim=(1,100,52),output_dim=128):
    origin_data = torch.zeros([len(data_loader.dataset), input_dim[0], input_dim[1], input_dim[2]])
    data_feature = torch.zeros([len(data_loader.dataset), input_dim[0]*input_dim[1]*input_dim[2]])
    data_label = torch.zeros([len(data_loader.dataset)],dtype=torch.int64)
    index=0
    data_iter=iter(data_loader)
    for x, label in data_iter:
        y = x.reshape(x.shape[0], -1)
        num = x.shape[0]
        origin_data[index:index + num]=x
        data_feature[index:index + num] = y
        data_label[index:index + num] = label
        index += num
    reducer = umap.UMAP(n_components=output_dim)
    embedding = reducer.fit_transform(data_feature)
    embedding = torch.from_numpy(embedding)
    return origin_data,embedding,data_label

def top_knn(origin_data,embedding,data_label,k=5,n=None,top_prop=0.5,task="action",mode=0):
    if task=="action":
        class_num=6
    elif task=="people":
        class_num=8
    else:
        print("ERROR")
        exit(-1)

    template, template_emb, template_label, origin_data, embedding, label = k_shot(origin_data,embedding,data_label,k,task)

    if n is None or n < 1:
        n = k // 2 + 1
    elif n > k:
        n = k
    else:
        n = int(n)

    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(template_emb, template_label)
    neigh_dist, neigh_ind = knn.kneighbors(embedding, return_distance=True)
    neighbor_id = template_label[neigh_ind]
    y_pred,_ = stats.mode(neighbor_id, axis=-1)
    y_pred = y_pred.ravel()
    y_pred = torch.from_numpy(y_pred)
    y_pred = y_pred.int()
    y_dis = torch.zeros_like(y_pred, dtype=torch.float32)
    for i in range(y_dis.shape[0]):
        for j in range(n):
            if neighbor_id[i, j] == y_pred[i]:
                y_dis[i] = neigh_dist[i, j]

    x_support = torch.zeros_like(origin_data,dtype=torch.float32)
    y_support = torch.zeros([origin_data.shape[0]],dtype=torch.int64)
    index = 0

    if mode==0: # 在每类数据中分别选择置信度最高的m个数据（m=M*p%,M为目标域总的样本数）
        select_num = int(y_dis.shape[0] * top_prop / class_num)
        for i in range(class_num):
            y_dis_i = y_dis[y_pred == i]
            if len(y_dis_i)==0:
                continue
            origin_data_i = origin_data[y_pred == i]
            # print(y_dis_i)
            # label_i = label[y_pred == i]
            # print(label_i)
            num = y_dis_i.shape[0]
            if num > select_num:
                _, indices = torch.topk(y_dis_i, select_num, largest=False)
                # acc = torch.mean((i == label_i[indices]).float())
                # print(acc)
                origin_data_i = origin_data_i[indices]
            # else:
                # acc = torch.mean((i == label_i).float())
                # print(acc)
            x_support[index:index+origin_data_i.shape[0]]=origin_data_i
            y_support[index:index+origin_data_i.shape[0]]=i
            index+=origin_data_i.shape[0]
        x_support=x_support[:index]
        y_support=y_support[:index]
    elif mode==1: # 直接在全部数据中选择置信度最高的p%数据
        _, indices = torch.topk(y_dis, int(y_dis.shape[0]*top_prop), largest=False)
        # acc = torch.mean((y_pred[indices]==label[indices]).float())
        # print(acc)
        # print(y_pred[indices])
        # print(label[indices])
        # result=y_pred[indices]
        # for i in range(class_num):
        #     print(torch.sum((result==i).sum()))
        x_support=origin_data[indices]
        y_support=label[indices]
    elif mode==2: # 在每类数据中分别选择置信度最高的p%数据
        for i in range(class_num):
            y_dis_i=y_dis[y_pred==i]
            if len(y_dis_i)==0:
                continue
            origin_data_i = origin_data[y_pred == i]
            # data_label_i=label[y_pred==i]
            # print(y_dis_i)
            # print(test_label_i)
            num=y_dis_i.shape[0]
            if int(num*top_prop)==0:
                continue
            _, indices = torch.topk(y_dis_i, int(num*top_prop), largest=False)
            # acc = torch.mean((i==data_label_i[indices]).float())
            # print(acc)
            origin_data_i = origin_data_i[indices]
            x_support[index:index+origin_data_i.shape[0]]=origin_data_i
            y_support[index:index+origin_data_i.shape[0]]=i
            index+=origin_data_i.shape[0]
            x_support = x_support[:index]
            y_support = y_support[:index]
    # print(x_support)
    # print(y_support)

    return template,template_label,x_support,y_support,origin_data,label

# test
# _,test_data=load_zero_shot(test_people_list=[0], data_path="./data",task="action")
# test_loader = DataLoader(test_data, batch_size=256, shuffle=True)
# origin_data,embedding,data_label=dimension_reducation(test_loader,input_dim=(1,100,52),output_dim=128)
# top_knn(origin_data,embedding,data_label,k=5,n=1,top_prop=0.5,task="action",mode=1)


def iteration(model,classifier,optim,train_loader,test_loader,support_loader,global_loader,device,task="action",train=True):
    loss_func=nn.CrossEntropyLoss()

    if train:
        model.train()
        classifier.train()
        torch.set_grad_enabled(True)
        data_loader=train_loader
    else:
        model.eval()
        classifier.eval()
        torch.set_grad_enabled(False)
        data_loader=test_loader

    loss_list = []
    acc_list = []

    pbar = tqdm.tqdm(data_loader, disable=False)

    for x, label in pbar:
        x=x.to(device)
        label=label.to(device)
        if task == "action":
            class_num=6
        elif task == "people":
            class_num=8
        else:
            print("ERROR")
            exit(-1)

        x_emb=model(x)
        y=classifier(x_emb)
        output=torch.argmax(y,dim=-1)
        acc=torch.mean((output==label).float())
        acc_list.append(acc.item())
        loss=loss_func(y,label.to(torch.int64))
        loss_list.append(loss.item())

        if train:
            # if x.shape[0]!=256:
            #     continue
            test_iter=iter(test_loader)
            x_t, label_t = next(test_iter)
            x_t=x_t.to(device)
            x_emb_t = model(x_t)
            label_t = label_t.to(device)

            loss_mmd = 0
            for i in range(class_num):
                x_emb_i = x_emb[label == i]
                xt_emb_i = x_emb_t[label_t == i]
                if x_emb_i.shape[0] == 0 or xt_emb_i.shape[0] == 0:
                    continue
                # if x_emb_i.shape[0]>xt_emb_i.shape[0]:
                #     x_emb_i=x_emb_i[:xt_emb_i.shape[0]]
                # elif x_emb_i.shape[0]<xt_emb_i.shape[0]:
                #     xt_emb_i=xt_emb_i[:x_emb_i.shape[0]]

                loss_mmd += mk_mmd_loss(x_emb_i, xt_emb_i, kernel_types=['gaussian','gaussian'], kernel_params=[0.5, 1.0]) / class_num
            loss+=loss_mmd


            if support_mmd:
                ##########
                # loss_mmd for real support set
                test_iter=iter(support_loader)
                x_t, label_t = next(test_iter)
                x_t=x_t.to(device)
                x_emb_t = model(x_t)
                label_t = label_t.to(device)
                loss_mmd = 0
                for i in range(class_num):
                    x_emb_i = x_emb[label == i]
                    xt_emb_i = x_emb_t[label_t == i]
                    if x_emb_i.shape[0] == 0 or xt_emb_i.shape[0] == 0:
                        continue
                    # if x_emb_i.shape[0]>xt_emb_i.shape[0]:
                    #     x_emb_i=x_emb_i[:xt_emb_i.shape[0]]
                    # elif x_emb_i.shape[0]<xt_emb_i.shape[0]:
                    #     xt_emb_i=xt_emb_i[:x_emb_i.shape[0]]
                    loss_mmd += mk_mmd_loss(x_emb_i, xt_emb_i, kernel_types=['gaussian','gaussian'], kernel_params=[0.5, 1.0])
                loss+=loss_mmd


            if global_mmd:
                ##########
                # loss_mmd between the whole training set and testing set
                test_iter=iter(global_loader)
                x_t, _ = next(test_iter)
                x_t=x_t.to(device)
                x_emb_t = model(x_t)
                # if x_emb.shape[0]>x_emb_t.shape[0]:
                #     x_emb=x_emb[:x_emb_t.shape[0]]
                # elif x_emb.shape[0]<x_emb_t.shape[0]:
                #     x_emb_t=x_emb_t[:x_emb.shape[0]]
                total_mmd = mk_mmd_loss(x_emb, x_emb_t, kernel_types=['gaussian','gaussian'], kernel_params=[0.5, 1.0])
                loss += total_mmd


                # y_t = classifier(x_emb_t)
                # y_t = torch.softmax(y_t, dim=-1)
                #
                # loss_inner = torch.mean(torch.std(y_t,dim=-1))
                # loss-=loss_inner
                #
                # # y_max, _ = torch.max(y_t, dim=-1, keepdim=True)
                # # y_max = y_max.repeat(1, 6)
                # # y_t[y_t == y_max] = 0
                # # y_t = torch.sum(y_t, dim=0)
                #
                # y_t=torch.mean(y_t,dim=0)
                #
                # loss_label_unbalance=torch.std(y_t)
                # loss+=loss_label_unbalance



            model.zero_grad()
            classifier.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)  # 用于裁剪梯度，防止梯度爆炸
            optim.step()

    return np.mean(loss_list), np.mean(acc_list)

def main():
    args=get_args()
    device_name = "cuda:"+args.cuda
    device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')
    task=args.task
    if task == "action":
        class_num = 6
        train_data,_=load_zero_shot(test_people_list=args.test_list+['2'], data_path=args.data_path, task=task)
        _,test_data = load_zero_shot(test_people_list=args.test_list, data_path=args.data_path, task=task)
    elif task == "people":
        class_num = 8
        train_data, _ = load_zero_shot(test_action_list=args.test_list+['1'], data_path=args.data_path, task=task)
        _, test_data = load_zero_shot(test_action_list=args.test_list, data_path=args.data_path, task=task)
    else:
        print("ERROR")
        exit(-1)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    origin_data,embedding,data_label=dimension_reducation(test_loader,input_dim=(1,100,52),output_dim=args.d)
    template,template_label,x_support,y_support,origin_data,label = top_knn(origin_data,embedding,data_label,k=args.k,n=args.n,top_prop=args.p,task=task,mode=args.mode)
    template, template_label, x_support, y_support, origin_data, label = template.detach(),template_label.detach(),x_support.detach(),y_support.detach(),origin_data.detach(),label.detach()

    x_support = torch.concat([x_support,template],dim=0)
    y_support = torch.concat([y_support,template_label],dim=0)

    if task=="action":
        support_data = CSI_dataset (template, label_action=template_label.int(), label_people=None, task="action")
        my_support_data = CSI_dataset (x_support, label_action=y_support.int(), label_people=None, task="action")
        test_data = CSI_dataset (origin_data, label_action=label.int(), label_people=None, task="action")
    elif task=="people":
        support_data = CSI_dataset (template, label_action=None, label_people=template_label.int(), task="people")
        my_support_data = CSI_dataset (x_support, label_action=None, label_people=y_support.int(), task="people")
        test_data = CSI_dataset (origin_data, label_action=None, label_people=label.int(), task="people")
    else:
        print("ERROR")
        exit(-1)

    support_loader = DataLoader(support_data, batch_size=args.batch_size, shuffle=True)
    my_support_loader = DataLoader(my_support_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    model = Resnet(output_dims=args.hidden_dim, channel=1, pretrained=True, norm=args.norm).to(device)
    classifier = Linear(input_dims=args.hidden_dim,output_dims=class_num).to(device)

    parameters = set(model.parameters()) | set(classifier.parameters())
    total_params = sum(p.numel() for p in parameters if p.requires_grad)
    print('total parameters:', total_params)
    optim = torch.optim.Adam(parameters, lr=args.lr, weight_decay=0.01)

    best_acc = 0
    best_loss = 1000
    acc_epoch = 0
    loss_epoch = 0
    same_epoch = 0
    j = 0

    while True:
        j += 1
        loss, acc = iteration(model, classifier, optim, train_loader, my_support_loader, support_loader, test_loader, device, task=task, train=True)
        log = "Epoch {} | Train Loss {:06f},  Train Acc {:06f} | ".format(j, loss, acc)
        print(log)
        with open(args.task + ".txt", 'a') as file:
            file.write(log)

        loss, acc = iteration(model, classifier, optim, train_loader, support_loader, None, None, device, task=task, train=False)
        # loss, acc = iteration(model, classifier, optim, train_loader, my_support_loader, None, None, device, task=task, train=False)
        log = "Valid Loss {:06f}, Valid Acc {:06f} | ".format(loss, acc)
        print(log)
        with open(args.task + ".txt", 'a') as file:
            file.write(log)

        test_loss, test_acc = iteration(model, classifier, optim, train_loader, test_loader, None, None, device, task=task, train=False)
        log = "Test Loss {:06f}, Test Acc {:06f} ".format(test_loss, test_acc)
        print(log)
        with open(args.task + ".txt", 'a') as file:
            file.write(log + "\n")

        if acc >= best_acc or loss <= best_loss:
            torch.save(model.state_dict(), args.task + ".pth")
            torch.save(classifier.state_dict(), args.task + "_cls.pth")
        if acc >= best_acc:
            if acc == best_acc:
                same_epoch+=1
            else:
                same_epoch=0
            best_acc = acc
            acc_epoch = 0
        else:
            acc_epoch += 1
            same_epoch = 0
        if loss < best_loss:
            best_loss = loss
            loss_epoch = 0
        else:
            loss_epoch += 1
        print("Acc Epoch {:}, Loss Epcoh {:}, Same Epoch {:}".format(acc_epoch, loss_epoch, same_epoch))
        if (((acc_epoch >= args.epoch and loss_epoch >= args.epoch) or same_epoch >= args.epoch) and j>200) or j>300:
            break
        if j==200:
            acc_epoch = 0
            loss_epoch = 0
            same_epoch = 0


if __name__ == '__main__':
    main()