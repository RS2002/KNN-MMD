import torch
import torch.nn.functional as F
import copy
import math
import random
from scipy.optimize import linear_sum_assignment

def compute_kernel(x, y, kernel_type='gaussian', kernel_param=1.0):
    """
    计算核矩阵
    Args:
        x: 样本集合X，大小为 (batch_size, feature_dim)
        y: 样本集合Y，大小为 (batch_size, feature_dim)
        kernel_type: 核函数类型，可以是 'gaussian' 或 'linear'
        kernel_param: 核函数参数，对于高斯核，表示高斯核的方差

    Returns:
        kernel_matrix: 核矩阵，大小为 (batch_size, batch_size)
    """
    if kernel_type == 'gaussian':
        x_norm = (x**2).sum(dim=-1, keepdim=True)  # 计算 x 的范数的平方
        y_norm = (y**2).sum(dim=-1, keepdim=True)  # 计算 y 的范数的平方
        xy = torch.matmul(x, y.t())  # 计算 x 和 y 的内积
        pairwise_distance = x_norm + y_norm.t() - 2 * xy  # 计算欧氏距离的平方
        kernel_matrix = torch.exp(-pairwise_distance / (2 * kernel_param**2))  # 高斯核函数
    elif kernel_type == 'linear':
        kernel_matrix = torch.matmul(x, y.t())  # 线性核函数
    else:
        raise ValueError("Unsupported kernel type.")

    return kernel_matrix

def mk_mmd_loss(x, y, kernel_types=['gaussian','gaussian','gaussian','gaussian','gaussian'], kernel_params=[0.1, 0.5, 1.0, 2.0, 5.0]):
    """
    计算MK-MMD损失函数
    Args:
        x: 样本集合X，大小为 (batch_size, feature_dim)
        y: 样本集合Y，大小为 (batch_size, feature_dim)
        kernel_types: 核函数类型列表，例如 ['gaussian', 'linear']
        kernel_params: 核函数参数列表，例如 [1.0, 0.5]

    Returns:
        mk_mmd: MK-MMD损失值
    """
    batch_size = x.size(0)
    n_kernels = len(kernel_types)

    # 计算各个核矩阵
    xx_kernels = [compute_kernel(x, x, kernel_type, kernel_param) for kernel_type, kernel_param in zip(kernel_types, kernel_params)]
    yy_kernels = [compute_kernel(y, y, kernel_type, kernel_param) for kernel_type, kernel_param in zip(kernel_types, kernel_params)]
    xy_kernels = [compute_kernel(x, y, kernel_type, kernel_param) for kernel_type, kernel_param in zip(kernel_types, kernel_params)]

    # 计算MK-MMD值
    mk_mmd = 0.0
    for i in range(n_kernels):

        xx = xx_kernels[i]
        yy = yy_kernels[i]
        xy = xy_kernels[i]
        mmd = torch.mean(xx) - 2 * torch.mean(xy) + torch.mean(yy)
        mk_mmd += torch.sqrt(torch.max(torch.tensor(0.0), mmd))


    mk_mmd /= n_kernels

    return mk_mmd

def mk_dis(x,y, kernel_types=['gaussian','gaussian','gaussian','gaussian','gaussian'], kernel_params=[0.1, 0.5, 1.0, 2.0, 5.0]):

    n_kernels = len(kernel_types)

    # 计算各个核矩阵
    xx_kernels = [compute_kernel(x, x, kernel_type, kernel_param) for kernel_type, kernel_param in zip(kernel_types, kernel_params)]
    yy_kernels = [compute_kernel(y, y, kernel_type, kernel_param) for kernel_type, kernel_param in zip(kernel_types, kernel_params)]
    xy_kernels = [compute_kernel(x, y, kernel_type, kernel_param) for kernel_type, kernel_param in zip(kernel_types, kernel_params)]

    total_dis=0

    for i in range(n_kernels):
        # xx = xx_kernels[i]
        # yy = yy_kernels[i]
        xy = xy_kernels[i]

        # xx_dig=torch.diag(xx)
        # yy_dig=torch.diag(yy)
        # x2=xx_dig.unsqueeze(1).repeat(1,xy.shape[1])
        # y2=yy_dig.unsqueeze(0).repeat(xy.shape[0],1)
        # print(x2)
        # print(y2)

        x2,y2=1,1 # 高斯分布对角线为1

        dis = x2 - 2 * xy + y2
        dis = torch.sqrt(torch.max(torch.tensor(0.0), dis))
        total_dis += dis

    avg_dis = total_dis / n_kernels
    return avg_dis

class KernelKMeans:
    def __init__(self, n_clusters=3, gamma=1.0, max_iter=100, device=None, centers=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.max_iter = max_iter
        self.device = device
        if self.device is None:
            torch.device("cpu")
        self.centers_=centers


    def _kernel_function(self, X, Y):
        pairwise_sq_dists = mk_dis(X,Y)
        K = torch.exp(-self.gamma * pairwise_sq_dists)
        return K

    def fit(self, X):
        n_samples = X.shape[0]
        self.labels_ = torch.zeros(n_samples, dtype=torch.int64).to(self.device)
        if self.centers_ is None:
            self.centers_ = X[torch.randperm(n_samples)[:self.n_clusters]].to(self.device)
        self.iterations_ = 0

        for _ in range(self.max_iter):
            self.iterations_ += 1
            distances = mk_dis(X, self.centers_)
            self.labels_ = torch.argmin(distances, dim=1).to(self.device)

            prev_centers = self.centers_.clone().to(self.device)
            for i in range(self.n_clusters):
                cluster_points = X[self.labels_ == i].to(self.device)
                self.centers_[i]= torch.mean(cluster_points,dim=0)
                # weights = self._kernel_function(cluster_points, cluster_points).sum(dim=1)
                # weighted_sum = torch.sum(cluster_points * weights.view(-1, 1), dim=0)
                # total_weights = torch.sum(weights)
                # self.centers_[i] = weighted_sum / total_weights

            if torch.allclose(prev_centers, self.centers_):
                break

        self.inertia_ = torch.sum(torch.min(distances, dim=1).values)

    def predict(self, X):
        distances = mk_dis(X, self.centers_)
        labels = torch.argmin(distances, dim=1).to(self.device)
        return labels,distances






def find_min_distance(A, B):
    # 构建距离矩阵
    distances=mk_dis(A,B)
    # print(distances)

    # 使用线性分配问题求解最小匹配
    row_indices, col_indices = linear_sum_assignment(distances)

    # print(row_indices)
    # print(col_indices)
    #
    # # 计算匹配的距离和
    # min_distance = distances[row_indices, col_indices].sum()

    return col_indices



if __name__ == '__main__':
    # a=torch.rand([2,5])
    # b=torch.rand([3,5])
    # print(mk_mmd_loss(a,b))
    # print(mk_dis(a,b))

    # model = KernelKMeans()
    # a = torch.rand([10, 5])
    # model.fit(a)
    # print(model.predict(a))

    a = torch.rand([6, 64])
    b = torch.rand([6, 64])

    result = find_min_distance(a,b)
    print(result)