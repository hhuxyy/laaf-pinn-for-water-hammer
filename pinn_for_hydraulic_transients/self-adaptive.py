import math
import time

import torch
import torch.nn as nn
import numpy as np
from Network import DNN
import scipy as sc
from torch.optim import lr_scheduler

import matplotlib.pyplot as plt

np.random.seed(1234)
torch.cuda.manual_seed(1234)
torch.manual_seed(1234)
torch.cuda.empty_cache()

if torch.cuda.is_available():
    print('cuda available')
    device = torch.device('cuda')
else:
    print('cuda not avail')
    device = torch.device('cpu')

total_epoch = 200000

pipe_D = 0.05
pipe_A = 0.25 * pipe_D ** 2 * math.pi
pipe_f = 0.015
pipe_a = 1000.
a_g = 9.806
pipe_L = 500

criterion = nn.MSELoss()


class PINN:
    def __init__(self):

        self.model = DNN(layers).to(device)

        y_o = Y_o.to(device)
        hv_o = HV_o.to(device)
        y_c = Y_c.to(device)

        # training points
        indices0 = torch.randperm(len(y_o[:, 0]))
        self.X_o = y_o[indices0, :]
        self.HV = hv_o[indices0, :]

        # residual points
        indices1 = torch.randperm(len(y_c[:, 0]))
        self.X_c = y_c[indices1, :].requires_grad_()

        # testing points
        self.X_t = Y_t.to(device)
        self.HV_t = HV_t.to(device)

        # number iteration
        self.iter = 1

        # Initialization
        self.lamda = nn.Parameter(torch.tensor([0.0], requires_grad=True, device=device))

        self.wf = nn.Parameter(0.1 * torch.rand(len(y_c[:, 0]), requires_grad=True, device=device))

        self.lbfgs = torch.optim.LBFGS(self.model.parameters(),
                                       lr=1., max_iter=20000, max_eval=20000,
                                       history_size=50,
                                       tolerance_grad=1e-7,
                                       tolerance_change=1.0 * np.finfo(float).eps,
                                       line_search_fn="strong_wolfe")

        self.adam = torch.optim.Adam(list(self.model.parameters())+self.lamda)
        self.adam_wf = torch.optim.Adam(self.wf, maximize=True)

        self.train_loss = []
        self.test_loss = []
        self.lamda_list = []

    def loss_data(self, x, hv):
        pred_hv = self.model(x)
        loss = criterion(pred_hv, hv)

        return loss

    def loss_res(self, x):

        hv = self.model(x)

        h = hv[:, 0]
        v = hv[:, 1]

        dH_dX = torch.autograd.grad(inputs=x,
                                    outputs=h,
                                    grad_outputs=torch.ones_like(h),
                                    create_graph=True)[0]
        dh_dx = dH_dX[:, 0]
        dh_dt = dH_dX[:, 1]

        dV_dX = torch.autograd.grad(inputs=x,
                                    outputs=v,
                                    grad_outputs=torch.ones_like(v),
                                    create_graph=True)[0]
        dv_dx = dV_dX[:, 0]
        dv_dt = dV_dX[:, 1]

        f_1 = dv_dt + a_g * dh_dx + pipe_f * v * abs(v) / 2 / pipe_D + \
              self.lamda * (dv_dt - pipe_a * dv_dx)
        f_2 = dh_dt + pipe_a ** 2 / a_g * dv_dx

        loss = torch.mean(self.wf * (f_1**2 + f_2**2))

        return loss

    def loss_func(self, x_ob, hv_ob):
        loss_data = self.loss_data(x_ob, hv_ob)
        loss_res = self.loss_res(self.X_c)

        loss = loss_data + loss_res

        if self.iter % 100 == 0:
            print(self.iter, loss.item(), loss_data.item(), loss_res.item())
        self.iter = self.iter + 1

        return loss, loss_data, loss_res

    def closure(self):  # for lbfgs
        self.lbfgs.zero_grad()
        loss, _, _ = self.loss_func(self.X_o, self.HV)
        # self.losses.append(loss.item())
        loss.backward()

        return loss

    def Train(self):
        print("Using Adam")
        for i in range(total_epoch):
            self.model.train()
            self.adam.zero_grad()
            self.adam_wf.zero_grad()
            loss_value, loss_data_value, loss_res_value = self.loss_func(self.X_o, self.HV)

            self.train_loss.append(loss_value.item())
            self.lamda_list.append(self.lamda.item())

            loss_value.backward()
            self.adam.step()
            self.adam_wf.step()

            if self.iter % 100 == 0:
                self.model.eval()
                with torch.no_grad():
                    pred_hv = self.model(self.X_t)
                    # pred = pred_HQ.detach()
                    val_loss = criterion(pred_hv, self.HV_t) / torch.mean(self.HV_t ** 2)
                    self.test_loss.append(val_loss.item())

        print("Using L-BFGS")
        self.lbfgs.step(self.closure)

        print(torch.mean(self.lamda))
        lamda_array = np.array(self.lamda_list)
        np.save('lambda_values.npy', lamda_array)

        plt.figure(1)
        plt.semilogy(self.train_loss, label='Train Loss', color='blue')
        plt.title('Model Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_train.png', format='png')
        plt.close()

        plt.figure(2)
        plt.semilogy(self.test_loss, linestyle='--', label='Test Loss', color='orange')
        plt.title('Evaluate Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_test.png', format='png')
        plt.close()


def ph_tensor1(y):
    y = y.astype(np.float32)
    Y = torch.from_numpy(y)
    YY = Y.squeeze(-1)
    return YY


def ph_tensor2(z):
    z = z.astype(np.float32)
    z = torch.from_numpy(z).T
    z = z.flatten()[:, None]
    return z


if __name__ == "__main__":
    data = sc.io.loadmat('case2/train.mat')

    H_star = data['H_star']  # N x T
    t_star = data['t']  # T x 1
    X_star = data['X_star']  # N x 1
    V_star = data['V_star']

    # Rearrange Data
    X = ph_tensor1(X_star)
    T = ph_tensor1(t_star)
    H_o = ph_tensor2(H_star)
    V_o = ph_tensor2(V_star)

    Y_o = torch.stack(torch.meshgrid(X, T)).reshape(2, -1).T
    HV_o = torch.cat([H_o, V_o], 1)

    n_c = 30
    X_c = torch.linspace(10, pipe_L, n_c)
    T_c = torch.linspace(float(T.min()), float(T.max()), 401)
    Y_c = torch.stack(torch.meshgrid(X_c, T_c)).reshape(2, -1).T

    testdata = sc.io.loadmat('case2/test.mat')
    H_test = testdata['H_test']  # N x T
    X_test = testdata['X_test']  # N x 1
    V_test = testdata['V_test']

    X_t = ph_tensor1(X_test)
    H_t = ph_tensor2(H_test)
    V_t = ph_tensor2(V_test)

    Y_t = torch.stack(torch.meshgrid(X_t, T)).reshape(2, -1).T
    HV_t = torch.cat([H_t, V_t], 1)

    layers = [2] + [20] * 8 + [2]
    pinn = PINN()
    start_time = time.time()

    pinn.Train()

    elapsed = time.time() - start_time
    print('Training time: %.4f' % elapsed)

    # model save
    torch.save(pinn.model.state_dict(), 'sa.pth')
