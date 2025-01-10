import math
import time

import torch
import torch.nn as nn
import numpy as np
from network import Network
import scipy as sc
from torch.utils.data import DataLoader, TensorDataset
from pyDOE import lhs
from torch.optim import lr_scheduler
from network_LAAF import DNN_LAAF

import matplotlib.pyplot as plt

np.random.seed(1234)
torch.cuda.manual_seed(1234)
torch.manual_seed(1234)

if torch.cuda.is_available():
    """ Cuda support """
    print('cuda available')
    device = torch.device('cuda')
else:
    print('cuda not avail')
    device = torch.device('cpu')

total_epoch = 200000
step_size = 10000

pipe_D = 0.05
pipe_A = 0.25 * pipe_D ** 2 * math.pi
pipe_f = 0.015
pipe_a = 1000.
a_g = 9.806


class PINN:
    def __init__(self, x, t, h, v, xt, tt, ht, vt):
        layers = [2] + [20] * 8 + [2]
        # self.model = Network(layers).to(device)
        self.model = Network(layers).to(device)

        XX0, TT0 = torch.meshgrid(x, t)

        self.pipe_L = 300
        self.n_collo = 30
        x_collocation = torch.linspace(10, self.pipe_L, self.n_collo)

        ub = float(t.max())
        lb = float(t.min())
        self.t_collocation = torch.linspace(lb, ub, 401)
        XX1, TT1 = torch.meshgrid(x_collocation, self.t_collocation)

        x0 = XX0.flatten()[:, None]  # NT x 1
        t0 = TT0.flatten()[:, None]  # NT x 1

        x1 = XX1.flatten()[:, None]  # NT x 1
        t1 = TT1.flatten()[:, None]  # NT x 1

        h0 = h.flatten()[:, None]  # NT x 1
        v0 = v.flatten()[:, None]  # NT x 1

        # training points
        x_o = torch.cat([x0, t0], 1)
        indices0 = torch.randperm(len(x0))
        self.X_o = x_o[indices0, :].to(device)

        # residual points
        x_c = torch.cat([x1, t1], 1)
        indices1 = torch.randperm(len(x1))
        self.X_c = x_c[indices1, :].requires_grad_().to(device)

        # exact solution of training points
        hv0 = torch.cat([h0, v0], 1)
        self.HV = hv0[indices0, :].to(device)

        # Testing points
        self.X_t = torch.stack(torch.meshgrid(xt, tt)).reshape(2, -1).T.to(device)
        ht = ht.flatten()[:, None]  # NT x 1
        vt = vt.flatten()[:, None]  # NT x 1
        self.HV_t = torch.cat([ht, vt], 1).to(device)

        # MSE
        self.criterion = nn.MSELoss()

        # no. iteration
        self.iter = 1

        # Initialization
        self.lamda = nn.Parameter(torch.tensor([0.0], requires_grad=True, device=device))
        self.adam_lamda = torch.optim.Adam([self.lamda], lr=0.001)

        # L-BFGS
        self.lbfgs = torch.optim.LBFGS(self.model.parameters(),
                                       lr=1., max_iter=20000, max_eval=20000,
                                       history_size=50,
                                       tolerance_grad=1e-7,
                                       tolerance_change=1.0 * np.finfo(float).eps,
                                       line_search_fn="strong_wolfe")

        # Adam
        self.adam = torch.optim.Adam(self.model.parameters())

        # lr_decay
        self.schedule = lr_scheduler.ExponentialLR(self.adam, gamma=0.9, verbose=False)
        self.step_size = step_size

        self.train_loss = []  # list_loss
        self.test_loss = []
        self.lamda_list = []

    def loss_PDE_func(self, x_co):

        HV1 = self.model(x_co)

        H1 = HV1[:, 0]
        V1 = HV1[:, 1]/100

        dH_dX = torch.autograd.grad(inputs=x_co,
                                    outputs=H1,
                                    grad_outputs=torch.ones_like(H1),
                                    # retain_graph=True,
                                    create_graph=True)[0]
        dh_dx = dH_dX[:, 0]
        dh_dt = dH_dX[:, 1]

        dV_dX = torch.autograd.grad(inputs=x_co,
                                    outputs=V1,
                                    grad_outputs=torch.ones_like(V1),
                                    # retain_graph=True,
                                    create_graph=True)[0]
        dv_dx = dV_dX[:, 0]
        dv_dt = dV_dX[:, 1]

        f_1 = dv_dt + a_g * dh_dx + pipe_f * V1 * abs(V1) / 2 / pipe_D + \
              self.lamda * (dv_dt - pipe_a * dv_dx)
        f_2 = dh_dt + pipe_a ** 2 / a_g * dv_dx

        return f_1, f_2

    def loss_func(self, x_ob, hv_ob):
        HV0 = self.model(x_ob)
        pred_f1, pred_f2 = self.loss_PDE_func(self.X_c)

        loss_data = self.criterion(HV0, hv_ob)
        hv00 = torch.zeros_like(pred_f1)
        loss_PDE = self.criterion(pred_f1, hv00) + self.criterion(pred_f2, hv00)

        loss = loss_data + 0.1 * loss_PDE

        if self.iter % 100 == 0:
            print(self.iter, loss.item(), loss_data.item(), loss_PDE.item())
        self.iter = self.iter + 1

        return loss, loss_data, loss_PDE

    def closure(self):  # closure for L-BFGS
        self.lbfgs.zero_grad()
        Loss, _, _ = self.loss_func(self.X_o, self.HV)
        # self.losses.append(Loss.item())
        Loss.backward()

        return Loss

    def Train(self):

        print("Using Adam:")
        for i in range(total_epoch):
            self.model.train()
            self.adam.zero_grad()
            self.adam_lamda.zero_grad()
            
            LOSS, loss0, loss1 = self.loss_func(self.X_o, self.HV)
            self.train_loss.append(LOSS.item())
            self.lamda_list.append(self.lamda.item())

            LOSS.backward()
            self.adam.step()
            self.adam_lamda.step()

            # if self.iter % step_size == 0:
            #    self.schedule.step()

            if self.iter % 100 == 0:
                self.model.eval()
                with torch.no_grad():
                    pred_HV = self.model(self.X_t)
                    val_loss = torch.sqrt(torch.sum((pred_HV - self.HV_t)**2) / torch.sum(self.HV_t ** 2))
                    self.test_loss.append(val_loss.item())

        print("Using L-BFGS")
        self.lbfgs.step(self.closure)

        print(torch.mean(self.lamda))
        # trainloss_array = np.array(self.train_loss)
        # np.save('loss_values.npy', trainloss_array)
        # testloss_array = np.array(self.test_loss)
        # np.save('L2error_values.npy', testloss_array)
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
    Z = torch.from_numpy(z).T
    return Z


if __name__ == "__main__":
    # load data
    data = sc.io.loadmat('Case_MOC_SFM/train.mat')

    H_star = data['H_star']  # N x T
    t_star = data['t']  # T x 1
    X_star = data['X_star']  # N x 1
    V_star = data['V_star']

    # Rearrange Data
    X = ph_tensor1(X_star)
    T = ph_tensor1(t_star)
    H_o = ph_tensor2(H_star)
    V_o = ph_tensor2(V_star)*100

    testdata = sc.io.loadmat('Case_MOC_SFM/test.mat')
    H_test = testdata['H_test']  # N x T
    t_test = testdata['t']  # T x 1
    X_test = testdata['X_test']  # N x 1
    V_test = testdata['V_test']

    X_t = ph_tensor1(X_test)
    T_t = ph_tensor1(t_test)
    H_t = ph_tensor2(H_test)
    V_t = ph_tensor2(V_test)*100

    pinn = PINN(X, T, H_o, V_o, X_t, T_t, H_t, V_t)
    start_time = time.time()

    pinn.Train()

    elapsed = time.time() - start_time
    print('Training time: %.4f' % elapsed)

    torch.save(pinn.model.state_dict(), 'model.pth')
