import torch
import matplotlib.pyplot as plt
import scipy as sc
from moc_train import ph_tensor1 as pf1
from moc_train import ph_tensor2 as pf2
from network import Network
from openpyxl import Workbook
import numpy as np
from network_LAAF import DNN_LAAF


layers = [2] + [20] * 8 + [2]
model = DNN_LAAF(layers)
model_loaded = torch.load('model.pth')
model.load_state_dict(model_loaded)
model.eval()
testdata = sc.io.loadmat('Case_MOC_SFM/test.mat')

H_test = testdata['H_test']  # N x T
T_test = testdata['t']  # T x 1
X_test = testdata['X_test']  # N x 1
V_test = testdata['V_test']

Xt = pf1(X_test)
tt = pf1(T_test)
Ht = pf2(H_test)
Vt = pf2(V_test)

# XX_t, TT_t = torch.meshgrid(Xt, tt)
x00 = torch.linspace(10, 300, 30)
X = torch.stack(torch.meshgrid(Xt, tt)).reshape(2, -1).T
x_test = X[:, 0]
t_test = X[:, 1]
# x_test = XX_t.flatten()[:, None]  # NT x 1
# t_test = TT_t.flatten()[:, None]  # NT x 1

h_test = Ht.flatten()[:, None]  # NT x 1
v_test = Vt.flatten()[:, None]  # NT x 1
hv_test = torch.cat([h_test, v_test], 1)

with torch.no_grad():
    HV_pred = model(X).cpu()

# Error
error_h = torch.sqrt(torch.sum((HQ_pred[:, 0] - hq_test[:, 0]) ** 2) / torch.sum(hq_test[:, 0] ** 2))
print('Error : %e' % error_h)

V0 = 0.412
v_pred = HV_pred[:, 1]/100

V_pred = v_pred + (V0 - v_pred[0])  # PINN-Ye-2022
error_v = torch.sqrt(torch.sum((hv_test[:, 1] - V_pred) ** 2) / torch.sum(hv_test[:, 1] ** 2))
print('Error : %e' % error_v)

HV_pred1 = torch.stack([HV_pred[:, 0], V_pred], 1)
DATA = torch.cat([t_test.unsqueeze(1), HV_pred1, hv_test], 1)
DATA = DATA.numpy()
HV_pred = HV_pred.numpy()

list_data = DATA.tolist()
workbook = Workbook()
sheet = workbook.active
for row in list_data:
    sheet.append(row)

workbook.save("output_moc.xlsx")

