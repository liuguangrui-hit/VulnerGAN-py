import os
from numpy.core.fromnumeric import transpose
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from data_process.my_dataset import Dataset_adv, Dataset, Dataset_mix, Dataset_adv_1,Dataset_shadow
from nids_models.models import MLP_NIDS, DNN_NIDS


# net = MLP_NIDS()
net = DNN_NIDS()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.5, 0.999))
criterion = nn.BCELoss()

def test_result(model, test_loader,file_name):
    model.eval()
    test_loss = 0
    correct = 0
    cnt = 0
    with torch.no_grad():
        i = 0
        for data in test_loader:
            x, y = data
            target = y.float().unsqueeze(1)
            optimizer.zero_grad()
            y_pred = model(x)
            test_loss += criterion(y_pred, target)
            pred = y_pred > 0.5
            correct += (pred == target).sum().item()
            cnt += pred.shape[0]

            with open(file_name, "a") as f:
                items = pred.numpy()
                np.savetxt(f, items, fmt='%d', delimiter=',')
            i += 1
        test_loss /= (i + 1)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
            test_loss, correct, cnt, 100. * correct / cnt))

if __name__ == '__main__':
    model_name = 'NIDS_GRU_DNN'
    file_saved = "same_record/" + model_name + ".csv"
    if os.path.exists(file_saved):
        os.unlink(file_saved)
    test_dl = DataLoader(Dataset("../data/cic_2017/data_sets/1.0_test.csv"), batch_size=32, shuffle=False)
    model_p = "model_record/" + model_name + ".pt"
    if os.path.exists(model_p):
        checkpoint = torch.load(model_p)
        net.model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['end_epoch']
        epoch = checkpoint['epoch']
        test_result(net, test_dl, file_saved)
    else:
        start_epoch = 0
        print('No saved model, try start NIDS training！')