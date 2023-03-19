'''
A simple regression model
Author: Oculins
Date: 2023.02
'''


import argparse
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.discriminator import Regressor
from utils.dataset import DataSet
from tqdm import tqdm



# modify for wider dataset and vit models

def Args():
    parser = argparse.ArgumentParser(description="settings")
    # model
    parser.add_argument("--mode", default="regression")
    parser.add_argument("--model", default="resnet50")
    # dataset
    parser.add_argument("--dataset", default="fundus", type=str)
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    # optimizer, default SGD
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--w_d", default=0.0001, type=float, help="weight_decay")
    parser.add_argument("--total_epoch", default=10, type=int)
    parser.add_argument("--print_freq", default=100, type=int)
    args = parser.parse_args()
    return args
    

def train(i, args, model, train_loader, optimizer):
    print()
    model.train()
    epoch_begin = time.time()
    for index, data in enumerate(train_loader):
        batch_begin = time.time() 
        img = data['img'].cuda()
        target = data['target'].cuda()

        optimizer.zero_grad()
        logit, loss = model(img, target)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        t = time.time() - batch_begin

        if index % args.print_freq == 0:
            print("Epoch {}[{}/{}]: loss:{:.5f}, lr:{:.5f}, time:{:.4f}".format(
                i, 
                args.batch_size * (index + 1),
                len(train_loader.dataset),
                loss,
                optimizer.param_groups[0]["lr"],
                float(t)
            ))

    t = time.time() - epoch_begin
    print("Epoch {} training ends, total {:.2f}s".format(i, t))


def val(i, args, model, test_loader):
    model.eval()
    print("Test on Epoch {}".format(i))
    result_list = []

    # calculate logit
    for index, data in enumerate(test_loader):
        img = data['img'].cuda()
        target = np.array(data['target'])
        img_path = data['img_path']

        with torch.no_grad():
            logit = model(img)

        result = logit.cpu().detach().numpy()

        result_list.append(np.mean(np.abs(target - result)))

    MAE = np.mean(np.array(result_list))

    print(f"Mean_Age_Error: {MAE * 100}")



def main():
    args = Args()

    # model
    if args.model == "resnet50":
        model = Regressor()
    model.cuda()
    if torch.cuda.device_count() > 1:
        print("lets use {} GPUs.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    train_file = os.path.join(args.data_path, "train.json")
    eval_file = os.path.join(args.data_path, "validation.json")

    train_dataset = DataSet(train_file, args.img_size, args.num_cls, args.speedup)
    test_dataset = DataSet(eval_file, args.img_size, args.num_cls, args.speedup)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


    save_path = f"checkpoint/{args.model}/{args.lr}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f"save path: {save_path}")

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=args.schedular_gamma)

    # training and validation
    for i in range(1, args.total_epoch + 1):
        train(i, args, model, train_loader, optimizer)
        torch.save(model.state_dict(), "{}/epoch_{}.pth".format(save_path, i))
        val(i, args, model, test_loader)
        scheduler.step()


if __name__ == "__main__":
    main()
