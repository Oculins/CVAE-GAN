'''
A simple classification model
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
from utils.dataset import DataSet
from utils.util import *
from models.cvae_net import ConditionalVAE
from models.discriminator import Discriminator
import random
from sklearn.metrics import roc_auc_score
from utils.evaluation import evaluation



def Args():
    parser = argparse.ArgumentParser(description="settings")
    # model
    parser.add_argument("--name", default="CVAE-GAN")
    parser.add_argument("--mode", default="train", choices=["train", "test"])
    parser.add_argument("--in_channels", default=3, type=int)
    parser.add_argument("--num_cls", default=7, type=int)
    # parser.add_argument("--latent_dim", default=2048, type=int)

    # data
    parser.add_argument("--data_path", default="./dataset", type=str)
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--speedup", default=False, type=bool)

    # train
    parser.add_argument("--experiment", default="classifier", type=str)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--save_root", default="./checkpoints", type=str)
    parser.add_argument("--save_freq", default=1, type=int)
    parser.add_argument("--print_freq", default=50, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--schedular_gamma", default=0.1, type=float)
    # parser.add_argument("--kld_weight", default=0.00025, type=float)
    # parser.add_argument("--manual_seed", default=1265, type=int)
    # parser.add_argument("--lambda_1", default=1.0, type=float)
    # parser.add_argument("--lambda_2", default=1.0, type=float)

    # test
    parser.add_argument("--load_from", default=" ", type=str)
    parser.add_argument("--test_num", default=10, type=int)

    args = parser.parse_args()
    return args


def train(i, args, model, train_loader, optimizer):
    print()
    model.train()
    epoch_begin = time.time()
    for index, data in enumerate(train_loader):
        batch_begin = time.time() 
        img = data['img'].cuda()
        label = data['label'].cuda()
        
        optimizer.zero_grad()
        result = model(img)
        loss = model.loss_function(result, label)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        t = time.time() - batch_begin


        if index % args.print_freq == 0:
            print("Epoch {}[{}/{}]: loss:{:.5f}, lr:{:.7f}, time:{:.4f}".format(
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
    print("Eval on Epoch {}".format(i))
    result_list = []
    loss_list = []

    # calculate logit
    for index, data in enumerate(test_loader):
        img = data['img'].cuda()
        label = data['label'].cuda()
        img_path = data['img_path']

        with torch.no_grad():
            logit = model(img)
            loss = model.loss_function(logit, label)

        loss_list.append(loss.cpu())
        preds =  torch.softmax(logit, dim=1)

        _, results = torch.max(preds, 1)
        _, labels = torch.max(label, 1)
        results = results.cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()
        for k in range(len(img_path)):
            result_list.append(
                {
                    "file_name": img_path[k].split("/")[-1].split(".")[0],
                    "label": labels[k],
                    "pred": results[k]
                }
            )
    eval_loss = np.mean(np.array(loss_list))
    print(f"eval loss: {eval_loss}")
    acc = evaluation(result_list, args.num_cls)
    print(f"eval acc: {acc}")

    return eval_loss


def test(args, V, test_loader, save_path):
    V.eval()
    print("Generation on test set.")

    for index, data in enumerate(test_loader):
        
        img = data["img"].cuda()
        real_label = data["age"][0].numpy()
        img_name = data["img_path"][0].split('/')[-1].split('.')[0]
        save_tensor_img(img, os.path.join(save_path, f"{img_name}_origin_age_{real_label}.png"))

        labels = np.eye(args.num_cls)
        for cls in range(args.num_cls):
            label = torch.from_numpy(labels[cls]).unsqueeze(0).cuda()
            with torch.no_grad():
                result = V.generate(img, label)

            save_tensor_img(result, os.path.join(save_path, f"{img_name}_trans_to_cls_{cls}.png"))
        
        print(f"Generation Done of {data['img_path'][0]}.")

        if index > args.test_num: break


def main():
    args = Args()
    print(args)

    # model
    if args.name == "CVAE-GAN": 
        classifier = Discriminator(args.num_cls).cuda()

    # random.seed(args.manual_seed)
    # torch.manual_seed(args.manual_seed)

    if args.mode == 'train':
        train_file = os.path.join(args.data_path, "train.json")
        eval_file = os.path.join(args.data_path, "validation.json")

        train_dataset = DataSet(train_file, args.img_size, args.num_cls, args.speedup)
        test_dataset = DataSet(eval_file, args.img_size, args.num_cls, args.speedup)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        save_path = f"{args.save_root}/classifier/{args.experiment}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print(f"save path: {save_path}")
        

        optimizer_C = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)    
        scheduler_C = optim.lr_scheduler.StepLR(optimizer_C, step_size=4, gamma=args.schedular_gamma)

        # training and validation
        min_loss = float('inf')
        for i in range(1, args.epochs + 1):
            train(i, args, classifier, train_loader, optimizer_C)
            if i % args.save_freq == 0:
                torch.save(classifier.state_dict(), f"{save_path}/classifier_{i}.pth")
            val_loss = val(i, args, classifier, test_loader)
            if val_loss < min_loss:
                min_loss = val_loss
                torch.save(classifier.state_dict(), f"{save_path}/classifier_best.pth")
                print("Best Model Saved.")
            scheduler_C.step()
    
    if args.mode == 'test':
        test_file = os.path.join(args.data_path, "test.json")
        test_dataset = DataSet(test_file, args.img_size, args.num_cls, args.speedup)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

        load_root = f"{args.save_root}/classifier/{args.experiment}"
        print(f"Load Model from: {load_root}")
        classifier.load_state_dict(torch.load(os.path.join(load_root, 'classifier_best.pth')))

        test(args, classifier, test_loader, save_path)

if __name__ == "__main__":
    main()
