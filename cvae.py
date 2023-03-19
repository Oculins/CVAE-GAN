'''
Conditional VAE model for age synthesis of face or fundus images
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


def Args():
    parser = argparse.ArgumentParser(description="settings")
    # model
    parser.add_argument("--name", default="ConditionalVAE")
    parser.add_argument("--mode", default="train", choices=["train", "test"])
    parser.add_argument("--in_channels", default=3, type=int)
    parser.add_argument("--num_cls", default=7, type=int)
    parser.add_argument("--latent_dim", default=2048, type=int)

    # data
    parser.add_argument("--data_path", default="./dataset", type=str)
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--speedup", default=False, type=bool)

    # train
    parser.add_argument("--experiment", default="try1", type=str)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--save_root", default="./checkpoints", type=str)
    parser.add_argument("--save_freq", default=5, type=int)
    parser.add_argument("--print_freq", default=50, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--schedular_gamma", default=0.1, type=float)
    parser.add_argument("--kld_weight", default=0.00025, type=float)
    parser.add_argument("--manual_seed", default=1265, type=int)

    # test
    parser.add_argument("--load_from", default="./try1/epoch_30.pth", type=str)
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
        results = model(img, label)
        losses = model.loss_function(results, kld=args.kld_weight)
        loss = losses['loss']
        loss.backward()
        optimizer.step()
        t = time.time() - batch_begin

        if index % args.print_freq == 0:
            print("Epoch {}[{}/{}]: loss:{:.5f}, reconstruction loss:{:.5f}, kld loss:{:.5f}, lr:{:.5f}, time:{:.4f}".format(
                i, 
                args.batch_size * (index + 1),
                len(train_loader.dataset),
                losses['loss'],
                losses['Reconstruction_Loss'],
                losses['KLD'],
                optimizer.param_groups[0]["lr"],
                float(t)
            ))
        
    
    t = time.time() - epoch_begin
    print("Epoch {} training ends, total {:.2f}s".format(i, t))


def val(i, args, model, test_loader):
    model.eval()
    print("Test on Epoch {}".format(i))
    loss_list = []
    recons_loss_list = []
    kld_loss_list = []

    # calculate logit
    for index, data in enumerate(test_loader):
        img = data['img'].cuda()
        label = data['label'].cuda()

        with torch.no_grad():
            results = model(img, label)
            loss = model.loss_function(results, kld=args.kld_weight)

        loss_list.append(loss['loss'].cpu())
        recons_loss_list.append(loss['Reconstruction_Loss'].cpu())
        kld_loss_list.append(loss['KLD'].cpu())

    print(f"Epoch {i} eval loss: {np.mean(np.array(loss_list))}")
    print(f"Epoch {i} reconstruction loss: {np.mean(np.array(recons_loss_list))}")
    print(f"Epoch {i} kld loss: {np.mean(np.array(kld_loss_list))}")

def test(args, model, test_loader, save_path):
    model.eval()
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
                result = model.generate(img, label)

            save_tensor_img(result, os.path.join(save_path, f"{img_name}_trans_to_cls_{cls}.png"))
        
        print(f"Generation Done of {data['img_path'][0]}.")

        if index > args.test_num: break


def main():
    args = Args()

    # model
    if args.name == "ConditionalVAE": 
        model = ConditionalVAE(img_size=args.img_size, in_channels=args.in_channels, \
                            num_classes=args.num_cls, latent_dim=args.latent_dim, hidden_dims=None)

    if args.mode == 'test':
        print(f"Load Model from: {args.load_from}")
        model.load_state_dict(torch.load(args.load_from))
        
    model.cuda()
    if torch.cuda.device_count() > 1:
        print("lets use {} GPUs.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    if args.mode == 'train':
        train_file = os.path.join(args.data_path, "train.json")
        eval_file = os.path.join(args.data_path, "validation.json")

        train_dataset = DataSet(train_file, args.img_size, args.num_cls, args.speedup)
        test_dataset = DataSet(eval_file, args.img_size, args.num_cls, args.speedup)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        save_path = f"{args.save_root}/latent_{args.latent_dim}/{args.experiment}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print(f"save path: {save_path}")

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)    
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=args.schedular_gamma)
        

        # training and validation
        for i in range(1, args.epochs + 1):
            train(i, args, model, train_loader, optimizer)
            if i % args.save_freq == 0:
                torch.save(model.state_dict(), "{}/epoch_{}.pth".format(save_path, i))
            val(i, args, model, test_loader)
            scheduler.step()
    
    if args.mode == 'test':
        test_file = os.path.join(args.data_path, "test.json")
        test_dataset = DataSet(test_file, args.img_size, args.num_cls, args.speedup)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

        save_path = f"{args.save_root}/latent_{args.latent_dim}/{args.experiment}/result_images"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print(f"Save images: {save_path}")

        test(args, model, test_loader, save_path)

if __name__ == "__main__":
    main()
