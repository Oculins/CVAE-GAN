'''
CVAE-GAN model for age synthesis of face or fundus images
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
from models.discriminator import *
import random
from sklearn.metrics import roc_auc_score


def Args():
    parser = argparse.ArgumentParser(description="settings")
    # model
    parser.add_argument("--name", default="CVAE-GAN")
    parser.add_argument("--mode", default="train", choices=["train", "test"])
    parser.add_argument("--in_channels", default=3, type=int)
    parser.add_argument("--num_cls", default=7, type=int)
    parser.add_argument("--latent_dim", default=2048, type=int)
    parser.add_argument("--phase", default="regressor", choices=["classifier", "regressor"])

    # data
    parser.add_argument("--data_path", default="./dataset", type=str)
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--speedup", default=False, type=bool)

    # train
    parser.add_argument("--experiment", default="debug", type=str)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--save_root", default="./checkpoints", type=str)
    parser.add_argument("--save_freq", default=1, type=int)
    parser.add_argument("--print_freq", default=50, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--schedular_gamma", default=0.1, type=float)
    parser.add_argument("--kld_weight", default=0.00025, type=float)
    parser.add_argument("--manual_seed", default=1265, type=int)
    parser.add_argument("--lambda_1", default=0.1, type=float, help="weight of classifier loss")
    parser.add_argument("--lambda_2", default=0.1, type=float, help="weight of discriminator loss")

    # test
    parser.add_argument("--regression_load_from", default="", type=str)
    parser.add_argument("--test_num", default=10, type=int)

    args = parser.parse_args()
    return args


def train(i, args, V, C, D, train_loader, optimizer_V, optimizer_D):
    epoch_begin = time.time()
    for index, data in enumerate(train_loader):
        batch_begin = time.time() 
        img = data['img'].cuda()
        real_label = data['label'].cuda()
        batch_size = img.shape[0]
        real_age = data["age"].cuda()

        # train discriminator
        optimizer_D.zero_grad()
        pos_label = torch.ones((batch_size, 1)).cuda()
        neg_label = torch.zeros((batch_size, 1)).cuda()
        pos_predict = D(img)
        D_loss_pos = D.loss_function(pos_predict, pos_label)
        fake_img = V.generate(img, real_label)
        neg_predict = D(fake_img.detach())
        D_loss_neg = D.loss_function(neg_predict, neg_label)
        D_loss = D_loss_pos + D_loss_neg
        D_loss.backward()
        optimizer_D.step()

        # train CVAE
        optimizer_V.zero_grad()
        results = V(img, real_label)
        V_losses = V.loss_function(results, kld=args.kld_weight)
        V_loss_1 = V_losses['loss']

        predict_C = C(results[0])
        V_loss_2 = C.loss_function(predict_C, real_age)

        predict_D = D(results[0])
        V_loss_3 = D.loss_function(predict_D, pos_label)

        V_loss = V_loss_1 + args.lambda_1 * V_loss_2 + args.lambda_2 * V_loss_3

        V_loss.backward()
        optimizer_V.step()
        t = time.time() - batch_begin

        if index % args.print_freq == 0:
            print("Epoch {}[{}/{}]: Discriminator loss: {:.5f}, CVAE loss:{:.5f}, CVAE reconstruction loss:{:.5f}, CVAE kld loss:{:.5f}, CVAE classifier loss:{:.5f}, CVAE discriminator loss:{:.5f}, CVAE lr:{:.5f}, time:{:.4f},".format(
                i, 
                args.batch_size * (index + 1),
                len(train_loader.dataset),
                D_loss,
                V_loss,
                V_losses['Reconstruction_Loss'],
                V_losses['KLD'],
                V_loss_2,
                V_loss_3,
                optimizer_V.param_groups[0]["lr"],
                float(t)
            ))
        
    
    t = time.time() - epoch_begin
    print("Epoch {} training ends, total {:.2f}s".format(i, t))


def val(i, args, V, C, D, test_loader):
    V.eval()
    C.eval()
    D.eval()
    print("Eval on Epoch {}".format(i))
    V_loss_list = []
    recons_loss_list = []
    kld_loss_list = []
    C_loss_list = []
    C_auc_list = []
    D_loss_list = []

    # calculate logit
    for index, data in enumerate(test_loader):
        img = data['img'].cuda()
        label = data['label'].cuda()
        batch_size = img.shape[0]
        real_age = data["age"]

        age_error_list = []

        with torch.no_grad():
            results = V(img, label)
            V_loss = V.loss_function(results, kld=args.kld_weight)

            D_pred = D(results[0])
            pos_label = torch.ones((batch_size, 1)).cuda()
            D_loss = D.loss_function(D_pred, pos_label)

            C_pred = C(results[0])
            C_loss = C.loss_function(C_pred, real_age.cuda())

            result = C_pred.cpu().detach().numpy()
            target = real_age.detach().numpy()
            age_error_list.append(np.mean(np.abs(target - result)))
            # outputs_softmax =  torch.softmax(C_pred, dim=1)
            # _, predictions = torch.max(outputs_softmax, 1)
            # predictions = predictions.cpu().numpy()
            # preds = np.zeros((batch_size, args.num_cls))
            # for i in range(batch_size):
            #     preds[i, predictions[i]] = 1
            # C_auc = roc_auc_score(preds, label.cpu().numpy(), multi_class='ovo')


        V_loss_list.append(V_loss["loss"].cpu())
        recons_loss_list.append(V_loss['Reconstruction_Loss'].cpu())
        kld_loss_list.append(V_loss['KLD'].cpu())
        D_loss_list.append(D_loss.cpu())
        C_loss_list.append(C_loss.cpu())

    print("Epoch {}/{}: Classifier loss: {:.5f}, Discriminator loss: {:.5f},    \n \
    CVAE loss:{:.5f}, CVAE reconstruction loss:{:.5f}, CVAE kld loss:{:.5f}, Eval mean age error:{:.5f}".format(
        i, 
        args.epochs,
        np.mean(np.array(C_loss_list)),
        np.mean(np.array(D_loss_list)),
        np.mean(np.array(V_loss_list)),
        np.mean(np.array(recons_loss_list)),
        np.mean(np.array(kld_loss_list)),
        np.mean(np.array(age_error_list)) * 100
    ))


def test(args, V, test_loader, save_path):
    V.eval()
    print("Generation on test set.")

    for index, data in enumerate(test_loader):
        
        img = data["img"].cuda()
        real_label = data["age"][0].numpy()
        real_label = real_label[0] * 100
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
        cvae = ConditionalVAE(img_size=args.img_size, in_channels=args.in_channels, \
                            num_classes=args.num_cls, latent_dim=args.latent_dim, hidden_dims=None).cuda()
        classifier = Discriminator(args.num_cls).cuda if args.phase == "classifier" else Regressor().cuda()
        discriminator = Discriminator().cuda()

        print(f"Load Regression Model from: {args.regression_load_from}")
        classifier.load_state_dict(torch.load(args.regression_load_from))        

    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

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
        
        optimizer_V = optim.Adam(cvae.parameters(), lr=args.lr, weight_decay=args.weight_decay)    
        scheduler_V = optim.lr_scheduler.StepLR(optimizer_V, step_size=4, gamma=args.schedular_gamma)

        optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay)    
        scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=4, gamma=args.schedular_gamma)

        # training and validation
        for i in range(1, args.epochs + 1):
            train(i, args, cvae, classifier, discriminator, train_loader, optimizer_V, optimizer_D)
            if i % args.save_freq == 0:
                torch.save(cvae.state_dict(), f"{save_path}/cvae.pth")
                torch.save(discriminator.state_dict(), f"{save_path}/discriminator.pth")
            val(i, args, cvae, classifier, discriminator, test_loader)
            scheduler_V.step()
            scheduler_D.step()
    
    if args.mode == 'test':
        test_file = os.path.join(args.data_path, "test.json")
        test_dataset = DataSet(test_file, args.img_size, args.num_cls, args.speedup)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

        load_root = f"{args.save_root}/latent_{args.latent_dim}/{args.experiment}"
        print(f"Load Model from: {load_root}")
        cvae.load_state_dict(torch.load(os.path.join(load_root, 'cvae.pth')))
        discriminator.load_state_dict(torch.load(os.path.join(load_root, 'discriminator.pth')))

        save_path = f"{args.save_root}/latent_{args.latent_dim}/{args.experiment}/result_images"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print(f"Save images: {save_path}")

        test(args, cvae, test_loader, save_path)

if __name__ == "__main__":
    main()
