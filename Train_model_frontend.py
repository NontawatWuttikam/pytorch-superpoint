"""This is the frontend interface for training
base class: inherited by other Train_model_*.py

Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""

import numpy as np
import torch
import random
import os
import pickle
import traceback
# from torch.autograd import Variable
# import torch.backends.cudnn as cudnn
import torch.optim
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm
from utils.loader import dataLoader, modelLoader, pretrainedLoader
import logging

from utils.tools import dict_update

from utils.utils import labels2Dto3D, flattenDetection, labels2Dto3D_flattened

from utils.utils import pltImshow, saveImg
from utils.utils import precisionRecall_torch
from utils.utils import save_checkpoint

from pathlib import Path
import sys

def thd_img(img, thd=0.015):
    """
    thresholding the image.
    :param img:
    :param thd:
    :return:
    """
    img[img < thd] = 0
    img[img >= thd] = 1
    return img


def toNumpy(tensor):
    return tensor.detach().cpu().numpy()


def img_overlap(img_r, img_g, img_gray):  # img_b repeat
    img = np.concatenate((img_gray, img_gray, img_gray), axis=0)
    img[0, :, :] += img_r[0, :, :]
    img[1, :, :] += img_g[0, :, :]
    img[img > 1] = 1
    img[img < 0] = 0
    return img


class Train_model_frontend(object):
    """
    # This is the base class for training classes. Wrap pytorch net to help training process.

    """

    default_config = {
        "train_iter": 170000,
        "save_interval": 2000,
        "tensorboard_interval": 200,
        "model": {"subpixel": {"enable": False}},
    }

    def __init__(self, config, proxyopt_checkpoint_object=None, save_path=Path("."), device="cpu", verbose=False):
        """
        ## default dimension:
            heatmap: torch (batch_size, H, W, 1)
            dense_desc: torch (batch_size, H, W, 256)
            pts: [batch_size, np (N, 3)]
            desc: [batch_size, np(256, N)]

        :param config:
            dense_loss, sparse_loss (default)

        :param save_path:
        :param device:
        :param verbose:
        """
        # config
        print("Load Train_model_frontend!!")
        self.config = self.default_config
        self.config = dict_update(self.config, config)
        self.train_proxy_from_it = self.config["train_proxy_from_it"]
        self.proxyopt_checkpoint_object = proxyopt_checkpoint_object
        print("check config!!", self.config)

        # init parameters
        self.device = device
        self.save_path = save_path
        self._train = True
        self._eval = True
        self.cell_size = 8
        self.subpixel = False
        self.loss = 0

        self.max_iter = config["train_iter"]

        if self.config["model"]["dense_loss"]["enable"]:
            ## original superpoint paper uses dense loss
            print("use dense_loss!")
            from utils.utils import descriptor_loss

            self.desc_params = self.config["model"]["dense_loss"]["params"]
            self.descriptor_loss = descriptor_loss
            self.desc_loss_type = "dense"
        elif self.config["model"]["sparse_loss"]["enable"]:
            ## our sparse loss has similar performace, more efficient
            print("use sparse_loss!")
            self.desc_params = self.config["model"]["sparse_loss"]["params"]
            from utils.loss_functions.sparse_loss import batch_descriptor_loss_sparse

            self.descriptor_loss = batch_descriptor_loss_sparse
            self.desc_loss_type = "sparse"

        if self.config["model"]["subpixel"]["enable"]:
            ## deprecated: only for testing subpixel prediction
            self.subpixel = True

            def get_func(path, name):
                logging.info("=> from %s import %s", path, name)
                mod = __import__("{}".format(path), fromlist=[""])
                return getattr(mod, name)

            self.subpixel_loss_func = get_func(
                "utils.losses", self.config["model"]["subpixel"]["loss_func"]
            )

        # load model
        # self.net = self.loadModel(*config['model'])
        self.printImportantConfig()

        pass

    def printImportantConfig(self):
        """
        # print important configs
        :return:
        """
        print("=" * 10, " check!!! ", "=" * 10)

        print("learning_rate: ", self.config["model"]["learning_rate"])
        print("lambda_loss: ", self.config["model"]["lambda_loss"])
        print("detection_threshold: ", self.config["model"]["detection_threshold"])
        print("batch_size: ", self.config["model"]["batch_size"])

        print("=" * 10, " descriptor: ", self.desc_loss_type, "=" * 10)
        for item in list(self.desc_params):
            print(item, ": ", self.desc_params[item])

        print("=" * 32)
        pass

    def dataParallel(self):
        """
        put network and optimizer to multiple gpus
        :return:
        """
        print("=== Let's use", torch.cuda.device_count(), "GPUs!")
        self.net = nn.DataParallel(self.net)
        self.optimizer = self.adamOptim(
            self.net, lr=self.config["model"]["learning_rate"]
        )
        pass

    def adamOptim(self, net, lr):
        """
        initiate adam optimizer
        :param net: network structure
        :param lr: learning rate
        :return:
        """
        print("adam optimizer")
        import torch.optim as optim

        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
        return optimizer

    def loadModel(self):
        """
        load model from name and params
        init or load optimizer
        :return:
        """
        model = self.config["model"]["name"]
        params = self.config["model"]["params"]
        print("model: ", model)
        net = modelLoader(model=model, **params).to(self.device)
        logging.info("=> setting adam solver")
        # optimize proxy instead of superpoint - BOAT
        # optimizer = self.adamOptim(net, lr=self.config["model"]["learning_rate"])

        # BOAT - warning!! optimizer will not be used since we create new optimizer everytime for each iteration in train_val_sample in train_model_heatmap!!
        optimizer = self.adamOptim(net, self.config["model"]["learning_rate"])
        scheduler_choice = self.config["proxyopt"]["scheduler"]
        if scheduler_choice == "cosine":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 800, eta_min=0, last_epoch=-1)
        elif scheduler_choice == "exponential":
            # Compute decay rate
            initial_lr = self.config["proxyopt"]["learning_rate"]
            final_lr = self.config["proxyopt"]["scheduler_param"]["final_lr"]
            total_steps = self.config["proxyopt"]["scheduler_param"]["total_steps"]
            gamma = (final_lr / initial_lr) ** (1 / total_steps)
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif scheduler_choice is None or scheduler_choice.lower() == "none":
            lr_scheduler = None
        else:
            raise Exception("please choose correct lr scheduler in config")
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95, last_epoch=-1)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9, last_epoch=-1)

        # load from proxyopt checkpoint
        if self.proxyopt_checkpoint_object is not None:
            if self.proxyopt_checkpoint_object["lr_scheduler_state_dict"] is not None:
                lr_scheduler.load_state_dict(self.proxyopt_checkpoint_object["lr_scheduler_state_dict"])

        n_iter = 0
        ## new model or load pretrained
        if self.config["retrain"] == True:
            logging.info("New model")
            logging.info("not allow training because retrain is set to True by BOAT")
            exit(0)
            pass
        else:
            path = self.config["pretrained"]
            mode = "" if path[-4:] == ".pth" else "full" # the suffix is '.pth' or 'tar.gz'
            logging.info("load pretrained model from: %s", path)
            net, optimizer, n_iter = pretrainedLoader(
                net, optimizer, n_iter, path, mode=mode, full_path=True
            )
            logging.info("successfully load pretrained model from: %s", path)

        def setIter(n_iter):
            if self.config["reset_iter"]:
                logging.info("reset iterations to 0")
                n_iter = 0
            return n_iter

        self.net = net
        self.optimizer = optimizer
        # self.n_iter = setIter(n_iter)
        self.n_iter = self.train_proxy_from_it # ignore sp iter and use proxy iter instead
        self.lr_scheduler = lr_scheduler
        pass


    @property
    def writer(self):
        """
        # writer for tensorboard
        :return:
        """
        # print("get writer")
        return self._writer

    @writer.setter
    def writer(self, writer):
        print("set writer")
        self._writer = writer

    @property
    def train_loader(self):
        """
        loader for dataset, set from outside
        :return:
        """
        print("get dataloader")
        return self._train_loader

    @train_loader.setter
    def train_loader(self, loader):
        print("set train loader")
        self._train_loader = loader

    @property
    def val_loader(self):
        print("get dataloader")
        return self._val_loader

    @val_loader.setter
    def val_loader(self, loader):
        print("set train loader")
        self._val_loader = loader

    def train(self, **options):
        """
        # outer loop for training
        # control training and validation pace
        # stop when reaching max iterations
        :param options:
        :return:
        """
        # training info
        logging.info("n_iter: %d", self.n_iter)
        logging.info("max_iter: %d", self.max_iter)
        running_losses = []
        epoch = 0
        # Train one epoch
        while self.n_iter < self.max_iter:
            print("epoch: ", epoch)
            epoch += 1
            # for i, sample_train in tqdm(enumerate(self.train_loader)):
            dataset_indices = list(range(len(self.train_set)))
            random.shuffle(dataset_indices)
            for i in tqdm(dataset_indices):
                solutions = self.es.ask()  # get a new solution from CMA-ES
                losses = []
                loss_dets = []
                loss_descs = []
                sol_count = 0
                while sol_count < len(solutions):
                    sol_id = sol_count
                    solution = solutions[sol_id]
                    print(f"solution {sol_id}/{len(solutions)}: {solution}")
                    solution = self.train_set.proxy_isp_dataset.denormalize_hyp(solution)  # denormalize the solution
                    solution = np.array(solution, dtype=np.float32) # convert to numpy array
                    self.train_set.openisp_current_hype = solution  # set the current hyperparameters for the dataset
                    try:
                        print("\tdenormalized solution: ", self.train_set.openisp_current_hype)
                        sample_train = self.train_set.__getitem__(i)
                    except Exception:
                        print("Exception processing dataset index", i, traceback.format_exc())
                        print("attempting to try solution", sol_id, "again")
                        continue

                    for k,v in sample_train.items():
                        # print(k,v)
                        if isinstance(sample_train[k], torch.Tensor):
                            sample_train[k] = v.unsqueeze(0) # add batch dim to sample
                    is_train = False # BOAT - warning!! set to False for cma-es.
                    print("running train_val_sample for this solution")
                    loss_out, loss_det, loss_desc = self.train_val_sample(sample_train, self.n_iter, is_train)
                    print("\tloss: ", loss_out)
                    losses.append(loss_out)
                    loss_dets.append(loss_det)
                    loss_descs.append(loss_desc)
                    sol_count += 1
                loss_mean = np.mean(losses)
                loss_det_mean = np.mean(loss_dets)
                loss_desc_mean = np.mean(loss_descs)
                print(f"Mean loss for this generation: {loss_mean}")
                print(f"Mean detection loss for this generation: {loss_det_mean}")
                print(f"Mean description loss for this generation: {loss_desc_mean}")

                self.proxy_writer.add_scalar("loss/total_loss", loss_mean, self.n_iter)
                self.proxy_writer.add_scalar("loss/detection_loss", loss_det_mean, self.n_iter)
                self.proxy_writer.add_scalar("loss/description_loss", loss_desc_mean, self.n_iter)

                print("telling cma-es the losses for the solutions")
                self.es.tell(solutions, losses)  # tell CMA-ES the losses for the solutions
                self.n_iter += 1
                running_losses.append(loss_mean)

                min_idx = np.argmin(losses)
                best_solution = solutions[min_idx]

                print("best solution for this generation: ", best_solution)
                # exit(0)

                n_iter = self.n_iter
                sample = sample_train

                if n_iter % self.config["proxyopt"]["log_param_iter"] == 0:
                    idx = 0
                    denormalized_hypes = self.train_set.proxy_isp_dataset.denormalize_hyp(best_solution)
                    print("denormalized hyperparameters to log: ", denormalized_hypes)
                    for param in self.train_set.proxy_isp_dataset.hyp_setting["parameters"]:
                        if param["type"] == "categorical":
                            for bin in range(param["values"].__len__()):
                                bin_name = param["values"][bin]
                                self.proxy_writer.add_scalar("ISP_hyperparameters/" + param["name"]+f"|{bin_name}", denormalized_hypes[idx], n_iter)
                                # self.proxy_writer.add_scalar("ISP_hyperparameters_raw_from_proxy/" + param["name"]+f"|{bin_name}", self.train_set.proxy.return_param_value()[idx], n_iter)
                                # if proxy_gradient_to_log is not None:
                                    # self.proxy_writer.add_scalar("grad/" + param["name"]+f"|{bin_name}", proxy_gradient_to_log[idx], n_iter)
                                idx += 1
                        else:
                            self.proxy_writer.add_scalar("ISP_hyperparameters/" + param["name"], denormalized_hypes[idx], n_iter)
                            # self.proxy_writer.add_scalar("ISP_hyperparameters_raw_from_proxy/" + param["name"], self.train_set.proxy.return_param_value()[idx], n_iter)
                            # if proxy_gradient_to_log is not None:
                                # self.proxy_writer.add_scalar("grad/" + param["name"], proxy_gradient_to_log[idx], n_iter)
                            idx += 1
                    assert idx == len(denormalized_hypes)
                    # self.proxy_writer.add_text("Proxy Hype", str(self.train_set.proxy.return_param_value()), n_iter)

                if n_iter % self.config["proxyopt"]["save_image_iter"] == 0:
                # if True:
                    current_hyp = self.train_set.proxy_isp_dataset.denormalize_hyp(best_solution)
                    current_hyp_image = self.train_set.proxy_isp_dataset.process_raw(sample["bayer"], current_hyp, original_hyp = False)
                    current_hyp_image = torch.tensor(current_hyp_image.astype("float32") / 255.0)
                    current_hyp_image = torch.permute(current_hyp_image, (2, 0, 1))

                    initial_hyp_image = self.train_set.proxy_isp_dataset.process_raw(sample["bayer"], original_hyp=True)
                    initial_hyp_image = torch.tensor(initial_hyp_image.astype("float32") / 255.0)
                    initial_hyp_image = torch.permute(initial_hyp_image, (2, 0, 1))

                    # current_hype_image = sample["proxy_output_image"][0]

                    # Concatenate images horizontally (dim=2 for width)
                    stitched_image = torch.cat((initial_hyp_image, current_hyp_image), dim=2)

                    self.proxy_writer.add_image("image (initial, current)", stitched_image, n_iter)

                if n_iter % self.config["proxyopt"]["save_training_image_iter"] == 0:
                    train_image = sample["image"][0]
                    train_image_warp = sample["warped_img"][0]

                    stitched_train_image = torch.cat((train_image, train_image_warp), dim=2)
                    stitched_train_image = self.resize_image(stitched_train_image)
                    self.proxy_writer.add_image("training image", stitched_train_image, n_iter)

                    # # dump images to folder for debugging
                    # torchvision.utils.save_image(initial_hyp_image, os.path.join("temp_log", f"initial_hyp_image_{n_iter}.png"))
                    # torchvision.utils.save_image(current_hyp_image, os.path.join("temp_log", f"current_hyp_image_{n_iter}.png"))
                    # torchvision.utils.save_image(train_image, os.path.join("temp_log", f"training_image_{n_iter}.png"))
                    # torchvision.utils.save_image(train_image_warp, os.path.join("temp_log", f"training_image_warp_{n_iter}.png"))
                    # exit(0)

                # saving checkpoint - BOAT
                if n_iter % self.config["proxyopt"]["save_hype_iter"] == 0:
                    cma_es_path = Path(self.save_path).parent / "cma_es_checkpoints"
                    os.makedirs(cma_es_path, exist_ok = True)
                    params = np.array(best_solution)
                    lr_scheduler_state_dict = None
                    lr_scheduler_class = None
                    if self.lr_scheduler is not None:
                        lr_scheduler_state_dict = self.lr_scheduler.state_dict()
                        lr_scheduler_class = self.lr_scheduler.__class__.__name__
                    checkpoint_path = str(cma_es_path / f"checkpoint_{n_iter}.pkl")
                    with open(checkpoint_path, "wb") as f:
                        obj = {
                            "proxy_hype":params,
                            "lr_scheduler_state_dict": lr_scheduler_state_dict,
                            # "optimizer_state_dict": optimizer.state_dict(),
                            "lr_scheduler_class": lr_scheduler_class,
                            "optimizer_class": "CMAEvolutionStrategy",
                            "train_proxy_from_it": n_iter
                        }
                        pickle.dump(obj, f)

                # run validation
                if self._eval and self.n_iter % self.config["validation_interval"] == 0:
                    logging.info("====== Validating...")
                    for j, sample_val in enumerate(self.val_loader):
                        self.train_val_sample(sample_val, self.n_iter + j, False)
                        if j > self.config.get("validation_size", 3):
                            break
                # save model
                if self.n_iter % self.config["save_interval"] == 0:
                    logging.info(
                        "save model: every %d interval, current iteration: %d",
                        self.config["save_interval"],
                        self.n_iter,
                    )
                    self.saveModel()
                # ending condition
                if self.n_iter > self.max_iter:
                    # end training
                    logging.info("End training: %d", self.n_iter)
                    break

        pass

    def getLabels(self, labels_2D, cell_size, device="cpu"):
        """
        # transform 2D labels to 3D shape for training
        :param labels_2D:
        :param cell_size:
        :param device:
        :return:
        """
        labels3D_flattened = labels2Dto3D_flattened(
            labels_2D.to(device), cell_size=cell_size
        )
        labels3D_in_loss = labels3D_flattened
        return labels3D_in_loss

    def getMasks(self, mask_2D, cell_size, device="cpu"):
        """
        # 2D mask is constructed into 3D (Hc, Wc) space for training
        :param mask_2D:
            tensor [batch, 1, H, W]
        :param cell_size:
            8 (default)
        :param device:
        :return:
            flattened 3D mask for training
        """
        mask_3D = labels2Dto3D(
            mask_2D.to(device), cell_size=cell_size, add_dustbin=False
        ).float()
        mask_3D_flattened = torch.prod(mask_3D, 1)
        return mask_3D_flattened

    def get_loss(self, semi, labels3D_in_loss, mask_3D_flattened, device="cpu"):
        """
        ## deprecated: loss function
        :param semi:
        :param labels3D_in_loss:
        :param mask_3D_flattened:
        :param device:
        :return:
        """
        loss_func = nn.CrossEntropyLoss(reduce=False).to(device)
        # if self.config['data']['gaussian_label']['enable']:
        #     loss = loss_func_BCE(nn.functional.softmax(semi, dim=1), labels3D_in_loss)
        #     loss = (loss.sum(dim=1) * mask_3D_flattened).sum()
        # else:
        loss = loss_func(semi, labels3D_in_loss)
        loss = (loss * mask_3D_flattened).sum()
        loss = loss / (mask_3D_flattened.sum() + 1e-10)
        return loss

    def train_val_sample(self, sample, n_iter=0, train=False):
        """
        # deprecated: default train_val_sample
        :param sample:
        :param n_iter:
        :param train:
        :return:
        """
        task = "train" if train else "val"
        tb_interval = self.config["tensorboard_interval"]

        losses = {}
        ## get the inputs
        # logging.info('get input img and label')
        img, labels_2D, mask_2D = (
            sample["image"],
            sample["labels_2D"],
            sample["valid_mask"],
        )
        # img, labels = img.to(self.device), labels_2D.to(self.device)

        # variables
        batch_size, H, W = img.shape[0], img.shape[2], img.shape[3]
        self.batch_size = batch_size
        # print("batch_size: ", batch_size)
        Hc = H // self.cell_size
        Wc = W // self.cell_size

        # warped images
        # img_warp, labels_warp_2D, mask_warp_2D = sample['warped_img'].to(self.device), \
        #     sample['warped_labels'].to(self.device), \
        #     sample['warped_valid_mask'].to(self.device)
        img_warp, labels_warp_2D, mask_warp_2D = (
            sample["warped_img"],
            sample["warped_labels"],
            sample["warped_valid_mask"],
        )

        # homographies
        # mat_H, mat_H_inv = \
        # sample['homographies'].to(self.device), sample['inv_homographies'].to(self.device)
        mat_H, mat_H_inv = sample["homographies"], sample["inv_homographies"]

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        if train:
            # print("img: ", img.shape, ", img_warp: ", img_warp.shape)
            outs, outs_warp = (
                self.net(img.to(self.device)),
                self.net(img_warp.to(self.device), subpixel=self.subpixel),
            )
            semi, coarse_desc = outs[0], outs[1]
            semi_warp, coarse_desc_warp = outs_warp[0], outs_warp[1]
        else:
            with torch.no_grad():
                outs, outs_warp = (
                    self.net(img.to(self.device)),
                    self.net(img_warp.to(self.device), subpixel=self.subpixel),
                )
                semi, coarse_desc = outs[0], outs[1]
                semi_warp, coarse_desc_warp = outs_warp[0], outs_warp[1]
                pass

        # detector loss
        ## get labels, masks, loss for detection
        labels3D_in_loss = self.getLabels(labels_2D, self.cell_size, device=self.device)
        mask_3D_flattened = self.getMasks(mask_2D, self.cell_size, device=self.device)
        loss_det = self.get_loss(
            semi, labels3D_in_loss, mask_3D_flattened, device=self.device
        )

        ## warping
        labels3D_in_loss = self.getLabels(
            labels_warp_2D, self.cell_size, device=self.device
        )
        mask_3D_flattened = self.getMasks(
            mask_warp_2D, self.cell_size, device=self.device
        )
        loss_det_warp = self.get_loss(
            semi_warp, labels3D_in_loss, mask_3D_flattened, device=self.device
        )

        mask_desc = mask_3D_flattened.unsqueeze(1)

        # print("mask_desc: ", mask_desc.shape)
        # print("mask_warp_2D: ", mask_warp_2D.shape)

        # descriptor loss

        # if self.desc_loss_type == 'dense':
        loss_desc, mask, positive_dist, negative_dist = self.descriptor_loss(
            coarse_desc,
            coarse_desc_warp,
            mat_H,
            mask_valid=mask_desc,
            device=self.device,
            **self.desc_params
        )

        loss = (
            loss_det + loss_det_warp + self.config["model"]["lambda_loss"] * loss_desc
        )

        if self.subpixel:
            # coarse to dense descriptor
            # work on warped level
            # dense_desc = interpolate_to_dense(coarse_desc_warp, cell_size=self.cell_size) # tensor [batch, 256, H, W]
            dense_map = flattenDetection(semi_warp)  # tensor [batch, 1, H, W]
            # concat image and dense_desc
            concat_features = torch.cat(
                (img_warp.to(self.device), dense_map), dim=1
            )  # tensor [batch, n, H, W]
            # prediction
            # pred_heatmap = self.subpixNet(concat_features.to(self.device)) # tensor [batch, 1, H, W]
            pred_heatmap = outs_warp[2]  # tensor [batch, 1, H, W]
            # print("pred_heatmap: ",  pred_heatmap.shape)
            # add histogram here
            # tensor [batch, channels, H, W]
            # loss
            labels_warped_res = sample["warped_res"]
            # writer.add_histogram(task + '-' + 'warped_res',
            #     labels_warped_res[0,...].clone().cpu().data.numpy().transpose(0,1).transpose(1,2).view(-1, 2),
            #     n_iter)

            # from utils.losses import subpixel_loss
            subpix_loss = self.subpixel_loss_func(
                labels_warp_2D.to(self.device),
                labels_warped_res.to(self.device),
                pred_heatmap.to(self.device),
                patch_size=11,
            )
            # print("subpix_loss: ", subpix_loss)
            # loss += subpix_loss
            # loss = subpix_loss

            # extract the patches from labels
            label_idx = labels_2D[...].nonzero()
            from utils.losses import extract_patches

            patch_size = 32
            patches = extract_patches(
                label_idx.to(self.device),
                img_warp.to(self.device),
                patch_size=patch_size,
            )  # tensor [N, patch_size, patch_size]
            # patches = extract_patches(label_idx.to(device), labels_2D.to(device), patch_size=15) # tensor [N, patch_size, patch_size]
            print("patches: ", patches.shape)

            def label_to_points(labels_res, points):
                labels_res = labels_res.transpose(1, 2).transpose(2, 3).unsqueeze(1)
                points_res = labels_res[
                    points[:, 0], points[:, 1], points[:, 2], points[:, 3], :
                ]  # tensor [N, 2]
                return points_res

            points_res = label_to_points(labels_warped_res, label_idx)

            num_patches_max = 500
            # feed into the network
            pred_res = self.subnet(
                patches[:num_patches_max, ...].to(self.device)
            )  # tensor [1, N, 2]

            # loss function
            def get_loss(points_res, pred_res):
                loss = points_res - pred_res
                loss = torch.norm(loss, p=2, dim=-1).mean()
                return loss

            loss = get_loss(points_res[:num_patches_max, ...].to(self.device), pred_res)

            losses.update({"subpix_loss": subpix_loss})

        self.loss = loss

        losses.update(
            {
                "loss": loss,
                "loss_det": loss_det,
                "loss_det_warp": loss_det_warp,
                "loss_det": loss_det,
                "loss_det_warp": loss_det_warp,
                "positive_dist": positive_dist,
                "negative_dist": negative_dist,
            }
        )
        # print("losses: ", losses)

        if train:
            loss.backward()
            self.optimizer.step()

        self.addLosses2tensorboard(losses, task)
        if n_iter % tb_interval == 0 or task == "val":
            logging.info(
                "current iteration: %d, tensorboard_interval: %d", n_iter, tb_interval
            )
            self.addImg2tensorboard(
                img,
                labels_2D,
                semi,
                img_warp,
                labels_warp_2D,
                mask_warp_2D,
                semi_warp,
                mask_3D_flattened=mask_3D_flattened,
                task=task,
            )

            if self.subpixel:
                # print("only update subpixel_loss")

                self.add_single_image_to_tb(
                    task, pred_heatmap, n_iter, name="subpixel_heatmap"
                )

            self.printLosses(losses, task)

            # if n_iter % tb_interval == 0 or task == 'val':
            # print ("add nms")
            self.add2tensorboard_nms(
                img, labels_2D, semi, task=task, batch_size=batch_size
            )

        return loss.item()

    def saveModel(self):
        """
        # save checkpoint for resuming training
        :return:
        """
        model_state_dict = self.net.module.state_dict()
        save_checkpoint(
            self.save_path,
            {
                "n_iter": self.n_iter + 1,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.loss,
            },
            self.n_iter,
        )
        pass

    def add_single_image_to_tb(self, task, img_tensor, n_iter, name="img"):
        """
        # add image to tensorboard for visualization
        :param task:
        :param img_tensor:
        :param n_iter:
        :param name:
        :return:
        """
        if img_tensor.dim() == 4:
            for i in range(min(img_tensor.shape[0], 5)):
                self.writer.add_image(
                    task + "-" + name + "/%d" % i, img_tensor[i, :, :, :], n_iter
                )
        else:
            self.writer.add_image(task + "-" + name, img_tensor[:, :, :], n_iter)

    # tensorboard
    def addImg2tensorboard(
        self,
        img,
        labels_2D,
        semi,
        img_warp=None,
        labels_warp_2D=None,
        mask_warp_2D=None,
        semi_warp=None,
        mask_3D_flattened=None,
        task="training",
    ):
        """
        # deprecated: add images to tensorboard
        :param img:
        :param labels_2D:
        :param semi:
        :param img_warp:
        :param labels_warp_2D:
        :param mask_warp_2D:
        :param semi_warp:
        :param mask_3D_flattened:
        :param task:
        :return:
        """
        # print("add images to tensorboard")

        n_iter = self.n_iter
        semi_flat = flattenDetection(semi[0, :, :, :])
        semi_warp_flat = flattenDetection(semi_warp[0, :, :, :])

        thd = self.config["model"]["detection_threshold"]
        semi_thd = thd_img(semi_flat, thd=thd)
        semi_warp_thd = thd_img(semi_warp_flat, thd=thd)

        result_overlap = img_overlap(
            toNumpy(labels_2D[0, :, :, :]), toNumpy(semi_thd), toNumpy(img[0, :, :, :])
        )

        self.writer.add_image(
            task + "-detector_output_thd_overlay", result_overlap, n_iter
        )
        saveImg(
            result_overlap.transpose([1, 2, 0])[..., [2, 1, 0]] * 255, "test_0.png"
        )  # rgb to bgr * 255

        result_overlap = img_overlap(
            toNumpy(labels_warp_2D[0, :, :, :]),
            toNumpy(semi_warp_thd),
            toNumpy(img_warp[0, :, :, :]),
        )
        self.writer.add_image(
            task + "-warp_detector_output_thd_overlay", result_overlap, n_iter
        )
        saveImg(
            result_overlap.transpose([1, 2, 0])[..., [2, 1, 0]] * 255, "test_1.png"
        )  # rgb to bgr * 255

        mask_overlap = img_overlap(
            toNumpy(1 - mask_warp_2D[0, :, :, :]) / 2,
            np.zeros_like(toNumpy(img_warp[0, :, :, :])),
            toNumpy(img_warp[0, :, :, :]),
        )

        # writer.add_image(task + '_mask_valid_first_layer', mask_warp[0, :, :, :], n_iter)
        # writer.add_image(task + '_mask_valid_last_layer', mask_warp[-1, :, :, :], n_iter)
        ##### print to check
        # print("mask_2D shape: ", mask_warp_2D.shape)
        # print("mask_3D_flattened shape: ", mask_3D_flattened.shape)
        for i in range(self.batch_size):
            if i < 5:
                self.writer.add_image(
                    task + "-mask_warp_origin", mask_warp_2D[i, :, :, :], n_iter
                )
                self.writer.add_image(
                    task + "-mask_warp_3D_flattened", mask_3D_flattened[i, :, :], n_iter
                )
        # self.writer.add_image(task + '-mask_warp_origin-1', mask_warp_2D[1, :, :, :], n_iter)
        # self.writer.add_image(task + '-mask_warp_3D_flattened-1', mask_3D_flattened[1, :, :], n_iter)
        self.writer.add_image(task + "-mask_warp_overlay", mask_overlap, n_iter)

    def tb_scalar_dict(self, losses, task="training"):
        """
        # add scalar dictionary to tensorboard
        :param losses:
        :param task:
        :return:
        """
        for element in list(losses):
            self.writer.add_scalar(task + "-" + element, losses[element], self.n_iter)
            # print (task, '-', element, ": ", losses[element].item())

    def tb_images_dict(self, task, tb_imgs, max_img=5):
        """
        # add image dictionary to tensorboard
        :param task:
            str (train, val)
        :param tb_imgs:
        :param max_img:
            int - number of images
        :return:
        """
        for element in list(tb_imgs):
            for idx in range(tb_imgs[element].shape[0]):
                if idx >= max_img:
                    break
                # print(f"element: {element}")
                self.writer.add_image(
                    task + "-" + element + "/%d" % idx,
                    tb_imgs[element][idx, ...],
                    self.n_iter,
                )


    def tb_hist_dict(self, task, tb_dict):
        for element in list(tb_dict):
            self.writer.add_histogram(
                task + "-" + element, tb_dict[element], self.n_iter
            )
        pass

    def printLosses(self, losses, task="training"):
        """
        # print loss for tracking training
        :param losses:
        :param task:
        :return:
        """
        for element in list(losses):
            # print ('add to tb: ', element)
            print(task, "-", element, ": ", losses[element].item())

    def add2tensorboard_nms(self, img, labels_2D, semi, task="training", batch_size=1):
        """
        # deprecated:
        :param img:
        :param labels_2D:
        :param semi:
        :param task:
        :param batch_size:
        :return:
        """
        from utils.utils import getPtsFromHeatmap
        from utils.utils import box_nms

        boxNms = False
        n_iter = self.n_iter

        nms_dist = self.config["model"]["nms"]
        conf_thresh = self.config["model"]["detection_threshold"]
        # print("nms_dist: ", nms_dist)
        precision_recall_list = []
        precision_recall_boxnms_list = []
        for idx in range(batch_size):
            semi_flat_tensor = flattenDetection(semi[idx, :, :, :]).detach()
            semi_flat = toNumpy(semi_flat_tensor)
            semi_thd = np.squeeze(semi_flat, 0)
            pts_nms = getPtsFromHeatmap(semi_thd, conf_thresh, nms_dist)
            semi_thd_nms_sample = np.zeros_like(semi_thd)
            semi_thd_nms_sample[
                pts_nms[1, :].astype(np.int), pts_nms[0, :].astype(np.int)
            ] = 1

            label_sample = torch.squeeze(labels_2D[idx, :, :, :])
            # pts_nms = getPtsFromHeatmap(label_sample.numpy(), conf_thresh, nms_dist)
            # label_sample_rms_sample = np.zeros_like(label_sample.numpy())
            # label_sample_rms_sample[pts_nms[1, :].astype(np.int), pts_nms[0, :].astype(np.int)] = 1
            label_sample_nms_sample = label_sample

            if idx < 5:
                result_overlap = img_overlap(
                    np.expand_dims(label_sample_nms_sample, 0),
                    np.expand_dims(semi_thd_nms_sample, 0),
                    toNumpy(img[idx, :, :, :]),
                )
                self.writer.add_image(
                    task + "-detector_output_thd_overlay-NMS" + "/%d" % idx,
                    result_overlap,
                    n_iter,
                )
            assert semi_thd_nms_sample.shape == label_sample_nms_sample.size()
            precision_recall = precisionRecall_torch(
                torch.from_numpy(semi_thd_nms_sample), label_sample_nms_sample
            )
            precision_recall_list.append(precision_recall)

            if boxNms:
                semi_flat_tensor_nms = box_nms(
                    semi_flat_tensor.squeeze(), nms_dist, min_prob=conf_thresh
                ).cpu()
                semi_flat_tensor_nms = (semi_flat_tensor_nms >= conf_thresh).float()

                if idx < 5:
                    result_overlap = img_overlap(
                        np.expand_dims(label_sample_nms_sample, 0),
                        semi_flat_tensor_nms.numpy()[np.newaxis, :, :],
                        toNumpy(img[idx, :, :, :]),
                    )
                    self.writer.add_image(
                        task + "-detector_output_thd_overlay-boxNMS" + "/%d" % idx,
                        result_overlap,
                        n_iter,
                    )
                precision_recall_boxnms = precisionRecall_torch(
                    semi_flat_tensor_nms, label_sample_nms_sample
                )
                precision_recall_boxnms_list.append(precision_recall_boxnms)

        precision = np.mean(
            [
                precision_recall["precision"]
                for precision_recall in precision_recall_list
            ]
        )
        recall = np.mean(
            [precision_recall["recall"] for precision_recall in precision_recall_list]
        )
        self.writer.add_scalar(task + "-precision_nms", precision, n_iter)
        self.writer.add_scalar(task + "-recall_nms", recall, n_iter)
        print(
            "-- [%s-%d-fast NMS] precision: %.4f, recall: %.4f"
            % (task, n_iter, precision, recall)
        )
        if boxNms:
            precision = np.mean(
                [
                    precision_recall["precision"]
                    for precision_recall in precision_recall_boxnms_list
                ]
            )
            recall = np.mean(
                [
                    precision_recall["recall"]
                    for precision_recall in precision_recall_boxnms_list
                ]
            )
            self.writer.add_scalar(task + "-precision_boxnms", precision, n_iter)
            self.writer.add_scalar(task + "-recall_boxnms", recall, n_iter)
            print(
                "-- [%s-%d-boxNMS] precision: %.4f, recall: %.4f"
                % (task, n_iter, precision, recall)
            )

    def get_heatmap(self, semi, det_loss_type="softmax"):
        if det_loss_type == "l2":
            heatmap = self.flatten_64to1(semi)
        else:
            heatmap = flattenDetection(semi)
        return heatmap

    ######## static methods ########
    @staticmethod
    def input_to_imgDict(sample, tb_images_dict):
        # for e in list(sample):
        #     print("sample[e]", sample[e].shape)
        #     if (sample[e]).dim() == 4:
        #         tb_images_dict[e] = sample[e]
        for e in list(sample):
            element = sample[e]
            if type(element) is torch.Tensor:
                if element.dim() == 4:
                    tb_images_dict[e] = element
                # print("shape of ", i, " ", element.shape)
        return tb_images_dict

    @staticmethod
    def interpolate_to_dense(coarse_desc, cell_size=8):
        dense_desc = nn.functional.interpolate(
            coarse_desc, scale_factor=(cell_size, cell_size), mode="bilinear"
        )
        # norm the descriptor
        def norm_desc(desc):
            dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
            desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
            return desc

        dense_desc = norm_desc(dense_desc)
        return dense_desc


if __name__ == "__main__":
    # load config
    filename = "configs/superpoint_coco_test.yaml"
    import yaml

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_default_tensor_type(torch.FloatTensor)
    with open(filename, "r") as f:
        config = yaml.load(f)

    from utils.loader import dataLoader as dataLoader

    # data = dataLoader(config, dataset='hpatches')
    task = config["data"]["dataset"]

    data = dataLoader(config, dataset=task, warp_input=True)
    # test_set, test_loader = data['test_set'], data['test_loader']
    train_loader, val_loader = data["train_loader"], data["val_loader"]

    # model_fe = Train_model_frontend(config)
    # print('==> Successfully loaded pre-trained network.')

    train_agent = Train_model_frontend(config, device=device)

    train_agent.train_loader = train_loader
    # train_agent.val_loader = val_loader

    train_agent.loadModel()
    train_agent.dataParallel()
    train_agent.train()

    # epoch += 1
    try:
        model_fe.train()
    # catch exception
    except KeyboardInterrupt:
        logging.info("ctrl + c is pressed. save model")
    # is_best = True
