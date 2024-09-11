import torch
import torch.nn as nn
import torch.nn.functional as F
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)
import time
from pytorch3d.loss import chamfer_distance
from dataset import MVP_CP
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive Agg backend
# matplotlib.use('TkAgg')  # Or 'PDF' or 'SVG'


from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from Models import Model
import os
import csv


#######################################################################################

BATCH_SIZE = 14
NUM_POINTS = 2048
MODEL_PATH = "models_2"
PLOT_PATH = "plots_2"
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
if not os.path.exists(PLOT_PATH):
    os.mkdir(PLOT_PATH)
epoch_start = 28
checkpoint_path = f"{MODEL_PATH}/model_epoch_{epoch_start}.pth"

LEARNING_RATE = 3e-5
EPOCHS = 80

# create a csv file to store the logs locally 
# Open the CSV file and create a CSV writer object
csv_file_path = f'local_train_logs_for_{MODEL_PATH}'
with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(["global_step", "epoch", "iter", "loss", "time", "learning_rate", "tag"])



plot_csv = f"local_train_image_plot_{MODEL_PATH}"
with open(plot_csv, mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(["i", 'x', 'y', 'z'])



#######################################################################################

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", filename="train2.log", filemode="a")
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.info("-"*80)
logger.info("Starting training")

#######################################################################################

import wandb
run = wandb.init(
    # Set the project where this run will be logged
    project="xlstm_cv",
    name="main2", 
    id="123458",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
    },
)
logger.info(f"Wandb initialized with project_name : {run.project_name}, run_name : {run.name}, {run.id}")


#######################################################################################
    
# def createdataset(batchsize=1, number_of_points=100):
#    return torch.rand(batchsize, number_of_points, 3)

# def plot(pc, name="plot"):
#     x = pc[:, 0]
#     y = pc[:, 1]
#     z = pc[:, 2]
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(x, y, z)

# def plot_f(pc, name="plot", writer2=None):
#     try:
#         x = pc[:, 0]
#         y = pc[:, 1]
#         z = pc[:, 2]
        
#         fig = plt.figure(figsize=(8, 8))
#         ax = fig.add_subplot(111, projection='3d')
#         ax.scatter(x, y, z, c='b')

#         ax.set_xlabel('X Axis')
#         ax.set_ylabel('Y Axis')
#         # ax.set_zlabel('Z Axis')
#         ax.set_title(f'3D Scatter Plot - {name}')

#         plt.savefig(f"{PLOT_PATH}/{name}.png")
#     except:
#         writer2.writerow([name, x, y, z])

#         logger.error('Error plotting image')


###################################################################################
from utils.train_utils import *
import math

metrics = ['dcd', 'cd_t', 'f1']
best_epoch_losses = {m: (0, 0) if m == 'f1' else (0, math.inf) for m in metrics}
train_loss_meter = AverageValueMeter()
val_loss_meters = {m: AverageValueMeter() for m in metrics}

###################################################################################

dataset = MVP_CP(prefix="train")


# subset the dataset
# logger.info("Subsetting the dataset to 1000")
# dataset = torch.utils.data.Subset(dataset, range(100))
logger.info(f"Dataset size: {len(dataset)}")

train, test = torch.utils.data.random_split(dataset, [0.8,0.2])
logger.info(f"Train dataset size: {len(train)}")
logger.info(f"Test dataset size: {len(test)}")
logger.info(f'Batch size: {BATCH_SIZE}')
logger.info(f"Batches in train: {len(train)/BATCH_SIZE}")
logger.info(f"Batches in test: {len(test)/BATCH_SIZE}")
dataloader_train = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
dataloader_test = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE)
#------------------------------------------------------------------------------------------
# model = ModelEncoder2(2048, scale=1, emb_dim=16).to("cuda")
model = Model(NUM_POINTS).to("cuda")
logger.info("Model loaded")
logger.info(f'{"-"*50}\n{model}')

# Construct the checkpoint file path

# Check if the checkpoint exists
if os.path.exists(checkpoint_path):
    # Load the state dictionary from the checkpoint
    model.load_state_dict(torch.load(checkpoint_path))
    logger.info(f"Model loaded from {checkpoint_path}")
else:
    logger.error(f"No checkpoint found at {checkpoint_path}")
    raise Exception("No checkpoint found")
logger.info("-------------------------------------------------------------------")
#------------------------------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

lrs = []


start = time.time()
trigger = 10

logger.info(f"Epochs {EPOCHS}")
logger.info("Starting training")

global_step = 0


with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    with open(plot_csv, mode='a', newline='') as file2:
        writer2 = csv.writer(file2)

        for epoch in range(epoch_start+1, EPOCHS):
            logger.info(f"Epoch {epoch} started at {time.ctime()}")
            train_loss_meter.reset()
            model.train()

            losses_per_epoch = {} # for plotting loss per epoch
            for i, (label,ic,c) in enumerate(dataloader_train):
                global_step += 1
                optimizer.zero_grad()
                ic = ic.to("cuda").transpose(2,1).contiguous()
                # a = model(ic)
                ac, af = model(ic)
                c = c.to("cuda")
                l = chamfer_distance(ac ,c, batch_reduction="sum", point_reduction="sum")[0] + (epoch/EPOCHS + 0.05)*chamfer_distance(af ,c, batch_reduction="sum", point_reduction="sum")[0]
                losses_per_epoch[i] = l.mean().item()
                #--------------------------------------------------
                train_loss_meter.update(l.mean().item())
                #--------------------------------------------------
                if not torch.isnan(l).item():
                    l.backward()
                    optimizer.step()
                    lrs.append(optimizer.param_groups[0]["lr"])
                    scheduler.step()

                if global_step%100 == 0:
                    logger.info(f"Epoch and iter {epoch} {i} Loss {l.item()} Time {time.time()-start}")
                    start = time.time()
                    # logging to wandb
                    log_data = {
                        "loss": l.item(),
                        "epoch": epoch,
                        "iter": i,
                        "time": time.time()-start,
                        "learning_rate": optimizer.param_groups[0]['lr']
                    }
                    # ["global_step", "epoch", "iter", "loss", "time", "learning_rate"]
                    writer.writerow([global_step, epoch, i, l.item(), time.time() - start, optimizer.param_groups[0]['lr'], f"train_ep_{epoch}"])
                    run.log(log_data, step = global_step)

                    if torch.isnan(l).item():
                        trigger -= 1
                        logger.info("Saving model with nan with trigger remaining", trigger)
                        torch.save(model.state_dict(), f"{MODEL_PATH}/modelnan.pth")
                        cd = 0
                        model.eval()
                        for i, (label, ic, c) in enumerate(dataloader_test):
                            with torch.no_grad():
                                ic = ic.to("cuda").transpose(2,1).contiguous()
                                ac, af = model(ic)
                                c = c.to("cuda")
                                l = chamfer_distance(af ,c, batch_reduction="mean", point_reduction="mean")[0]
                                # logger.info(f'TEST l.item() : {l.item()}') 
                                log_data = {
                                    "TEST loss": l.item()
                                }
                                run.log(log_data, step = i)
                                cd += l.item()
                        logger.info(f"Average chamfer distance on val {cd/(i+1)}")
                        log_data = {
                            "TEST Average chamfer distance": cd/(i+1)
                        }
                        run.log(log_data, step = global_step)
                        if trigger == 0:
                            exit()
            
            #------------------------------- plot the loss
            # Assuming losses_per_epoch is a dictionary with epoch numbers as keys and losses as values
            epoch_list = list(losses_per_epoch.keys())
            loss_list = list(losses_per_epoch.values())
            try:
                plt.figure(figsize=(10, 5))
                plt.plot(epoch_list, loss_list, marker='o', linestyle='-', color='b')

                plt.title('Loss per Epoch')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.grid(True)
                # Save the plot as an image file
                # plt.savefig(f"{PLOT_PATH}/loss_per_epoch_{epoch}.png")
                plt.savefig(f"{PLOT_PATH}/loss_per_epoch_{epoch}.pdf")
                plt.show()
            except:
                logger.error(f'Error while plotting the image')

            #------------------------------- save model
            logger.info(f"Saving this epoch  {epoch} +1")
            torch.save(model.state_dict(), f"{MODEL_PATH}/model_epoch_{epoch+1}.pth")    
            
            #------------------------------- eval step
            cd = 0
            model.eval()
            logger.info("Testing------------------------------------")
            for i, (label, ic, c) in enumerate(dataloader_test):
                with torch.no_grad():
                    ic = ic.to("cuda").transpose(2,1).contiguous()
                    ac, af = model(ic)
                    c = c.to("cuda")
                    l = chamfer_distance(af ,c, batch_reduction="mean", point_reduction="mean")[0]
                    # logger.info(f'TEST l.item() : {l.item()}') 
                    log_data = {
                        "TEST loss": l.item()
                    }

                    writer.writerow([global_step, epoch, i, l.item(), time.time() - start, optimizer.param_groups[0]['lr'], f"test_ep_{epoch}"])
                    run.log(log_data, step = i)
                    cd += l.item()
            logger.info(f"Average chamfer distance on val {cd/(i+1)}")
            log_data = {
                "TEST Average chamfer distance": cd/(i+1)
            }
            run.log(log_data, step = global_step)
            
            logger.info(f"Plotting the results at epoch {epoch}")
            gpc = af.detach().cpu().numpy()[0]
            # plot_f(gpc, name=f"gpc_epoch_{epoch}", writer2=writer2)
            try:
                x = gpc[:, 0]
                y = gpc[:, 1]
                z = gpc[:, 2]
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(x, y, z, c='b')
                ax.set_xlabel('X Axis')
                ax.set_ylabel('Y Axis')
                ax.set_title(f'3D Scatter Plot - "gpc_epoch_{epoch}')
                plt.savefig(f"{PLOT_PATH}/gpc_epoch_{epoch}.png")
            except:
                writer2.writerow([f"gpc_epoch_{epoch}", x, y, z])
                logger.error('Error plotting image')
            
            c = c.detach().cpu().numpy()[0]
            # plot_f(c, name=f"c_epoch_{epoch}", writer2=writer2)
            try:
                x = c[:, 0]
                y = c[:, 1]
                z = c[:, 2]
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(x, y, z, c='b')
                ax.set_xlabel('X Axis')
                ax.set_ylabel('Y Axis')
                ax.set_title(f'3D Scatter Plot - "c_epoch_{epoch}')
                plt.savefig(f"{PLOT_PATH}/c_epoch_{epoch}.png")
            except:
                writer2.writerow([f"c_epoch_{epoch}", x, y, z])
                logger.error('Error plotting image')
        try:
            plt.plot(lrs)
        except:
            logger.error("Error plotting LRS")

logger.info("Saving model final")
torch.save(model.state_dict(), f"{MODEL_PATH}/model.pth")
cd = 0
model.eval()
logger.info("Testing")
for i, (label, ic, c) in enumerate(dataloader_test):
    with torch.no_grad():
        ic = ic.to("cuda").transpose(2,1).contiguous()
        ac, af = model(ic)
        c = c.to("cuda")
        l = chamfer_distance(af ,c, batch_reduction="mean", point_reduction="mean")[0]
        logger.info(f'l.item() : {l.item()}')
        cd += l.item()

logger.info(f"Average chamfer distance on val {cd/(i+1)}")
run.finish()
