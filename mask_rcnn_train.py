from torch._C import dtype
from torch.optim import optimizer
from dataset import MaskRCNNTrainDataset, collate_fn
from model import get_model_instance_segmentation, get_optimizer, load_model, get_scheduler
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from mask_rcnn_eval import drawbox
from torchvision import transforms


def DataToDevice(images, targets, device):
    # print(f'images type: {type(images)}, targets type: {type(targets)}')
    # >> images type: <class 'tuple'>, targets type: <class 'tuple'>
    images = list(images)
    for i in range(len(images)):
        images[i] = images[i].to(device).type(torch.cuda.FloatTensor)
        for key in targets[i].keys():
            # boxes, area
            if targets[i][key].dtype == torch.float32:
                datatype = torch.cuda.FloatTensor
            # masks, label, image_id, iscrowd
            else:
                datatype = torch.cuda.LongTensor
            # This step will cause out of cuda memory
            targets[i][key] = targets[i][key].to(device).type(datatype)
    return tuple(images), targets


def getemptytotalloss():
    total_train_loss = {
        'total_loss': 0.0,
        'loss_classifier': 0.0,
        'loss_box_reg': 0.0,
        'loss_mask': 0.0,
        'loss_objectness': 0.0,
        'loss_rpn_box_reg': 0.0,
    }
    return total_train_loss


def getmessage(loss_dict, total_train_loss, batch_index):
    message = []
    for key, loss in loss_dict.items():
        total_train_loss[key] += loss
        message.append(f'{key}: {total_train_loss[key]/(batch_index+1):.2f}')
    message = ' '.join(message)

    return message, total_train_loss


def getlosses(loss_dict):
    return sum(loss for loss in loss_dict.values())


def train(date):
    num_epochs = 100  # one epoch need 5min, 100 epochs about 8 hr
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    # model = get_model_instance_segmentation(num_classes=2)
    model_path = os.path.join('model', 'Mask R-CNN',
                              '2021-12-09 21-03-21-epoch-5.pkl')
    model = load_model(model_path)
    model.to(device)

    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)

    # train_dir = os.path.join('dataset', 'train')
    # train_dir = os.path.join('dataset', 'test_dataloader')
    train_dir = os.path.join('dataset', 'divide-2')
    dataset = MaskRCNNTrainDataset(train_dir)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, collate_fn=collate_fn)

    best_total_loss = float('inf')

    for i in range(1, 1+num_epochs):
        model.train()
        total_train_loss = getemptytotalloss()
        pbar = tqdm(dataloader)
        for batch_index, (batch_image, batch_target, batch_image_name) in enumerate(pbar):

            # pil_image = transforms.ToPILImage()(torch.squeeze(batch_image[0]))
            # pil_image = drawbox(batch_target[0]['boxes'][0], pil_image)
            # pil_image.show()
            # pil_mask = transforms.ToPILImage()(
            #     torch.squeeze(batch_target[0]['masks'][0].type(torch.float32)))
            # pil_mask = drawbox(batch_target[0]['boxes'][0], pil_mask)
            # pil_mask.show()
            # break

            x, y = DataToDevice(batch_image, batch_target, device)

            del batch_image
            del batch_target

            loss_dict = model(x, y)

            del x
            del y

            # losses = sum(loss for loss in loss_dict.values())
            losses = getlosses(loss_dict)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            lr = scheduler.get_last_lr()[0]

            # loss_value = losses.item()
            # total_train_loss['total_loss'] += loss_value
            total_train_loss['total_loss'] += float(losses)
            del losses

            message, total_train_loss = getmessage(
                loss_dict, total_train_loss, batch_index)

            del loss_dict

            pbar.set_description(f'Epoch: {i}, lr: {lr:.0e}, {message}')

        scheduler.step()

        save_model_path = os.path.join('model', 'Mask R-CNN')
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)

        if total_train_loss['total_loss'] < best_total_loss:
            best_total_loss = total_train_loss['total_loss']
            print(f'Save model at epoch {i}.')
            torch.save(model, os.path.join(
                save_model_path, f'{date}-epoch-{i}.pkl'))

        del total_train_loss
