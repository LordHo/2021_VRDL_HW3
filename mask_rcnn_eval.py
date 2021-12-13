from dataset import MaskRCNNEvalDivideDataset
import PIL
import torch
import os
from PIL import Image, ImageDraw
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import MaskRCNNEvalDataset
from torch.utils.data import DataLoader

import json
import time

from pycocotools.mask import encode, decode


def drawbox(box, image):
    draw = ImageDraw.Draw(image)
    draw.rectangle([box[0], box[1], box[2], box[3]], outline="red")
    return image


def load_model(model_path):
    return torch.load(model_path)


def createdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


@torch.no_grad()
def eval(date, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model_path = os.path.join('model', 'Mask R-CNN',
    #   '2021-12-09 12-36-55-epoch-98.pkl')

    model = load_model(model_path)
    # model.box_detections_per_img = 400
    model.eval()
    model.to(device)

    eval_image_dir = os.path.join('dataset', 'test')
    dataset = MaskRCNNEvalDataset(eval_image_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for (image, image_name) in tqdm(dataloader):
        image_name_ = image_name[0].split('.')[0]
        result_dir = os.path.join('result', date, image_name_)
        createdir(result_dir)
        pil_image = transforms.ToPILImage()(torch.squeeze(image))

        pred = model(image.to(device))

        boxes = pred[0]["boxes"]
        masks = pred[0]["masks"]
        scores = pred[0]["scores"]

        mask_dir = os.path.join(result_dir, 'masks')
        createdir(mask_dir)

        for index, (box, mask, score) in enumerate(zip(boxes, masks, scores)):
            # print(f'score: {score}')
            if score < 0:
                continue
            else:
                # pil_mask = transforms.ToPILImage()(torch.squeeze(mask))
                # numpy_mask = np.array(pil_mask, dtype=np.uint8)
                # if len(np.unique(numpy_mask)) > 1:
                #     pil_mask.save(os.path.join(mask_dir, f'mask_{index}.png'))
                #     image = drawbox(box, pil_image)

                numpy_mask = torch.squeeze(mask).cpu().numpy().astype(np.uint8)
                if len(np.unique(numpy_mask)) > 1:
                    rle_mask = encode(np.asfortranarray(numpy_mask))
                    after_rle_mask = decode(rle_mask)
                    after_rle_mask = (after_rle_mask*255).astype(np.uint8)
                    pil_mask = Image.fromarray(after_rle_mask)
                    pil_mask.save(os.path.join(mask_dir, f'mask_{index}.png'))
                    image = drawbox(box, pil_image)

        image.save(os.path.join(result_dir, f'{image_name_}.png'))


def RLE(numpy_mask):
    encode_result = encode(numpy_mask)
    # encode_result['counts'] = str(encode_result['counts'])
    encode_result['counts'] = encode_result['counts'].decode("utf-8")
    return encode_result


def getemptyanswer(image_id=None, image_size=None):
    answer = {}
    if image_id is None:
        answer['image_id'] = -1
    else:
        answer['image_id'] = image_id
    answer['bbox'] = [-1., -1., -1., -1.]
    answer['score'] = -1.
    answer['category_id'] = 1  # Alawys set to 1
    if image_size is None:
        answer['segmenation'] = {'size': [-1, -1], 'counts': ""}
    else:
        answer['segmenation'] = {'size': image_size, 'counts': ""}
    return answer


def loadtestimageinfo():
    json_path = os.path.join('dataset', 'test_img_ids.json')
    f = open(json_path, 'r')
    test_image_info = json.load(f)
    f.close()
    return test_image_info


def getcocobox(box):
    coco_box = [box[0], box[1], (box[2]-box[0]), (box[3]-box[1])]
    return coco_box


@torch.no_grad()
def eval_coco(date, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path)
    # model.box_detections_per_img = 400

    model.eval()
    model.to(device)

    eval_image_dir = os.path.join('dataset', 'test')
    dataset = MaskRCNNEvalDataset(eval_image_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    answers = []
    test_image_info = loadtestimageinfo()

    for (image, image_name) in tqdm(dataloader):
        image_id = None
        image_size = None
        for image_info in test_image_info:
            if str(image_name[0]) == str(image_info['file_name']):
                image_id = int(image_info['id'])
                image_size = [
                    int(image_info['width']), int(image_info['height'])]
                print(image_id, image_size)
                break

        # image_name_ = image_name[0].split('.')[0]
        # result_dir = os.path.join('result', date, image_name_)
        # createdir(result_dir)
        # pil_image = transforms.ToPILImage()(torch.squeeze(image))

        pred = model(image.to(device))

        boxes = pred[0]["boxes"]
        masks = pred[0]["masks"]
        scores = pred[0]["scores"]

        # mask_dir = os.path.join(result_dir, 'masks')
        # createdir(mask_dir)
        for index, (box, mask, score) in enumerate(zip(boxes, masks, scores)):
            answer = getemptyanswer(image_id, image_size)

            # print(f'score: {score}')
            if score < 0:
                continue
            else:
                # pil_mask = transforms.ToPILImage()(torch.squeeze(mask))
                # numpy_mask = np.array(pil_mask)

                # if len(np.unique(numpy_mask)) > 1:
                #     answer['score'] = float(score.cpu().numpy())

                #     answer['bbox'] = getcocobox(box.cpu().numpy().tolist())
                #     numpy_mask = np.asfortranarray(numpy_mask)
                #     answer['segmenation'] = RLE(numpy_mask)
                #     answers.append(answer)

                numpy_mask = torch.squeeze(mask).cpu().numpy().astype(np.uint8)
                if len(np.unique(numpy_mask)) > 1:
                    answer['score'] = float(score.cpu().numpy())
                    answer['bbox'] = getcocobox(box.cpu().numpy().tolist())

                    # rle_mask = encode(np.asfortranarray(numpy_mask))
                    rle_mask = RLE(np.asfortranarray(numpy_mask))
                    answer['segmenation'] = rle_mask
                    answers.append(answer)

    answers = sorted(answers, key=lambda x: x['image_id'])
    dir_path = os.path.join('answers', f'answer-{date}')
    createdir(dir_path)
    print(json.dumps(answers, indent=4), file=open(
        os.path.join(dir_path, f'answer.json'), 'w'))


def getarea(box):
    return int((box[2]-box[0]) * (box[3]-box[1]))


@torch.no_grad()
def eval_divide(date, eval_image_dir, model_path, threshold=0.9, minarea=15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model_path = os.path.join('model', 'Mask R-CNN',
    #   '2021-12-09 12-36-55-epoch-98.pkl')

    model = load_model(model_path)
    # model.box_detections_per_img = 400
    model.eval()
    model.to(device)

    # eval_image_dir = os.path.join('dataset', 'test')
    # eval_image_dir = os.path.join('dataset', 'test_only_one_image')
    # dataset = MaskRCNNEvalDivideDataset(eval_image_dir, divide=2)
    dataset = MaskRCNNEvalDivideDataset(eval_image_dir, divide=4)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for (images, partitions, image_name) in tqdm(dataloader):
        pil_image_list = [transforms.ToPILImage()(torch.squeeze(image))
                          for image in images]
        whole_image = dataset.__mergecropimages__(
            pil_image_list, partitions, 'RGB')
        whole_mask = None

        image_name_ = image_name[0].split('.')[0]
        result_dir = os.path.join('result', date, image_name_)
        createdir(result_dir)
        mask_dir = os.path.join(result_dir, 'masks')
        createdir(mask_dir)

        image_list = []
        for out_index, (image, partition) in enumerate(zip(images, partitions)):
            pil_image = transforms.ToPILImage()(torch.squeeze(image))

            pred = model(image.to(device))

            boxes = pred[0]["boxes"]
            masks = pred[0]["masks"]
            scores = pred[0]["scores"]

            for index, (box, mask, score) in enumerate(zip(boxes, masks, scores)):
                # print(f'score: {score}')
                # if score < 0.999995:
                if score < threshold:
                    continue
                else:
                    # pil_mask = transforms.ToPILImage()(torch.squeeze(mask))
                    # numpy_mask = np.array(pil_mask, dtype=np.uint8)
                    # if len(np.unique(numpy_mask)) > 1:
                    #     pil_mask.save(os.path.join(mask_dir, f'mask_{index}.png'))
                    #     image = drawbox(box, pil_image)

                    numpy_mask = torch.squeeze(
                        mask).cpu().numpy()
                    numpy_mask[numpy_mask > 0.3] = 1
                    numpy_mask = numpy_mask.astype(np.uint8)
                    if len(np.unique(numpy_mask)) > 1 and getarea(box) > minarea:
                        # rle_mask = encode(np.asfortranarray(numpy_mask))
                        rle_mask = RLE(np.asfortranarray(numpy_mask))
                        after_rle_mask = decode(rle_mask)
                        after_rle_mask = (after_rle_mask*255).astype(np.uint8)
                        pil_mask = Image.fromarray(after_rle_mask)
                        pil_mask = dataset.__resizecropmask__(
                            pil_mask, partition[0], partitions, 'L')
                        pil_mask.save(os.path.join(
                            mask_dir, f'mask_{out_index}_{index}.png'))

                        # box = getresizebox(box, partition)
                        print(f'area: {getarea(box)}')
                        box = getcocobox(box)
                        box = getresizecocobox(box, partition)
                        box[2] += box[0]
                        box[3] += box[1]

                        # image = drawbox(box, pil_image)
                        whole_image = drawbox(box, whole_image)

                        if whole_mask is None:
                            whole_mask = np.array(pil_mask)
                        else:
                            if partition[0] in [1, 2]:
                                whole_mask += np.array(pil_mask)
                            else:
                                whole_mask += np.array(pil_mask)

            # image.save(os.path.join(result_dir, f'{image_name_}.png'))
            # image_list.append(image)
        # whole_image = dataset.__mergecropimages__(
        #     image_list, partitions, 'RGB')
        whole_mask = Image.fromarray(whole_mask)
        whole_mask.save(os.path.join(
            result_dir, f'{image_name_}_whole_mask.png'))
        whole_image.save(os.path.join(result_dir, f'{image_name_}.png'))


def getcocoboxarea(coco_box):
    return int((coco_box[2]*coco_box[3]))


def getresizebox(box, partition):
    area = [i.item() for i in partition[1]]
    left, top = area[0], area[1]
    resize_box = [box[0]+left, box[1]+top, box[2]+left, box[3]+top]
    return resize_box


def getresizecocobox(coco_box, partition):
    # print(partition)
    area = [i.item() for i in partition[1]]
    # print(area)
    left, top = area[0], area[1]
    resize_coco_box = [coco_box[0]+left,
                       coco_box[1]+top, coco_box[2], coco_box[3]]
    return resize_coco_box


@torch.no_grad()
def eval_coco_divide(date, eval_image_dir, model_path, threshold=0.9, minarea=15, return_answers=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path)

    model.eval()
    model.to(device)

    # eval_image_dir = os.path.join('dataset', 'test')
    # eval_image_dir = os.path.join('dataset', 'test_only_one_image')
    # dataset = MaskRCNNEvalDivideDataset(eval_image_dir, divide=2)
    dataset = MaskRCNNEvalDivideDataset(eval_image_dir, divide=4)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    answers = []
    test_image_info = loadtestimageinfo()

    for (images, partitions, image_name) in tqdm(dataloader):
        image_id = None
        image_size = None
        for image_info in test_image_info:
            if str(image_name[0]) == str(image_info['file_name']):
                image_id = int(image_info['id'])
                image_size = [
                    int(image_info['width']), int(image_info['height'])]
                print(image_id, image_size)
                break

        for image, partition in zip(images, partitions):

            pred = model(image.to(device))

            boxes = pred[0]["boxes"]
            masks = pred[0]["masks"]
            scores = pred[0]["scores"]

            for index, (box, mask, score) in enumerate(zip(boxes, masks, scores)):
                answer = getemptyanswer(image_id, image_size)

                # print(f'score: {score}')
                # threahold = 0.9999
                if score < threshold:
                    continue
                else:
                    # numpy_mask = torch.squeeze(
                    #     mask).cpu().numpy().astype(np.uint8)
                    numpy_mask = torch.squeeze(mask).cpu().numpy()
                    numpy_mask[numpy_mask > 0.3] = 1
                    numpy_mask = numpy_mask.astype(np.uint8)

                    pil_mask = Image.fromarray(numpy_mask)
                    pil_mask = dataset.__resizecropmask__(
                        pil_mask, partition[0], partitions, 'L')
                    numpy_mask = np.array(pil_mask)
                    # print(np.unique(numpy_mask))
                    # print(numpy_mask.shape)
                    if len(np.unique(numpy_mask)) > 1 and getcocoboxarea(box) > minarea:
                        answer['score'] = float(score.cpu().numpy())
                        answer['bbox'] = getcocobox(box.cpu().numpy().tolist())
                        answer['bbox'] = getresizecocobox(
                            answer['bbox'], partition)

                        rle_mask = RLE(np.asfortranarray(numpy_mask))
                        answer['segmenation'] = rle_mask
                        answers.append(answer)
                        # print(answer)
    if return_answers:
        return answers
    else:
        answers = sorted(answers, key=lambda x: x['image_id'])
        dir_path = os.path.join('answers', f'answer-{date}')
        createdir(dir_path)
        print(json.dumps(answers, indent=4), file=open(
            os.path.join(dir_path, f'answer.json'), 'w'))


def mergeanswers(answers_list, date):
    mergeed_answers = None
    for answers in answers_list:
        if mergeed_answers is None:
            mergeed_answers = answers
        else:
            mergeed_answers = mergeed_answers + answers
    mergeed_answers = sorted(mergeed_answers, key=lambda x: x['image_id'])
    dir_path = os.path.join('answers', f'answer-{date}')
    createdir(dir_path)
    print(json.dumps(mergeed_answers, indent=4), file=open(
        os.path.join(dir_path, f'answer.json'), 'w'))
