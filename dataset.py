from torch.utils.data import Dataset, DataLoader
import glob
import os
from PIL import Image
import numpy as np
import torch
import torchvision


def collate_fn(batch):
    return tuple(zip(*batch))


class AbstractDataset(Dataset):
    def __init__(self, dir) -> None:
        super().__init__()
        self.dir = dir

    def __len__(self):
        raise NotImplementedError


class MaskRCNNTrainDataset(AbstractDataset):
    def __init__(self, dir) -> None:
        super().__init__(dir)
        self.image_list = self.__getimagelist__()
        self.PILToTensor = torchvision.transforms.PILToTensor()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index % self.__len__()]
        image = self.__getimage__(image_name)
        targets = self.__gettargets__(image_name, index)
        return image, targets, image_name

    def __getimagelist__(self):
        image_list = []
        for image_dir in os.listdir(self.dir):
            image_name = f'{image_dir}.png'
            image_list.append(image_name)
        return image_list

    def __getimage__(self, image_name):
        image_dir = image_name.split('.')[0]
        image_path = os.path.join(
            self.dir, image_dir, 'images', image_name)
        image = Image.open(image_path).convert('RGB')
        image = self.PILToTensor(image)
        image = image.to(torch.float32)
        image /= 255
        return image

    def __emptytargets__(self):
        targets = {}
        targets["boxes"] = []
        targets["labels"] = []
        targets["masks"] = []
        targets["image_id"] = []
        targets["area"] = []
        targets["iscrowd"] = []
        return targets

    def __targetsToNumpy__(self, targets):
        # for key in targets.keys():
        #     # print(key, type(targets[key]))
        #     # print(targets[key])
        #     targets[key] = np.array(targets[key])

        targets['masks'] = np.array(targets['masks'], dtype=np.int64)
        targets['boxes'] = np.array(targets['boxes'], dtype=np.float32)
        targets['labels'] = np.array(targets['labels'], dtype=np.int64)
        targets['area'] = np.array(targets['area'], dtype=np.float32)
        targets['iscrowd'] = np.array(targets['iscrowd'], dtype=np.int64)
        targets['image_id'] = np.array(targets['image_id'], dtype=np.int64)
        return targets

    def __tragtesToTensor__(self, targets):
        for key in targets.keys():
            # print(key, type(targets[key]))
            targets.update({key: torch.as_tensor(targets[key])})
        return targets

    def __gettargets__(self, image_name, index):
        image_dir = image_name.split('.')[0]
        mask_name_list = os.listdir(os.path.join(self.dir, image_dir, 'masks'))
        targets = self.__emptytargets__()
        for mask_name in mask_name_list:
            mask_path = os.path.join(self.dir, image_dir, 'masks', mask_name)

            mask = Image.open(mask_path).convert('L')
            mask = self.PILToTensor(mask)
            mask = mask//255
            mask = np.squeeze(mask)
            box = self.__getbox__(mask)
            if not self.__checkbox__(box):
                continue

            mask = mask.tolist()
            # print(f'mask size: {mask.size()}')
            targets['masks'].append(mask)
            # targets['boxes'].append(torch.as_tensor(box, dtype=torch.float32))
            targets['boxes'].append(box)

            label = 1  # only one class
            # targets['labels'].append(torch.as_tensor(label, dtype=torch.int64))
            targets['labels'].append(label)

            area = (box[2] - box[0]) * (box[3] - box[1])
            # targets['area'].append(torch.as_tensor(area, dtype=torch.float32))
            targets['area'].append(area)

            iscrowd = 0  # not sure
            # targets['iscrowd'].append(
            # torch.as_tensor(iscrowd, dtype=torch.int64))
            targets['iscrowd'].append(iscrowd)

        # targets['image_id'].append(torch.as_tensor(index, dtype=torch.int64))
        targets['image_id'].append(index)

        targets = self.__targetsToNumpy__(targets)

        # return targets
        return self.__tragtesToTensor__(targets)

    def __getbox__(self, mask):
        # (x_list, y_list) = np.where(mask == 1)
        (y_list, x_list) = np.where(mask == 1)
        if len(x_list) == 0 or len(y_list) == 0:
            return [-1, -1, -1, -1]
        xmin = min(x_list)
        xmax = max(x_list)
        ymin = min(y_list)
        ymax = max(y_list)
        box = [xmin, ymin, xmax, ymax]
        return box

    def __checkbox__(self, box):
        xmin, ymin, xmax, ymax = box
        return True if (xmin < xmax) and (ymin < ymax) else False


def test_MaskRCNNTrainDataset():
    test_dir = os.path.join('dataset', 'train')
    dataset = MaskRCNNTrainDataset(test_dir)
    dataloader = DataLoader(dataset, batch_size=2,
                            shuffle=False, collate_fn=collate_fn)
    for index, (images, targets, image_name) in enumerate(dataloader):
        print(f'index: {index}')
        for image, target in zip(images, targets):
            print(f'image shape: {image.size()}, type: {type(image)}')
            print(f'targets')
            print('='*20)
            for k in target:
                print(f'key: {k}')
                for i, item in enumerate(target[k]):
                    if k == "masks":
                        print(f'mask shape: {item.size()}')
                    else:
                        print(item)

                    if i > 4:
                        break


class TestDataset(AbstractDataset):
    def __init__(self, dir) -> None:
        super().__init__(dir)

    def __len__(self):
        return len(glob.glob(os.path.join(self.dir, '*.png')))


class MaskRCNNEvalDataset(AbstractDataset):
    def __init__(self, dir) -> None:
        super().__init__(dir)
        self.image_list = sorted(self.__loadimagelist__())
        self.PILToTensor = torchvision.transforms.PILToTensor()

    def __len__(self):
        return len(self.image_list)

    def __loadimagelist__(self):
        return [os.path.basename(path) for path in glob.glob(os.path.join(self.dir, '*.png'))]

    def __getimage__(self, image_name):
        image_path = os.path.join(self.dir, image_name)
        image = Image.open(image_path).convert('RGB')
        image = self.PILToTensor(image)
        image = image.to(torch.float32)
        image /= 255
        return image

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image = self.__getimage__(image_name)
        return image.type(torch.cuda.FloatTensor), image_name


if __name__ == '__main__':
    test_MaskRCNNTrainDataset()
