from genericpath import exists
from PIL import Image
import os
import numpy as np
import glob


def createdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class AbstractDivide:
    def __init__(self, n, source_dir, target_dir) -> None:
        self.n = self.__check__(n)
        self.source_dir = source_dir
        self.target_dir = target_dir

    def __loadimage__(self, image_path):
        raise NotImplementedError

    def __loadmask__(self, mask_path):
        raise NotImplementedError

    def __divideimage__(self, image):
        raise NotImplementedError

    def __dividemask__(self, mask):
        raise NotImplementedError
        # need to check the label position
        # return crop

    def __check__(self, n):
        if not isinstance(n, int):
            raise "n must be integer!"
        else:
            return n


class Divide(AbstractDivide):
    def __init__(self, n, source_dir, target_dir) -> None:
        super().__init__(n, source_dir, target_dir)

    def __getpartitions__(self, width, height):
        """
            if divide a image with n = 2,
            -------
            | 0| 1|
            |--|--|
            | 2| 3|
            -------
        """
        partitions = []
        p_width, p_height = (width//self.n), (height//self.n)
        for x in range(self.n):
            for y in range(self.n):
                partition_num = x * self.n + y
                left = y * p_width
                top = x * p_height
                right = (y+1) * p_width
                bottom = (x+1) * p_height
                crop_area = (left, top, right, bottom)
                partitions.append((partition_num, crop_area))
        return partitions

    def __loadimage__(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return image

    def __divideimage__(self, image, partition):
        # width, height = image.size
        # partitions = self.__getpartition__(width, height)
        # for (partition_num, crop_area) in partitions:
        # crop_image = image.crop(crop_area)
        # crop_image.show(title=f'partition number: {partition_num}')
        partition_num, crop_area = partition
        crop_image = image.crop(crop_area)
        return crop_image

    def __loadmask__(self, mask_path):
        mask = Image.open(mask_path).convert('L')
        # mask = mask//255
        # assert np.unique(mask) == np.array([0, 1])
        return mask

    def __getbox__(self, mask):
        mask = np.array(mask)
        # (x_list, y_list) = np.where(mask == 1)
        (x_list, y_list) = np.where(mask == 255)
        xmin = min(x_list)
        xmax = max(x_list)
        ymin = min(y_list)
        ymax = max(y_list)
        box = [xmin, ymin, xmax, ymax]
        return box

    def __checkinregion__(self, point, region):
        x, y = point
        ymin, xmin, ymax, xmax = region
        return True if (xmin <= x and x < xmax) and (ymin <= y and y < ymax) else False

    def __dividemask__(self, mask, partition):
        box = self.__getbox__(mask)
        xmin, ymin, xmax, ymax = box
        point1, point2 = (xmin, ymin), (xmax, ymax)
        # print(point1, point2)
        # for (partition_num, crop_area) in partitions:
        #     inregion = self.__checkinregion__(
        #         point1, crop_area) or self.__checkinregion__(point2, crop_area)
        #     if inregion:
        #         crop_mask = mask.crop(crop_area)
        #         # crop_mask.show(title=f'partition number: {partition_num}')
        #         # print(f'partition number: {partition_num}')
        partition_num, crop_area = partition
        inregion = self.__checkinregion__(
            point1, crop_area) or self.__checkinregion__(point2, crop_area)
        if inregion:
            crop_mask = mask.crop(crop_area)
            return crop_mask
        else:
            return None

    def __savepicture__(self, picture, path):
        picture.save(path)

    # not finish
    def run(self):
        for image_dir in os.listdir(self.source_dir):
            image_name = image_dir
            image_path = os.path.join(
                self.source_dir, image_dir, 'images', f'{image_name}.png')
            image = self.__loadimage__(image_path)
            width, height = image.size
            partitions = self.__getpartitions__(width, height)
            for partition in partitions:
                partition_num, area = partition
                crop_image = self.__divideimage__(image, partition)
                crop_image_dir_name = f'{image_name}-partition-{partition_num}'
                crop_image_dir = os.path.join(
                    self.target_dir, crop_image_dir_name, 'images')
                createdir(crop_image_dir)
                crop_image_path = os.path.join(
                    crop_image_dir, f'{crop_image_dir_name}.png')
                self.__savepicture__(crop_image, crop_image_path)

                masks_dir = os.path.join(self.source_dir, image_dir, 'masks')
                for mask_name in os.listdir(masks_dir):
                    mask_path = os.path.join(masks_dir, mask_name)
                    mask = self.__loadmask__(mask_path)
                    crop_mask = self.__dividemask__(mask, partition)
                    crop_mask_dir = os.path.join(
                        self.target_dir, crop_image_dir_name, 'masks')
                    createdir(crop_mask_dir)
                    crop_mask_path = os.path.join(
                        crop_mask_dir, f'{mask_name}.png')
                    if not (crop_mask is None):
                        self.__savepicture__(crop_mask, crop_mask_path)


class TestDivide(Divide):
    def __init__(self, n, source_dir, target_dir) -> None:
        super().__init__(n, source_dir, target_dir)

    def getpartition(self, width, height):
        return self.__getpartitions__(width, height)

    def testdivdeimage(self, image_path):
        image = self.__loadimage__(image_path)
        self.__divideimage__(image)

    def testdivdemask(self, mask_path, width, height):
        mask = self.__loadmask__(mask_path)
        partitions = self.__getpartitions__(width, height)
        self.__dividemask__(mask, partitions)


def test_Divide():
    n = 2
    source_dir = None
    target_dir = None
    divide = TestDivide(n, source_dir, target_dir)
    width = 1000
    height = 1000
    print(divide.getpartition(width, height))
    image_path = os.path.join(
        'dataset', 'test_divide', 'TCGA-18-5592-01Z-00-DX1', 'images', 'TCGA-18-5592-01Z-00-DX1.png')
    # divide.testdivdeimage(image_path)
    # for i in range(1, 240):
    for i in [38, 65, 94]:
        if i < 10:
            mask_name = f'mask_000{i}.png'
        elif i < 100:
            mask_name = f'mask_00{i}.png'
        elif i < 1000:
            mask_name = f'mask_0{i}.png'
        else:
            raise "Out of range!"
        mask_path = os.path.join(
            'dataset', 'test_divide', 'TCGA-18-5592-01Z-00-DX1', 'masks', mask_name)
        print(mask_name)
        divide.testdivdemask(mask_path, width, height)
