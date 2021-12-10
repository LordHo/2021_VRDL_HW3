import timm
# from torchsummary import summary
import torchvision

# Swin Transformer
# model = timm.models.swin_transformer.swin_large_patch4_window12_384_in22k(
#     pretrained=True)

# print(model, file=open('swin_large_patch4_window12_384_in22k.txt', 'w'))


# Mask R-CNN
model = torchvision.models.detection.maskrcnn_resnet50_fpn()
print(model, file=open('Mask R-CNN.txt', 'w'))
