#%%
import image_creation as ic

for i in range(0, 3):
    # Generate a screw image with OBB label file
    ic.create_screwImage(mask_dir='Masks', ouput_dir='generated_ImageLabel', image_name = 'generated_image', number_generated_objects=40, ImageSave=True)
    # Display the OBB bounding boxes with labels
    ic.display_obb_with_labels(image_path='generated_ImageLabel/generated_image.png', label_path='generated_ImageLabel/generated_image.txt')

# %% implentation idea for ultralytics
# import os
# import torch
# from torch.utils.data import DataLoader, Dataset
# dataset = CustomDataset(transforms=transforms)
# #dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
# class CustomDataset(Dataset):
#     def __init__(self, mask_dir='Masks', transforms=None):
#         self.transforms = transforms
#         self.mask_dir = mask_dir

#     def __getitem__(self, idx):
#         # Generate image and label file
#         image_name = f'generated_image_{idx}' # Unique name for each image/label pair
#         image = ic.create_screwImage(mask_dir=self.mask_dir, ouput_dir='generated_ImageLabel', 
#                                      image_name=image_name, number_generated_objects=40, imageSave=False)
#         label_path = os.path.join('generated_ImageLabel', image_name + '.txt')

#         # Read the label file
#         labels = []
#         with open(label_path, 'r') as f:
#             for line in f:
#                 parts = line.strip().split()
#                 class_name = parts[0]
#                 points_n = [float(x) for x in parts[1:]]
#                 labels.append({'class_name': class_name, 'points_normalized': points_n})

#         if self.transforms:
#             # You might need to adapt your transforms to handle both image and labels
#             augmented = self.transforms(image=image)
#             image = augmented['image']
#             # If you have transforms that modify bounding boxes, they should be applied here too
#             # For now, assuming transforms only affect the image
            
#         return image, labels

# Example usage:
# Assuming you have your transforms defined (e.g., from Albumentations)
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
#transforms =# A.Compose([
#     A.Resize(width=640, height=640),
#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ToTensorV2()
# ], bbox_params=A.BboxParams(format='
# %%
