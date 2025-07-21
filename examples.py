#%%
import image_creation as ic

for i in range(0, 10):
    # Generate a screw image with OBB label file
    ic.create_screwImage(mask_dir='Masks', ouput_dir='generated_ImageLabel', image_name = 'generated_image', number_generated_objects=40, ImageSave=True)
    # Display the OBB bounding boxes with labels
    ic.display_obb_with_labels(image_path='generated_ImageLabel/generated_image.png', label_path='generated_ImageLabel/generated_image.txt')



# %%
