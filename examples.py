#%%
import image_creation as ic

for i in range(0, 3):
    # Generate a screw image with OBB label file
    ic.create_screwImage(mask_dir='Masks', ouput_dir='generated_ImageLabel', image_name = 'generated_image', number_generated_objects=40, number_generated_otherScrews=40, number_generated_shapes=30, simpleBackground=False, ImageSave=True)
    # Display the OBB bounding boxes with labels
    ic.display_obb_with_labels(image_path='generated_ImageLabel/generated_image.png', label_path='generated_ImageLabel/generated_image.txt')

#%% 5000 images:  simpleBackground = True, number_generated_objects=[1,40], number_generated_otherScrews=[1,60], number_generated_shapes=[1,30]
import random
for i in range(5000):
    try:
        number_generated_objects = random.randint(5, 40)
        number_generated_otherScrews = random.randint(5, 60)
        number_generated_shapes = random.randint(3, 30)

        ic.create_screwImage(
            mask_dir='Masks',
            ouput_dir='generated_ImageLabel',
            image_name=f'generated_image_{i+1}',
            number_generated_objects=number_generated_objects,
            number_generated_otherScrews=number_generated_otherScrews,
            number_generated_shapes=number_generated_shapes,
            simpleBackground=True,
            ImageSave=True,
            Png=False
        )
    except Exception as e:
        print(f"Failed to generate image {i+1}: {e}")

for i in range(5000, 10001):
    try:
        number_generated_objects = random.randint(5, 40)
        number_generated_otherScrews = random.randint(5, 60)
        number_generated_shapes = random.randint(3, 30)

        ic.create_screwImage(
            mask_dir='Masks',
            ouput_dir='generated_ImageLabel',
            image_name=f'generated_image_{i+1}',
            number_generated_objects=number_generated_objects,
            number_generated_otherScrews=number_generated_otherScrews,
            number_generated_shapes=number_generated_shapes,
            simpleBackground=False,
            ImageSave=True,
            Png=False
        )
    except Exception as e:
        print(f"Failed to generate image {i+1}: {e}")
# %%
