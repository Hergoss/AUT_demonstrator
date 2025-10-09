#%% Example to generate one image with OBB labels and display them
import image_creation as ic

for i in range(0, 3):
    # Generate a screw image with OBB label file
    ic.create_screwImage(mask_dir='Masks', 
                         output_dir='generated_ImageLabel', image_name = 'generated_image', 
                         number_generated_objects=40, number_generated_otherScrews=40, number_generated_shapes=30, 
                         simpleBackground=False, ImageSave=True)
    # Display the OBB bounding boxes with labels
    ic.display_obb_with_labels(
        image_path='generated_ImageLabel/generated_image.png', 
        label_path='generated_ImageLabel/generated_image.txt'
    )

#%% Example to generate 1000 images (500 with simple background and 500 with hintergrundbildern)
# import random
# # for i in range(500):
# #     try:
# #         number_generated_objects = random.randint(30, 40)
# #         number_generated_otherScrews = random.randint(50, 60)
# #         number_generated_shapes = random.randint(20, 30)

# #         ic.create_screwImage(
# #             mask_dir='Masks',
# #             output_dir='generated_ImageLabel',
# #             image_name=f'generated_image_{i+1}',
# #             number_generated_objects=number_generated_objects,
# #             number_generated_otherScrews=number_generated_otherScrews,
# #             number_generated_shapes=number_generated_shapes,
# #             simpleBackground=True,
# #             ImageSave=True,
# #             Png=False
# #         )
# #     except Exception as e:
# #         print(f"Failed to generate image {i+1}: {e}")

# for i in range(500, 1001):
#     try:
#         number_generated_objects = random.randint(60, 70)
#         number_generated_otherScrews = random.randint(40, 60)
#         number_generated_shapes = random.randint(20, 30)

#         ic.create_screwImage(
#             mask_dir='Masks',
#             output_dir='generated_ImageLabel',
#             image_name=f'generated_image_{i+1}',
#             number_generated_objects=number_generated_objects,
#             number_generated_otherScrews=number_generated_otherScrews,
#             number_generated_shapes=number_generated_shapes,
#             simpleBackground=False,
#             ImageSave=True,
#             Png=False
#         )
#     except Exception as e:
#         print(f"Failed to generate image {i+1}: {e}")

#%% Example to convert generated images and lables to ultralitics folder structure
# # to correct label names to indices from the data.yml file
# ic.convert_labels_to_indices(label_folder="./generated_ImageLabel")
# ic.organize_generated_files()
# ic.find_labels_without_images(base_folder="./generated_ImageLabel")

