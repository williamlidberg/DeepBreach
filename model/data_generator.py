from tensorflow.keras.preprocessing.image import ImageDataGenerator

'''
Note that only the training generator contains data augmentation for the images and masks. 
The test generator will only be used for evaluation and I want to evaluate without augmentation. 
This code contains three types of generators, one for the 4th band (downslope), one for the first three bands (RGB) and one for both combined.
'''

 # Single channel generators
def create_train_generator_single_channel(image_dir, mask_dir, target_size=(256, 256), seed=42):
    # Data generation parameters
    data_gen_args = {
        'rotation_range': 45,
    #    'width_shift_range': 0.1,
    #    'height_shift_range': 0.1,
    #    'shear_range': 0.2,
    #    'zoom_range': 0.2,
        'horizontal_flip': True,
        'vertical_flip': True,
        'rescale': 1./255,
        'fill_mode': 'reflect'
    }
    
    # Create ImageDataGenerators
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    # Create generators for images and masks
    image_generator = image_datagen.flow_from_directory(
        image_dir,
        class_mode=None,
        color_mode='rgba',#'rgba',
        seed=seed,
        target_size=target_size
    )
    mask_generator = mask_datagen.flow_from_directory(
        mask_dir,
        class_mode=None,
        color_mode='grayscale',
        seed=seed,
        target_size=target_size
    )
    
    def adjust_images(images):
        # Select only the last channel
        return images[:, :, :, -1:]

    train_generator = ((adjust_images(image), mask) for image, mask in zip(image_generator, mask_generator))
    return train_generator

def create_test_generator_single_channel(image_dir, mask_dir, target_size=(256, 256), seed=42):
    # Data generation parameters
    data_gen_args = {
        'rescale': 1./255
    }
    # Create ImageDataGenerators
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    # Create generators for images and masks
    image_generator = image_datagen.flow_from_directory(
        image_dir,
        class_mode=None,
        color_mode='rgba',  # Change to 'rgb' to get only first three bands
        seed=seed,
        target_size=target_size
    )
    mask_generator = mask_datagen.flow_from_directory(
        mask_dir,
        class_mode=None,
        color_mode='grayscale',
        seed=seed,
        target_size=target_size
    )

    # Combine the generators into one generator with both images and masks
    def adjust_images(images):
        # Select only the last channel
        return images[:, :, :, -1:]

    test_generator = ((adjust_images(image), mask) for image, mask in zip(image_generator, mask_generator))
    return test_generator




'''
This is the RGB generator.
'''

def create_train_generator_RGB(image_dir, mask_dir, target_size=(256, 256), seed=42):
    # Data generation parameters
    data_gen_args = {
        'rotation_range': 45,
    #    'width_shift_range': 0.1,
    #    'height_shift_range': 0.1,
    #    'shear_range': 0.2,
    #    'zoom_range': 0.2,
        'horizontal_flip': True,
        'vertical_flip': True,
        'rescale': 1./255,
        'fill_mode': 'reflect'
    }
    
    # Create ImageDataGenerators
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    # Create generators for images and masks
    image_generator = image_datagen.flow_from_directory(
        image_dir,
        class_mode=None,
        color_mode='rgba',#'rgba',
        seed=seed,
        target_size=target_size
    )
    mask_generator = mask_datagen.flow_from_directory(
        mask_dir,
        class_mode=None,
        color_mode='grayscale',
        seed=seed,
        target_size=target_size
    )
    
    def adjust_images(images):
        # Select only the first three bands
        return images[:, :, :, :3]

    train_generator = ((adjust_images(image), mask) for image, mask in zip(image_generator, mask_generator))
    return train_generator


def create_test_generator_RGB(image_dir, mask_dir, target_size=(256, 256), seed=42):
    # Data generation parameters
    data_gen_args = {
        'rescale': 1./255
    }
    # Create ImageDataGenerators
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    # Create generators for images and masks
    image_generator = image_datagen.flow_from_directory(
        image_dir,
        class_mode=None,
        color_mode='rgb',  # Change to 'rgb' to get only first three bands
        seed=seed,
        target_size=target_size
    )
    mask_generator = mask_datagen.flow_from_directory(
        mask_dir,
        class_mode=None,
        color_mode='grayscale',
        seed=seed,
        target_size=target_size
    )

    # Combine the generators into one generator with both images and masks
    def adjust_images(images):
        # Select only the first three bands
        return images[:, :, :, :3]

    test_generator = ((adjust_images(image), mask) for image, mask in zip(image_generator, mask_generator))
    return test_generator

'''
This is the combined generator.
'''

# Single channel generators
def create_train_generator_combined(image_dir, mask_dir, target_size=(256, 256), seed=42):
    # Data generation parameters
    data_gen_args = {
        'rotation_range': 45,
    #    'width_shift_range': 0.1,
    #    'height_shift_range': 0.1,
    #    'shear_range': 0.2,
    #    'zoom_range': 0.2,
        'horizontal_flip': True,
        'vertical_flip': True,
        'rescale': 1./255,
        'fill_mode': 'reflect'
    }
    # Create ImageDataGenerators
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    image_generator = image_datagen.flow_from_directory(
        image_dir,
        class_mode=None,
        color_mode='rgba',#'rgba',
        seed=seed,
        target_size=target_size
    )
    mask_generator = mask_datagen.flow_from_directory(
        mask_dir,
        class_mode=None,
        color_mode='grayscale',
        seed=seed,
        target_size=target_size
    )
    train_generator = (pair for pair in zip(image_generator, mask_generator))  
    return train_generator

def create_test_generator_combined(image_dir, mask_dir, target_size=(256, 256), seed=42):
    # Data generation parameters
    data_gen_args = {
        'rescale': 1./255
    }
    # Create ImageDataGenerators
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    # Create generators for images and masks
    image_generator = image_datagen.flow_from_directory(
        image_dir,
        class_mode=None,
        color_mode='rgba',#'rgba',
        seed=seed,
        target_size=target_size
    )
    mask_generator = mask_datagen.flow_from_directory(
        mask_dir,
        class_mode=None,
        color_mode='grayscale',
        seed=seed,
        target_size=target_size
    )
    # Combine the generators into one generator with both images and masks
    test_generator = (pair for pair in zip(image_generator, mask_generator))
    return test_generator
