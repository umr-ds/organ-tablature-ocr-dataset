# -*- coding: utf-8 -*-
"""
Classes for Transformation augmentations (extensions of classes from the augmentor package)

@author: Daniel Schneider
"""

import random
import math
from Augmentor.Operations import Operation
from PIL import Image


class ShearFloatPrecision(Operation):
    """
    This class is used to shear images, that is to tilt them in a certain
    direction. Tilting can occur along either the x- or y-axis and in both
    directions (i.e. left or right along the x-axis, up or down along the
    y-axis).
    Images are sheared **in place** and an image of the same size as the input
    image is returned by this class. That is to say, that after a shear
    has been performed, the largest possible area of the same aspect ratio
    of the original image is cropped from the sheared image, and this is
    then resized to match the original image size. The
    :ref:`shearing` section describes this in detail.
    For sample code with image examples see :ref:`shearing`.
    """
    def __init__(self, probability, max_shear_left, max_shear_right, keep_size=False):
        """
        The shearing is randomised in magnitude, from 0 to the
        :attr:`max_shear_left` or 0 to :attr:`max_shear_right` where the
        direction is randomised. The shear axis is also randomised
        i.e. if it shears up/down along the y-axis or
        left/right along the x-axis.
        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param max_shear_left: The maximum shear to the left.
        :param max_shear_right: The maximum shear to the right.
        :type probability: Float
        :type max_shear_left: Integer
        :type max_shear_right: Integer
        """
        Operation.__init__(self, probability)
        self.max_shear_left = max_shear_left
        self.max_shear_right = max_shear_right
        self.keep_size = keep_size

    def perform_operation(self, images):
        """
        Shears the passed image according to the parameters defined during
        instantiation, and returns the sheared image.
        :param images: The image to shear.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        
        width, height = images[0].size

        # For testing.
        # max_shear_left = 20
        # max_shear_right = 20

        directions = ["x", "y"]
        direction = random.choice(directions)

        def do(image):
            
            if direction == "x":
                
                angle_to_shear = random.uniform(-self.max_shear_left, self.max_shear_left)

                # We use the angle phi in radians later
                phi = math.tan(math.radians(angle_to_shear))
            
                # Here we need the unknown b, where a is
                # the height of the image and phi is the
                # angle we want to shear (our knowns):
                # b = tan(phi) * a
                shift_in_pixels = int(round(phi * height))

                # For negative tilts, we reverse phi and set offset to 0
                # Also matrix offset differs from pixel shift for neg
                # but not for pos so we will copy this value in case
                # we need to change it
                matrix_offset = shift_in_pixels
                if angle_to_shear <= 0:
                    shift_in_pixels = abs(shift_in_pixels)
                    matrix_offset = 0
                    phi = abs(phi) * -1

                # Note: PIL expects the inverse scale, so 1/scale_factor for example.
                transform_matrix = (1, phi, -matrix_offset,
                                    0, 1, 0)
                
                image = image.transform((int(round(width + shift_in_pixels)), height),
                                        Image.AFFINE,
                                        transform_matrix,
                                        Image.BICUBIC)
                
                if self.keep_size:
                    image = image.crop((int(shift_in_pixels/2), 0, width+int(shift_in_pixels/2), height))
                
                return image

            elif direction == "y":
                angle_to_shear = random.uniform(-self.max_shear_right, self.max_shear_right)
                
                phi = math.tan(math.radians(angle_to_shear))
                
                shift_in_pixels = int(round(phi * width))
                
                matrix_offset = shift_in_pixels
                if angle_to_shear <= 0:
                    shift_in_pixels = abs(shift_in_pixels)
                    matrix_offset = 0
                    phi = abs(phi) * -1

                transform_matrix = (1, 0, 0,
                                    phi, 1, -matrix_offset)

                image = image.transform((width, int(round(height + shift_in_pixels))),
                                        Image.AFFINE,
                                        transform_matrix,
                                        Image.BICUBIC)
                
                if self.keep_size:
                    image = image.crop((0, int(shift_in_pixels/2), width, height+int(shift_in_pixels/2)))

                return image

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class Translation(Operation):
    """
    This class is used to randomly translate images in x or y direction. 
    Images are translated **in place** and an image of the same size as the input
    image is returned by this class.
    """
    def __init__(self, probability, max_translate_x, max_translate_y):
        """
        The translation is randomised in magnitude, from 0 to the
        :attr:`max_translate_x` or 0 to :attr:`max_translate_y` where the
        direction is randomised. The translation axis is also randomised
        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param max_translate_x: The maximum translation in x direction.
        :param max_translate_y: The maximum translation in y direction.
        :type probability: Float
        :type max_translate_x: Integer
        :type max_translate_y: Integer
        """
        Operation.__init__(self, probability)
        self.max_translate_x = max_translate_x
        self.max_translate_y = max_translate_y

    def perform_operation(self, images):
        """
        Translates the passed image according to the parameters defined during
        instantiation, and returns the translated image.
        :param images: The image to translate.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

        def do(image):
            
            translate_x = random.uniform(-self.max_translate_x, self.max_translate_x)
            translate_y = random.uniform(-self.max_translate_y, self.max_translate_y)

            transform_matrix = (1, 0, -translate_x,
                                0, 1, -translate_y)
                
            image = image.transform(image.size,
                                    Image.AFFINE,
                                    transform_matrix,
                                    Image.BICUBIC)

            return image

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images
    
    
class Scale(Operation):
    """
    This class is used to randomly scale images in x or y direction.
    """
    def __init__(self, probability, min_scale_x, max_scale_x, min_scale_y, max_scale_y, keep_size=False):
        """
        The scale is randomised in magnitude, in between :attr:`min_scale_x`
        and :attr:`max_scale_x` or :attr:`min_scale_y` to :attr:`max_scale_y`.
        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param min_scale_x: The minimum scale in x direction.
        :param max_scale_x: The maximum scale in x direction.
        :param min_scale_y: The minimum scale in y direction.
        :param max_scale_y: The maximum scale in y direction.
        :param keepSize: Indicates whether to crop the rescaled image to original size.
        :type probability: Float
        :type min_scale_x: Float
        :type max_scale_x: Float
        :type min_scale_y: Float
        :type max_scale_y: Float
        :type max_scale_y: Bool
        """
        Operation.__init__(self, probability)
        self.min_scale_x = min_scale_x
        self.max_scale_x = max_scale_x
        self.min_scale_y = min_scale_y
        self.max_scale_y = max_scale_y
        self.keep_size = keep_size

    def perform_operation(self, images):
        """
        Scales the passed image according to the parameters defined during
        instantiation, and returns the scaled image.
        :param images: The image to scale.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

        def do(image):
            
            scale_x = round(random.uniform(self.min_scale_x, self.max_scale_x), 2)
            scale_y = round(random.uniform(self.min_scale_y, self.max_scale_y), 2)

            image_scaled = image.resize((int(round(image.size[0] * scale_x)),
                                         int(round(image.size[1] * scale_y))),
                                         resample=Image.BICUBIC)
            
            if self.keep_size:
                w, h = image.size
                w_scaled, h_scaled = image_scaled.size
                
                return image_scaled.crop(
                                        (math.floor((float(w_scaled) / 2) - (float(w) / 2)),
                                         math.floor((float(h_scaled) / 2) - (float(h) / 2)),
                                         math.floor((float(w_scaled) / 2) + (float(w) / 2)),
                                         math.floor((float(h_scaled) / 2) + (float(h) / 2)))
                                        )
            else:
                return image_scaled

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images
