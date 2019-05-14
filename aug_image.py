from imgaug import augmenters as iaa

def aug_image(image, is_infer=False, augment = 1):
    if is_infer:
        flip_code = augment

        if flip_code == 1:
            seq = iaa.Sequential([iaa.Fliplr(1.0)])
        elif flip_code == 2:
            seq = iaa.Sequential([iaa.Flipud(1.0)])
        elif flip_code == 3:
            seq = iaa.Sequential([iaa.Flipud(1.0),
                                  iaa.Fliplr(1.0)])
        elif flip_code ==0:
            return image

    else:

        seq = iaa.Sequential([
            iaa.Affine(rotate= (-15, 15),
                       shear = (-15, 15),
                       mode='edge'),

            iaa.SomeOf((0, 2),
                       [
                           iaa.GaussianBlur((0, 1.5)),
                           iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01 * 255), per_channel=0.5),
                           iaa.AddToHueAndSaturation((-5, 5)),
                           iaa.EdgeDetect(alpha=(0, 0.5)), 
                           iaa.CoarseSaltAndPepper(0.2, size_percent=(0.05, 0.1)), 
                       ],
                       random_order=True
                       )
        ])

    image = seq.augment_image(image)
    return image
