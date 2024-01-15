from typing import List, Sequence, Tuple, Union, Sequence
from torch import Tensor
from torchvision import transforms as T
from byol_a2.augmentations import (
    RandomResizeCrop,
    MixupBYOLA,
    RandomLinearFader,
    RunningNorm,
)
from lightly.transforms.utils import IMAGENET_NORMALIZE

Image = Tensor


class ViewTransform:
    def __init__(
        self,
        crop_size: int = 224,
        crop_scale: Tuple[float, float] = (0.3, 1.0),
        epoch_samples=10,
        normalize=IMAGENET_NORMALIZE,
    ):
        transform = [
            MixupBYOLA(ratio=0.2, log_mixup_exp=True),
            RandomLinearFader(),
            T.RandomResizedCrop(size=crop_size, scale=crop_scale, antialias=False),
            RunningNorm(epoch_samples=epoch_samples),
            # RandomResizeCrop(
            #     virtual_crop_scale=(1.0, 1.5),
            #     freq_scale=crop_scale,
            #     time_scale=crop_scale,
            #     # freq_scale=(0.6, 1.5),
            #     # time_scale=(0.6, 1.5),
            # ),
            # T.RandomResizedCrop(size=crop_size, scale=crop_scale, antialias=True),
            # T.RandomHorizontalFlip(p=hf_prob),
            # T.RandomVerticalFlip(p=vf_prob),
            # T.RandomApply([color_jitter], p=cj_prob),
            # T.RandomGrayscale(p=random_gray_scale),
            # GaussianBlur(kernel_size=kernel_size, sigmas=sigmas, prob=gaussian_blur),
            # T.ToTensor(),
            # T.Normalize(mean=normalize["mean"], std=normalize["std"]),
        ]

        self.transform = T.Compose(transform)

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        """
        Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        transformed: Tensor = self.transform(image)
        return transformed


class MultiViewTransform:
    """Transforms an image into multiple views."""

    """Implements the transformations for MSN [0].

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of Tensor of length 2 * random_views + focal_views. (12 by default)

    Applies the following augmentations by default:
        - Random resized crop
        - Random horizontal flip
        - Color jitter
        - Random gray scale
        - Gaussian blur
        - ImageNet normalization

    Generates a set of random and focal views for each input image. The generated output
    is (views, target, filenames) where views is list with the following entries:
    [random_views_0, random_views_1, ..., focal_views_0, focal_views_1, ...].

    """

    def __init__(
        self,
        random_size: int = 224,
        focal_size: int = 96,
        random_views: int = 2,
        focal_views: int = 10,
        random_crop_scale: Tuple[float, float] = (0.3, 1.0),
        focal_crop_scale: Tuple[float, float] = (0.05, 0.3),
        epoch_samples=10,
    ):
        random_view_transform = ViewTransform(
            crop_size=random_size,
            crop_scale=random_crop_scale,
            epoch_samples=epoch_samples,
        )
        focal_view_transform = ViewTransform(
            crop_size=focal_size,
            crop_scale=focal_crop_scale,
            epoch_samples=epoch_samples,
        )
        transforms = [random_view_transform] * random_views
        transforms += [focal_view_transform] * focal_views
        self.transforms = transforms

    def __call__(self, image: Union[Tensor, Image]) -> Union[List[Tensor], List[Image]]:
        return [transform(image) for transform in self.transforms]
