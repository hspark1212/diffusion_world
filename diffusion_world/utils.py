import os
from PIL import Image

import torchvision


def save2img(imgs, path, nrow=10):

    torchvision.utils.save_image(
        imgs.float(),
        path,
        normalize=True,
        nrow=nrow,
        padding=0,
        pad_value=0,
    )


def make_gif(plot_paths, git_name):
    frames = [Image.open(fn) for fn in plot_paths]
    frames[0].save(
        os.path.join(f"{git_name}"),
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=300,
        loop=0,
    )
