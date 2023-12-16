from .clip_patches import ClipPatches
from .download_patches import DownloadPatches
from .fooocus_patches import FooocusPatches

all_patches = {}

def patch_all():
    global all_patches

    all_patches["Clip"] = ClipPatches()
    all_patches["Download"] = DownloadPatches()
    all_patches["Fooocus"] = FooocusPatches()

    return all_patches

def undo_all():
    for key, patches in all_patches.items():
        patches.undo()