# swarmui_exif_to_civitai

[SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI) is great, but its exif format is not compatible with [Civitai](https://civitai.com).

When you create an image using SwarmUI and upload it to Civitai, none of the exif metadata is used. You have to manually enter in the positive/negative prompt, seed, CFG scale, steps, model, loras, embeds, sampler, scheduler, etc. It sucks.

I made this tool to quickly convert SwarmUI's exif metadata to what Civitai expect, so you can drag & drop your images directly into Civitai and have all (read: most) of your data automatically applied. Like magic.

* run against a directory of images: `python main.py ./path/to/images`
* or run against a single image: `python main.py ./path/to_image.png`

By default a copy of the image is saved, using the md5 hash of the new exif data in the filename:

`some_image.png` is copied to `some_image_16f2d71f356612859f0d83c029965128.png`

You can pass the `--overwrite` to overwrite the exif data directly against the original file: `python main.py ./path/to/images --overwrite`
