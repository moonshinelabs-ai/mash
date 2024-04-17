<br />
<p align="center">
    <a href="https://github.com/moonshinelabs-ai/moonshine">
      <img src="https://moonshine-assets.s3.us-west-2.amazonaws.com/mash_full_logo_light.png" width="50%"/>
    </a>
</p>

<h2><p align="center">Shared utility functions powering Moonshine tools.</p></h2>

<p align="center">
    <a href="https://moonshine-mash.readthedocs.io/en/latest/">
        <img alt="Documentation" src="https://readthedocs.org/projects/moonshine-mash/badge/?version=latest">
    </a>
    <a href="https://pypi.org/project/moonshinelabs-ai/">
        <img alt="PyPi Version" src="https://img.shields.io/pypi/pyversions/mashlib">
    </a>
    <a href="https://pypi.org/project/mashlib/">
        <img alt="PyPi Package Version" src="https://img.shields.io/pypi/v/mashlib">
    </a>
    <a href="https://pepy.tech/project/mashlib/">
        <img alt="PyPi Downloads" src="https://static.pepy.tech/personalized-badge/mashlib?period=month&units=international_system&left_color=grey&right_color=blue&left_text=Downloads/month">
    </a>
    <a href="https://join.slack.com/t/moonshinecommunity/shared_invite/zt-1rg1vnvmt-pleUR7TducaDiAhcmnqAQQ">
        <img alt="Chat on Slack" src="https://img.shields.io/badge/slack-chat-2eb67d.svg?logo=slack">
    </a>
    <a href="https://github.com/moonshinelabs-ai/moonshine/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
    </a>
</p>
<br />

## What is Mash?
Mash is a straightforward utility library for common tasks in computer vision and deep model training. The library was broken out of previous Moonshine projects like Moonshine and Zeroshot.

## What can Mash Do?
Mash broadly supports a few utilities, but the main ones are:

1. Easy image conversion: simply call `to_pil`, `to_numpy`, and `to_tensor` to convert image formats. Accepts other images, URLs, or local files.
2. Image processing files: convenience functions like `crop_to_multiple_of_dimensions` for transformer based patch models like ViT.
3. Console UI: for long running jobs, a fullscreen console utility that has a progress bar at the bottom and text logging.
4. Cloud functions: use `glob` or `exists` on AWS or GCS links.

For a complete list of functions, see [the documentation](https://moonshine-mash.readthedocs.io/en/latest/index.html)

## Installation
To install via pip:

`pip install mashlib`

## Usage
To use:

```python
# Import base package
import mash

# Import image processing
import mash.images as mi
image = mi.to_numpy("/path/to/image.png")
```