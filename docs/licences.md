# Licences

just-matching does not distribute FlexGlue's weights, and this site does not serve them. You download them from
FlexGlue's authors, <a href="https://github.com/cvg/FlexGlue/releases/tag/v0.1">FlexGlue's v0.1 release</a>
(`flexglue.pth`, 567 MB), and give the file to the library or to the demo. Your copy comes with two licences:

| Tensors | What they are | Licence |
|---|---|---|
| `extractor.backbone.*` | the DINOv3 ConvNeXt-B backbone, by Meta | the DINOv3 License |
| everything else | FlexGlue's matcher and its descriptor heads | the MIT licence |

## DINOv3 License

Using the backbone's weights means accepting Meta's
[DINOv3 License](https://github.com/cvg/FlexGlue/blob/main/DINOv3_LICENSE.md). Among other things, it forbids uses
subject to the International Traffic in Arms Regulations and military, nuclear or espionage uses, and requires a
copy of it with any copy of the weights you pass on. Read it before you download the file.

The demo reads the file in your browser and, if the browser allows, keeps a copy in the site's private storage on
your computer so that you need not load it again; **Forget** deletes that copy. Nothing is uploaded.

## FlexGlue: MIT License

FlexGlue's code and the weights of its matcher and descriptor heads are under the
[MIT licence](https://github.com/cvg/FlexGlue/blob/main/LICENSE), copyright the Computer Vision and Geometry Lab.

## Images in the demo

The sample images have their own licences: see [References](references.md).
