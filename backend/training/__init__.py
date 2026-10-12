# Lazy imports to prevent circular dependency with preprocessing module
def __getattr__(name):
    if name == 'ImageDataset':
        from .image_dataset import ImageDataset
        return ImageDataset
    if name == 'VideoDataset':
        from .video_dataset import VideoDataset
        return VideoDataset
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'ImageDataset',
    'VideoDataset'
]

