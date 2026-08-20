"""viam-labs:yolo-mono-depth — monocular YOLO depth as a point-cloud camera."""

from viam.components.camera import Camera
from viam.resource.registry import Registry, ResourceCreatorRegistration

from .camera import MonoDepth

Registry.register_resource_creator(
    Camera.API,
    MonoDepth.MODEL,
    ResourceCreatorRegistration(MonoDepth.new, MonoDepth.validate_config),
)
