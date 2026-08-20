import asyncio

from viam.components.camera import Camera
from viam.module.module import Module

import src  # noqa: F401 — registers the model
from src.camera import MonoDepth


async def main() -> None:
    module = Module.from_args()
    module.add_model_from_registry(Camera.API, MonoDepth.MODEL)
    await module.start()


if __name__ == "__main__":
    asyncio.run(main())
