"""Minimal MovementSensor → body twist reader (wheel odom for scale)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from viam.components.movement_sensor import MovementSensor


@dataclass(frozen=True)
class OdomReading:
    vx: float
    vy: float
    vtheta: float


@dataclass(frozen=True)
class TypedOdomConfig:
    use_linear_velocity: bool = True


class TypedMovementSensorOdom:
    """Read angular + linear velocity via the typed MovementSensor API."""

    def __init__(
        self,
        sensor: MovementSensor,
        cfg: Optional[TypedOdomConfig] = None,
        logger=None,
    ):
        self._sensor = sensor
        self._cfg = cfg or TypedOdomConfig()
        self._logger = logger
        self._props: Optional[MovementSensor.Properties] = None

    async def properties(self) -> MovementSensor.Properties:
        if self._props is None:
            self._props = await self._sensor.get_properties()
            if self._logger is not None:
                self._logger.info(
                    "typed odom: angular_velocity=%s linear_velocity=%s",
                    self._props.angular_velocity_supported,
                    self._props.linear_velocity_supported,
                )
        return self._props

    async def read(self) -> OdomReading:
        p = await self.properties()
        vx = vy = vtheta = 0.0
        if p.angular_velocity_supported:
            av = await self._sensor.get_angular_velocity()
            vtheta = math.radians(float(av.z))
        if self._cfg.use_linear_velocity and p.linear_velocity_supported:
            lv = await self._sensor.get_linear_velocity()
            vx, vy = float(lv.x), float(lv.y)
        return OdomReading(vx, vy, vtheta)
