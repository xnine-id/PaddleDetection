from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from internal.database.entity.camera import Camera
from internal.api.schemas import AddCameraRequest, UpdateCameraRequest
from typing import List, Optional
from internal.core.camera_manager import CameraManager

class CameraService:
    def __init__(self, session: AsyncSession, camera_manager: CameraManager):
        self.session = session
        self.camera_manager = camera_manager

    async def get_cameras(self) -> List[Camera]:
        result = await self.session.execute(select(Camera))
        return list(result.scalars().all())

    async def get_camera(self, camera_name: str) -> Optional[Camera]:
        result = await self.session.execute(select(Camera).where(Camera.name == camera_name))
        return result.scalar_one_or_none()

    async def create_camera(self, data: AddCameraRequest) -> Camera:
        new_camera = Camera(
            name=data.name,
            url=data.url,
            fight_enabled=data.fight_enabled,
            vehicle_plate_enabled=data.vehicle_plate_enabled,
        )
        self.session.add(new_camera)
        await self.session.commit()
        await self.session.refresh(new_camera)

        # Add to camera manager only if at least one detection is enabled
        if new_camera.fight_enabled or new_camera.vehicle_plate_enabled:
            self.camera_manager.add_camera_processor(new_camera)

        return new_camera

    async def update_camera(self, camera_name: str, data: UpdateCameraRequest) -> Optional[Camera]:
        camera = await self.get_camera(camera_name)
        if not camera:
            return None

        old_name = camera.name
        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(camera, key, value)

        await self.session.commit()
        await self.session.refresh(camera)

        # Structural changes (name/url) require full restart
        is_enabled = camera.fight_enabled or camera.vehicle_plate_enabled
        needs_restart = "name" in update_data or "url" in update_data

        if needs_restart:
            self.camera_manager.remove_camera_processor(old_name)
            if is_enabled:
                self.camera_manager.add_camera_processor(camera)
        elif is_enabled:
            self.camera_manager.update_camera_processor(camera)
        else:
            # Not enabled — remove processor from thread
            self.camera_manager.remove_camera_processor(camera.name)

        return camera

    async def delete_camera(self, camera_name: str) -> bool:
        camera = await self.get_camera(camera_name)
        if not camera:
            return False

        cam_name = camera.name
        await self.session.delete(camera)
        await self.session.commit()

        # Remove from camera manager
        self.camera_manager.remove_camera_processor(cam_name)

        return True

    async def sync_cameras(self, cameras_data: List[AddCameraRequest]) -> List[Camera]:
        """
        Synchronizes the camera database with the provided list.
        - Adds new cameras
        - Updates existing cameras (matched by name)
        - Removes cameras not in the list
        """
        existing_cameras = await self.get_cameras()
        existing_map = {c.name: c for c in existing_cameras}

        incoming_names = {c.name for c in cameras_data}
        synced_cameras = []

        # 1. Update or Create
        for data in cameras_data:
            if data.name in existing_map:
                # Update existing
                camera = existing_map[data.name]
                needs_restart = camera.url != data.url  # url changed

                update_data = data.model_dump()
                for key, value in update_data.items():
                    setattr(camera, key, value)

                is_enabled = camera.fight_enabled or camera.vehicle_plate_enabled

                if needs_restart:
                    self.camera_manager.remove_camera_processor(camera.name)
                    if is_enabled:
                        self.camera_manager.add_camera_processor(camera)
                elif is_enabled:
                    self.camera_manager.update_camera_processor(camera)
                else:
                    # Not enabled — remove processor from thread
                    self.camera_manager.remove_camera_processor(camera.name)
                synced_cameras.append(camera)
            else:
                # Create new
                new_camera = Camera(
                    name=data.name,
                    url=data.url,
                    fight_enabled=data.fight_enabled,
                    vehicle_plate_enabled=data.vehicle_plate_enabled,
                )
                self.session.add(new_camera)
                await self.session.flush()  # Flush to ensure it's tracked and ready for CameraManager

                # Add to camera manager only if at least one detection is enabled
                if new_camera.fight_enabled or new_camera.vehicle_plate_enabled:
                    self.camera_manager.add_camera_processor(new_camera)
                synced_cameras.append(new_camera)

        # 2. Delete cameras not in incoming list
        for name, camera in existing_map.items():
            if name not in incoming_names:
                await self.session.delete(camera)
                self.camera_manager.remove_camera_processor(name)

        await self.session.commit()
        return synced_cameras
