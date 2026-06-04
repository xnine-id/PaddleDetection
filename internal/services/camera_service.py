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

        # Add to camera manager
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

        # Update in camera manager
        if old_name != camera.name:
            self.camera_manager.remove_camera_processor(old_name)
            self.camera_manager.add_camera_processor(camera)
        else:
            self.camera_manager.update_camera_processor(camera)

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

    async def sync_cameras(self, data: List[AddCameraRequest]) -> List[Camera]:
        # 1. Get all current cameras
        existing_cameras = await self.get_cameras()
        existing_map = {c.name: c for c in existing_cameras}
        
        input_names = {item.name for item in data}
        result_cameras = []

        # 2. Add or Update
        for item in data:
            if item.name in existing_map:
                # Update
                camera = existing_map[item.name]
                camera.url = item.url
                camera.fight_enabled = item.fight_enabled
                camera.vehicle_plate_enabled = item.vehicle_plate_enabled
                
                # Update in camera manager
                self.camera_manager.update_camera_processor(camera)
                result_cameras.append(camera)
            else:
                # Create
                new_camera = Camera(
                    name=item.name,
                    url=item.url,
                    fight_enabled=item.fight_enabled,
                    vehicle_plate_enabled=item.vehicle_plate_enabled,
                )
                self.session.add(new_camera)
                # We'll commit and refresh later to get IDs
                result_cameras.append(new_camera)

        # 3. Delete ones not in input
        for name, camera in existing_map.items():
            if name not in input_names:
                await self.session.delete(camera)
                self.camera_manager.remove_camera_processor(name)

        await self.session.commit()

        # Refresh new cameras to get IDs and add them to camera manager
        for camera in result_cameras:
            await self.session.refresh(camera)
            if camera.name not in existing_map:
                self.camera_manager.add_camera_processor(camera)

        return result_cameras
