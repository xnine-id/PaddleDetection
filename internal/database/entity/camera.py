from sqlalchemy import Column, Integer, String, Boolean, JSON
from internal.database.entity.base import Base

class Camera(Base):
    __tablename__ = 'm_cameras'

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    url = Column(String, nullable=False)
    fight_enabled = Column(Boolean, nullable=False, default=True)
    vehicle_plate_enabled = Column(Boolean, nullable=False, default=True)
    snapshot_enabled = Column(Boolean, nullable=False, default=True)
    mqtt_enabled = Column(Boolean, nullable=False, default=True)
