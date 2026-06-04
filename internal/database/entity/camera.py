from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Boolean
from internal.database.entity.base import Base

class Camera(Base):
    __tablename__ = 'm_cameras'

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    url: Mapped[str] = mapped_column(String, nullable=False)
    fight_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    vehicle_plate_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
