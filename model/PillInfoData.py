from pydantic import BaseModel


class PillInfoDataRequest(BaseModel):
    pillName: str