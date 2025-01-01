from pydantic import BaseModel


class DiseaseRequestData(BaseModel):
    disease: str