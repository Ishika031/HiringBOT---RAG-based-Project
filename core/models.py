from pydantic import BaseModel, Field
from typing import Optional, List

class CandidateMetadata(BaseModel):
    name: str = Field(description="Full name of the candidate")
    email: str = Field(description="Professional email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    location: str = Field(description="City, State/Country")
    experience_years: float = Field(description="Total years of experience (numeric)")
    gender: Optional[str] = Field(default=None, description="Male / Female / Other or None")
    primary_field: str = Field(description="Primary professional field e.g. Human Resources, Sales, AI/ML Engineer, Frontend Developer, Cloud Architect, UI/UX Designer etc.")
    skills: List[str] = Field(description="List of key skills")
    education: str = Field(description="Highest education + institution")
    summary: str = Field(description="2-3 sentence professional summary")