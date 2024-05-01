from sqlmodel import SQLModel, Field


# Shared properties
class ProjectBase(SQLModel):
    __tablename__ = "projects"
    id: int
    name: str
    dataset_url: str
    user_id: int

# Database model, database table inferred from class name
class Project(ProjectBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    dataset_url: str
    user_id: int
    edit_dataset_url: str
    final_dataset_url: str


# Properties to receive on item creation
# class ProjectCreate(ProjectBase):
#     id: int
#     name: str
#     dataset_url: str
#     user_id: int


# Properties to receive on item update
# class ProjectUpdate(ProjectBase):
#     title: str | None = None  # type: ignore


# Properties to return via API, id is always required
# class ProjectPublic(ProjectBase):
#     id: int
#     name: str
#     dataset_url: str
#     user_id: int


# class HistoriesPublic(SQLModel):
#     data: list[ProjectBase]
#     count: int
