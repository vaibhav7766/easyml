from sqlmodel import SQLModel, Field


# Shared properties
class HistoryBase(SQLModel):
    id: int
    project_name: str
    dataset_url: str


# Database model, database table inferred from class name
class History(HistoryBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    project_name: str
    dataset_url: str


# Properties to receive on item creation
# class HistoryCreate(HistoryBase):
#     id: int
#     project_name: str
#     dataset_url: str


# Properties to receive on item update
# class HistoryUpdate(HistoryBase):
#     title: str | None = None  # type: ignore


# Properties to return via API, id is always required
# class HistoryPublic(HistoryBase):
#     id: int
#     project_name: str
#     dataset_url: str


# class HistoriesPublic(SQLModel):
#     data: list[HistoryBase]
#     count: int
