from sqlmodel import SQLModel, Session, create_engine
from fastapi import Depends
from typing import Annotated


# Create a database engine
DATABASE_URL = "postgresql://postgres:admin@localhost/easyml"
engine = create_engine(DATABASE_URL)
SQLModel.metadata.create_all(engine)


# Create a session
def get_session():
    with Session(engine) as session:
        yield session


# Create a session dependency
SessionDep = Annotated[Session, Depends(get_session)]
