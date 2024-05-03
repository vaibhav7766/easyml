from sqlmodel import Session, select, update
from models import Project


# Project CRUD operations
class ProjectCRUD:

    @staticmethod
    def create_project(session: Session, project: Project) -> Project:
        project = Project.model_validate(project)
        session.add(project)
        session.commit()
        session.refresh(project)
        return project

    @staticmethod
    def get_projects(session: Session) -> list[Project]:
        statement = select(Project)
        projects = session.exec(statement).all()
        return projects

    @staticmethod
    def get_project(session: Session, id: int) -> Project | None:
        statement = select(Project).where(Project.id == id)
        project = session.exec(statement).first()
        return project

    @staticmethod
    def update_project(session: Session, project: Project) -> Project | None:
        statement = (
            update(Project)
            .where(Project.id == project.id)
            .values(**project.model_dump())
        )
        session.exec(statement)
        session.commit()
        session.refresh(project)
        return project
