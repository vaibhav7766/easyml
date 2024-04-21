from sqlmodel import Session, select
from models import History


# History CRUD operations
class HistoryCRUD:

    @staticmethod
    def create_history(session: Session, history: History) -> History:
        history = History.model_validate(history)
        session.add(history)
        session.commit()
        session.refresh(history)
        return history

    @staticmethod
    def get_histories(session: Session) -> list[History]:
        statement = select(History)
        histories = session.exec(statement).all()
        return histories

    @staticmethod
    def get_history(session: Session, id: int) -> History | None:
        statement = select(History).where(History.id == id)
        history = session.exec(statement).first()
        return history
