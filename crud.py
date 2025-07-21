from bson import ObjectId

class ProjectCRUD:
    def __init__(self, session):
        self.session = session  # session is a MongoDB database

    def get_project(self, project_id):
        """Fetch a project by its ObjectId string."""
        project = self.session["projects"].find_one({"_id": ObjectId(project_id)})
        return project

    def add_project(self, project):
        """Insert a new project document and return its inserted_id."""
        result = self.session["projects"].insert_one(project)
        return str(result.inserted_id)
