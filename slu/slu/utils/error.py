class MissingArtifact(Exception):
    def __init__(self, artifact, artifact_path) -> None:
        message = f"Missing artifact {artifact} at path {artifact_path}. Possibly your model isn't trained!"
        super().__init__(message)
