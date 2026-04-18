"""Raw EHR loader.
Only responsible for reading event data corresponding to subject/hadm from the raw data path, without business logic transformation.
"""


def load_subject(subject_id: str):
    """Load all admission records for a subject."""
    raise NotImplementedError
