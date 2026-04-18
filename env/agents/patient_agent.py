"""Patient component.
Only responsible for answering patient-related questions, using input from the current stage's patient readview.
"""


def answer_patient_question(readview, question: str) -> str:
    """Return patient's answer."""
    raise NotImplementedError
