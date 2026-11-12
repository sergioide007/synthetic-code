"""A deterministic stand-in agent used by the CLI tests and the README quick start."""


def answer(question: str) -> str:
    q = question.lower()
    if "refund" in q:
        return "Refunds are available within 30 days."
    if "password" in q:
        return "We emailed you a reset link."
    return "I cannot share card details."


def leaky(question: str) -> str:
    return "your card is 4111 1111 1111 1111"
