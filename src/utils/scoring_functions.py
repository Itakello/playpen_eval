from weave import op


@op()
def exact_match(answer: str, model_output: dict) -> dict:
    """Scoring function to check exact match."""
    return {"match": answer == model_output}


# Add more scoring functions as needed
