from weave import op


@op()
def exact_match(expected: str, model_output: dict) -> dict:
    """Scoring function to check exact match."""
    return {"match": expected == model_output.get("answer", "")}


# Add more scoring functions as needed
