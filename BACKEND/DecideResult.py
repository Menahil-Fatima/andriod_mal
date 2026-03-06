def decide(best_family: str, best_score: float, threshold: float):
    """
    smaller distance => more similar
    """
    if best_score <= threshold:
        return {"is_malware": True, "family": best_family}
    return {"is_malware": False, "family": "Unknown"}
                                                            