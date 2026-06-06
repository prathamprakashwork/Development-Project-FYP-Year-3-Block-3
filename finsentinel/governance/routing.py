def route_predictions(
    probs,
    approve_threshold,
    block_threshold
):
    """
    Routes fraud probabilities into:
    - Auto Approve
    - Human Review
    - Auto Block
    """

    routing_actions = []

    for p in probs:

        if p < approve_threshold:
            routing_actions.append("Auto Approve")

        elif p >= block_threshold:
            routing_actions.append("Auto Block")

        else:
            routing_actions.append("Human Review")

    return routing_actions