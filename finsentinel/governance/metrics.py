import numpy as np


def calculate_governance_metrics(
    y_true,
    routing_actions
):
    """
    Calculates governance routing statistics.
    """

    routing_array = np.array(routing_actions)

    legit_auto_approve = (
        (y_true == 0) &
        (routing_array == "Auto Approve")
    ).sum()

    legit_review = (
        (y_true == 0) &
        (routing_array == "Human Review")
    ).sum()

    legit_block = (
        (y_true == 0) &
        (routing_array == "Auto Block")
    ).sum()

    fraud_auto_approve = (
        (y_true == 1) &
        (routing_array == "Auto Approve")
    ).sum()

    fraud_review = (
        (y_true == 1) &
        (routing_array == "Human Review")
    ).sum()

    fraud_block = (
        (y_true == 1) &
        (routing_array == "Auto Block")
    ).sum()

    return {
        "legit_auto_approve": legit_auto_approve,
        "legit_review": legit_review,
        "legit_block": legit_block,
        "fraud_auto_approve": fraud_auto_approve,
        "fraud_review": fraud_review,
        "fraud_block": fraud_block,
    }