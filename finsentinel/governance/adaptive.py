def adaptive_governance_thresholds(
    drift_score,
    approve_threshold,
    block_threshold
):

    """
    Adaptive threshold control based
    on behavioural drift severity
    """

    if drift_score < 0.02:

        severity="Low"

        approve=approve_threshold

        block=block_threshold


    elif drift_score < 0.05:

        severity="Moderate"

        approve=max(
            approve_threshold-0.05,
            0.05
        )

        block=max(
            block_threshold-0.05,
            approve+0.10
        )


    else:

        severity="High"

        approve=max(
            approve_threshold-0.10,
            0.05
        )

        block=max(
            block_threshold-0.10,
            approve+0.10
        )


    return {

        "severity":severity,

        "approve":approve,

        "block":block
    }