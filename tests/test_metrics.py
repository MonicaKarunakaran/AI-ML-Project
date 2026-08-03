from src.evaluation import calculate_metrics



def test_metrics_range():

    y_true = [0,1,1,0,1]

    y_pred = [0,1,1,0,0]

    y_prob = [0.1,0.8,0.9,0.2,0.4]


    result = calculate_metrics(
        y_true,
        y_pred,
        y_prob
    )


    for value in result.values():

        assert 0 <= value <= 1