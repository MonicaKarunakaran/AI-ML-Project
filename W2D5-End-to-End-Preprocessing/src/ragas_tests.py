def validate_data(df):

    print("Running data validation...")


    # No missing values
    assert (
        df.isnull()
        .sum()
        .sum()
        == 0
    )


    # Dataset should not be empty
    assert df.shape[0] > 0


    print(
        "Validation passed"
    )