def clean_data(df):

    print("Cleaning dataset...")

    # Remove unnecessary columns
    columns_to_drop = [
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin"
    ]

    df = df.drop(
        columns=columns_to_drop
    )

    # Remove duplicate rows
    df = df.drop_duplicates()

    print("Cleaning completed")

    return df