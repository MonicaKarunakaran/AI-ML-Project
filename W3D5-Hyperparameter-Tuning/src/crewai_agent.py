def review_model(name, params, score):

    print("\n------------------------------")
    print("CIA Mentor Review")
    print("------------------------------")
    print(f"Model : {name}")
    print(f"Best Parameters : {params}")
    print(f"CV Score : {score:.4f}")

    print("\nSuggestions")
    print("- Model trained successfully.")
    print("- Hyperparameters tuned correctly.")
    print("- Consider adding more parameter values.")
    print("- Log all experiments using MLflow.")