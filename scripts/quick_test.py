# from src.data_loader import load_dataset
# # from configs.config import load_config  # we’ll add this import next step


# def main():
#     df = load_dataset("data/raw/insurance.csv")
#     print("Dataset loaded successfully")
#     print(df.head())
#     print(df.shape)


# if __name__ == "__main__":
#     main()


# from src.data_loader import load_dataset
# from src.preprocessing import (
#     split_features_target,
#     build_preprocessor,
#     train_test_split_data,
# )


# def main():
#     df = load_dataset("data/raw/insurance.csv")

#     X, y = split_features_target(df, target_column="charges")

#     X_train, X_test, y_train, y_test = train_test_split_data(
#         X, y, test_size=0.2, random_state=42
#     )

#     preprocessor = build_preprocessor(X_train)

#     X_train_processed = preprocessor.fit_transform(X_train)
#     X_test_processed = preprocessor.transform(X_test)

#     print("Preprocessing successful")
#     print("Train shape:", X_train_processed.shape)
#     print("Test shape:", X_test_processed.shape)


# if __name__ == "__main__":
#     main()


# from src.data_loader import load_dataset
# from src.preprocessing import (
#     split_features_target,
#     build_preprocessor,
#     train_test_split_data,
# )
# from src.train import train_model
# from src.evaluate import evaluate_model


# def main():
#     df = load_dataset("data/raw/insurance.csv")

#     X, y = split_features_target(df, target_column="charges")

#     X_train, X_test, y_train, y_test = train_test_split_data(
#         X, y, test_size=0.2, random_state=42
#     )

#     preprocessor = build_preprocessor(X_train)

#     model = train_model(preprocessor, X_train, y_train)

#     metrics = evaluate_model(model, X_test, y_test)

#     print("Baseline Linear Regression Performance:")
#     for k, v in metrics.items():
#         print(f"{k}: {v:.2f}")


# if __name__ == "__main__":
#     main()


from src.data_loader import load_dataset
from src.preprocessing import (
    split_features_target,
    build_preprocessor,
    train_test_split_data,
)
from src.model import (
    get_linear_regression_model,
    get_random_forest_model,
)
from src.train import train_model
from src.evaluate import evaluate_model


def main():
    df = load_dataset("data/raw/insurance.csv")

    X, y = split_features_target(df, target_column="charges")

    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(X_train)

    # Linear Regression
    lr_model = get_linear_regression_model()
    lr_pipeline = train_model(preprocessor, lr_model, X_train, y_train)
    lr_metrics = evaluate_model(lr_pipeline, X_test, y_test)

    # Random Forest
    rf_model = get_random_forest_model()
    rf_pipeline = train_model(preprocessor, rf_model, X_train, y_train)
    rf_metrics = evaluate_model(rf_pipeline, X_test, y_test)

    print("\nLinear Regression Performance:")
    for k, v in lr_metrics.items():
        print(f"{k}: {v:.2f}")

    print("\nRandom Forest Performance:")
    for k, v in rf_metrics.items():
        print(f"{k}: {v:.2f}")


if __name__ == "__main__":
    main()
