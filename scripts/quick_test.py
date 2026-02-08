# from src.data_loader import load_dataset
# # from configs.config import load_config  # we’ll add this import next step


# def main():
#     df = load_dataset("data/raw/insurance.csv")
#     print("Dataset loaded successfully")
#     print(df.head())
#     print(df.shape)


# if __name__ == "__main__":
#     main()


from src.data_loader import load_dataset
from src.preprocessing import (
    split_features_target,
    build_preprocessor,
    train_test_split_data,
)


def main():
    df = load_dataset("data/raw/insurance.csv")

    X, y = split_features_target(df, target_column="charges")

    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("Preprocessing successful")
    print("Train shape:", X_train_processed.shape)
    print("Test shape:", X_test_processed.shape)


if __name__ == "__main__":
    main()
