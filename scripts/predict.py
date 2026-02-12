import pandas as pd
from src.save_load import load_model


def main():
    # Load trained model
    model = load_model("model.joblib")

    # New unseen data (single example)
    input_data = pd.DataFrame(
        [
            {
                "age": 40,
                "sex": "male",
                "bmi": 32.5,
                "children": 2,
                "smoker": "yes",
                "region": "southeast",
            }
        ]
    )

    prediction = model.predict(input_data)
    print(f"Predicted medical cost: {prediction[0]:.2f}")


if __name__ == "__main__":
    main()
