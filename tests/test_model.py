import pandas as pd
from src.save_load import load_model


def test_prediction_shape():
    model = load_model("model.joblib")

    input_data = pd.DataFrame(
        [
            {
                "age": 30,
                "sex": "male",
                "bmi": 25.0,
                "children": 1,
                "smoker": "no",
                "region": "southwest",
            }
        ]
    )

    prediction = model.predict(input_data)

    assert len(prediction) == 1
    assert prediction[0] > 0
