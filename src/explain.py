import shap
import matplotlib.pyplot as plt


def explain_model(model, X_sample):
    """
    Generate SHAP values for model explainability.
    """

    explainer = shap.TreeExplainer(model.named_steps["model"])
    X_transformed = model.named_steps["preprocessing"].transform(X_sample)
    shap_values = explainer(X_transformed)
    shap.summary_plot(shap_values, X_transformed)

    # explainer = shap.Explainer(model)
    # shap_values = explainer(X_sample)

    # shap.summary_plot(shap_values, X_sample)
    plt.show()
