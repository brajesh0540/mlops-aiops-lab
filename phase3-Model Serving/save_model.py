import bentoml
import joblib

# Load your sklearn model
model = joblib.load("model.pkl")

# Save it into BentoML model store
bentoml.sklearn.save_model("iris_classifier", model)

print("✅ Model saved to BentoML store")

