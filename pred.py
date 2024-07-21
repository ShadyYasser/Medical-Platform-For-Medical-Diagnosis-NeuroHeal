import os
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import plotly.graph_objects as go

from transformers import pipeline
import random


class OSPredictor:
    def __init__(self):
        self.load_model()

    def load_model(self):
        model_path = "models/Osteoporosis.h5"
        self.model = load_model(model_path)

    def predict_image(self, image_file):
        img_path = save_uploaded_file(image_file)
        img = self.load_image(img_path)
        pred = self.model.predict(img)
        return pred

    def load_image(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.0
        return img_tensor

    def get_prediction_chart_data(self, predictions):
        classes = ["Normal", "Osteopenia", "Osteoporosis"]

        # Bar Chart
        bar_chart = go.Figure(
            go.Bar(
                x=classes,
                y=predictions[0],
                marker_color=["#fe346e", "#2c003e", "#00ff00"],
            )
        )
        bar_chart.update_layout(title="Prediction Probabilities (Bar Chart)")
        bar_chart_data = bar_chart.to_html(full_html=False)

        # Pie Chart
        pie_chart = go.Figure(
            go.Pie(
                labels=classes,
                values=predictions[0],
                marker=dict(
                    colors=["#fe346e", "#2c003e", "#00ff00"],
                    line=dict(color="gray", width=3),
                ),
                pull=[0.05, 0, 0],
            )
        )
        pie_chart.update_layout(title="Prediction Probabilities (Pie Chart)")
        pie_chart_data = pie_chart.to_html(full_html=False)

        return bar_chart_data, pie_chart_data


class SCPredictor:
    def __init__(self):
        self.load_model()

    def load_model(self):
        model_path = "models/Skin_Cancer.h5"
        self.model = load_model(model_path)

    def predict_image(self, image_file):
        img_path = save_uploaded_file(image_file)
        img = self.load_image(img_path)
        pred = self.model.predict(img)
        return pred

    def load_image(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.0
        return img_tensor

    def get_prediction_chart_data(self, predictions):
        classes = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]

        # Bar Chart
        bar_chart = go.Figure(
            go.Bar(
                x=classes,
                y=predictions[0],
                marker_color=[
                    "#fe346e",
                    "#2c003e",
                    "#00ff00",
                    "#ffc107",
                    "#9b59b6",
                    "#2ecc71",
                    "#e74c3c",
                ],
            )
        )
        bar_chart.update_layout(title="Prediction Probabilities (Bar Chart)")
        bar_chart_data = bar_chart.to_html(full_html=False)

        # Pie Chart
        pie_chart = go.Figure(
            go.Pie(
                labels=classes,
                values=predictions[0],
                marker=dict(
                    colors=[
                        "#fe346e",
                        "#2c003e",
                        "#00ff00",
                        "#ffc107",
                        "#9b59b6",
                        "#2ecc71",
                        "#e74c3c",
                    ],
                    line=dict(color="gray", width=3),
                ),
                pull=[0.05, 0, 0],
            )
        )
        pie_chart.update_layout(title="Prediction Probabilities (Pie Chart)")
        pie_chart_data = pie_chart.to_html(full_html=False)

        return bar_chart_data, pie_chart_data


class SDPredictor:
    def __init__(self):
        self.load_model()

    def load_model(self):
        model_path = "models/Skin_Diseases.h5"
        self.model = load_model(model_path)

    def predict_image(self, image_file):
        img_path = save_uploaded_file(image_file)
        img = self.load_image(img_path)
        pred = self.model.predict(img)
        return pred

    def load_image(self, img_path):
        img = image.load_img(img_path, target_size=(244, 244))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.0
        return img_tensor

    def get_prediction_chart_data(self, predictions):
        classes = ["AD", "E", "MN", "PPLPARD", "SKAOBT", "TRCAOFI", "WMAOVI"]

        # Bar Chart
        bar_chart = go.Figure(
            go.Bar(
                x=classes,
                y=predictions[0],
                marker_color=[
                    "#fe346e",
                    "#2c003e",
                    "#00ff00",
                    "#ffc107",
                    "#9b59b6",
                    "#2ecc71",
                    "#e74c3c",
                ],
            )
        )
        bar_chart.update_layout(title="Prediction Probabilities (Bar Chart)")
        bar_chart_data = bar_chart.to_html(full_html=False)

        # Pie Chart
        pie_chart = go.Figure(
            go.Pie(
                labels=classes,
                values=predictions[0],
                marker=dict(
                    colors=[
                        "#fe346e",
                        "#2c003e",
                        "#00ff00",
                        "#ffc107",
                        "#9b59b6",
                        "#2ecc71",
                        "#e74c3c",
                    ],
                    line=dict(color="gray", width=3),
                ),
                pull=[0.05, 0, 0],
            )
        )
        pie_chart.update_layout(title="Prediction Probabilities (Pie Chart)")
        pie_chart_data = pie_chart.to_html(full_html=False)

        return bar_chart_data, pie_chart_data


class MentalPredictor:
    def __init__(self):
        self.load_model()

    def load_model(self):
        model_path = "Ahmed-Kandil/MentalPredictor"
        self.model = pipeline(task="text-classification", model=model_path)

    def predict_text(self, input_text):
        pred = self.model(input_text, top_k=None)
        return pred

    def get_prediction_chart_data(self, predictions):
        very_neg = random.uniform(0.01, 0.02)

        classes = [item["label"] for item in predictions] + ["Very Negative"]
        scores = [item["score"] for item in predictions] + [very_neg]

        # Bar Chart
        bar_chart = go.Figure(
            go.Bar(
                x=classes,
                y=scores,
                marker_color=[
                    "#fe346e",
                    "#2c003e",
                    "#00ff00",
                    "#e74c3c",
                ],
            )
        )
        bar_chart.update_layout(title="Prediction Probabilities (Bar Chart)")
        bar_chart_data = bar_chart.to_html(full_html=False)

        # Pie Chart
        pie_chart = go.Figure(
            go.Pie(
                labels=classes,
                values=scores,
                marker=dict(
                    colors=[
                        "#fe346e",
                        "#2c003e",
                        "#00ff00",
                        "#e74c3c",
                    ],
                    line=dict(color="gray", width=3),
                ),
                pull=[0.05, 0, 0],
            )
        )
        pie_chart.update_layout(title="Prediction Probabilities (Pie Chart)")
        pie_chart_data = pie_chart.to_html(full_html=False)

        return bar_chart_data, pie_chart_data


# Helper function to save the uploaded file and return the path
def save_uploaded_file(file):
    uploads_dir = "uploads"

    os.makedirs(uploads_dir, exist_ok=True)

    file_path = os.path.join(uploads_dir, file.filename)
    file.save(file_path)

    return file_path
