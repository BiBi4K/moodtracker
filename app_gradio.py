import gradio as gr
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import torch
from torchvision import transforms
from model import ResNetEmocje
import face_recognition
from mtcnn import MTCNN


class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(48),
    transforms.CenterCrop(48),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

tta_transforms = [
    transform_test,
    transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.Resize(48),
        transforms.CenterCrop(48),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]),
    transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(10),
        transforms.Resize(48),
        transforms.CenterCrop(48),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetEmocje().to(device)
checkpoint = torch.load('model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

mtcnn_detector = MTCNN()


def predict_emotion(image, detector_choice):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    rgb_image = np.array(image.convert('RGB'))
    
    # Detekcja twarzy
    faces = []
    if detector_choice == 'mtcnn':
        try:
            detections = mtcnn_detector.detect_faces(rgb_image)
            print(f"MTCNN: Wykryto {len(detections)} twarzy")
            for det in detections:
                if det['confidence'] > 0.9:
                    x, y, w, h = det['box']
                    faces.append((y, y+h, x, x+w))
        except Exception as e:
            print(f"Błąd MTCNN: {e}")
    elif detector_choice == 'hog':
        try:
            face_locations = face_recognition.face_locations(rgb_image, model="hog")
            print(f"HOG: Wykryto {len(face_locations)} twarzy")
            faces = [(top, bottom, left, right) for (top, right, bottom, left) in face_locations]
        except Exception as e:
            print(f"Błąd HOG: {e}")
    else:
        raise ValueError(f"Nieprawidłowy wybór detektora: {detector_choice}")

    results = []
    warning = ""

    if not faces:
        warning = f"Uwaga: Nie wykryto twarzy za pomocą {detector_choice}. Predykcja może być niewiarygodna."
        input_image = image
        probabilities = []
        with torch.no_grad():
            for t in tta_transforms:
                img = t(input_image).unsqueeze(0).to(device)
                output = model(img)
                probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
                probabilities.append(probs)
        probabilities = np.mean(probabilities, axis=0)
        predicted_class = np.argmax(probabilities)
        emotion = class_names[predicted_class]
        confidence = probabilities[predicted_class]
        results.append({"emotion": emotion, "confidence": float(confidence), "box": None, "probabilities": probabilities.tolist()})
    else:
        for top, bottom, left, right in faces:
            try:
                face_image = image.crop((left, top, right, bottom))
                probabilities = []
                with torch.no_grad():
                    for t in tta_transforms:
                        img = t(face_image).unsqueeze(0).to(device)
                        output = model(img)
                        probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
                        probabilities.append(probs)
                probabilities = np.mean(probabilities, axis=0)
                predicted_class = np.argmax(probabilities)
                emotion = class_names[predicted_class]
                confidence = probabilities[predicted_class]
                results.append({"emotion": emotion, "confidence": float(confidence), "box": [left, top, right, bottom], "probabilities": probabilities.tolist()})
            except Exception as e:
                print(f"Błąd podczas przetwarzania twarzy: {e}")

    draw = ImageDraw.Draw(image)
    for res in results:
        if res["box"]:
            left, top, right, bottom = res["box"]
            draw.rectangle([left, top, right, bottom], outline="red", width=2)
            draw.text((left, top - 10), f"{res['emotion']} ({res['confidence']*100:.2f}%)", fill="red")

    output_text = warning + "\n" if warning else ""
    for res in results:
        output_text += f"Emocja: {res['emotion']} ({res['confidence']*100:.2f}%)\n"

    plot_path = None
    if len(results) > 1:
        emotions = [res["emotion"] for res in results]
        plot_path = create_emotion_histogram(emotions)
    elif len(results) == 1:
        probabilities = results[0]["probabilities"]
        plot_path = create_probability_plot(probabilities)

    return image, output_text, plot_path

def create_probability_plot(probabilities):
    plt.figure(figsize=(8, 4))
    sns.barplot(x=np.array(probabilities) * 100, y=class_names, palette='Blues_d')
    plt.xlabel('Pewność (%)')
    plt.title('Prawdopodobieństwa emocji')
    plt.tight_layout()
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        plot_path = tmp.name
        plt.savefig(plot_path)
        plt.close()
    return plot_path

def create_emotion_histogram(emotions):
    plt.figure(figsize=(8, 4))
    sns.histplot(emotions, bins=len(class_names), palette='Blues_d', stat='count')
    plt.xlabel('Emocje')
    plt.ylabel('Liczba')
    plt.title('Rozkład emocji')
    plt.xticks(rotation=45)
    plt.tight_layout()
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        plot_path = tmp.name
        plt.savefig(plot_path)
        plt.close()
    return plot_path

with gr.Blocks() as demo:
    gr.Markdown("# MoodTracker – Rozpoznawanie Emocji")
    gr.Markdown("Prześlij zdjęcie lub użyj kamery. Dokładność modelu: 80%")
    
    with gr.Row():
        with gr.Column():
            detector_choice = gr.Dropdown(choices=["mtcnn", "hog"], label="Detektor twarzy", value="mtcnn")
            image_input = gr.Image(type="numpy", label="Zrób zdjęcie lub prześlij obraz", interactive=True)
            predict_button = gr.Button("Rozpoznaj emocje")
        with gr.Column():
            output_image = gr.Image(label="Przetworzony obraz")
            output_text = gr.Textbox(label="Predykcja")
            output_plot = gr.Image(label="Rozkład emocji")

    predict_button.click(fn=predict_emotion, inputs=[image_input, detector_choice], outputs=[output_image, output_text, output_plot])

demo.launch(server_name="127.0.0.1", server_port=7860)