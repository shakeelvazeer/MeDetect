import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from transformers import ViTModel
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import uuid
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ------------------- Flask Setup -------------------
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------- Config -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get path to project root
project_root = Path(__file__).resolve().parent.parent

# Build full paths to the model files
binary_model_path = project_root / "MODEL" / "resnet50_binary_model.pth"
hybrid_model_path = project_root / "MODEL" / "pill_detection_fusion_hybrid.pth"

# Verify model files exist
if not binary_model_path.exists():
    logger.error(f"Binary model not found at {binary_model_path}")
if not hybrid_model_path.exists():
    logger.error(f"Hybrid model not found at {hybrid_model_path}")

# Microsoft Translator API Config
translator_endpoint = "https://api.cognitive.microsofttranslator.com"
translator_location = os.getenv("TRANSLATOR_LOCATION", "southeastasia")
translator_path = '/translate'
translator_url = translator_endpoint + translator_path

# Translator key from environment variable
translator_key = os.getenv("TRANSLATOR_API_KEY")
if not translator_key:
    logger.error("Translator key is not set. Please set TRANSLATOR_API_KEY in .env file")
else:
    logger.info("Translator key loaded successfully")

# ------------------- Transforms -------------------
binary_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

efficientnet_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

vit_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ------------------- Binary Model Definition (ResNet50) -------------------
class BinaryResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = models.resnet50(pretrained=False).layer1
        self.layer2 = models.resnet50(pretrained=False).layer2
        self.layer3 = models.resnet50(pretrained=False).layer3
        self.layer4 = models.resnet50(pretrained=False).layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Load binary model
binary_model = BinaryResNet()
try:
    state_dict = torch.load(binary_model_path, map_location=device)
    binary_model.load_state_dict(state_dict)
    logger.info(f"Binary model loaded successfully from {binary_model_path}")
except RuntimeError as e:
    logger.error(f"Error loading binary model state dict: {e}")
    state_dict = torch.load(binary_model_path, map_location=device)
    model_dict = binary_model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict)
    binary_model.load_state_dict(model_dict)
binary_model = binary_model.to(device)
binary_model.eval()

# ------------------- Hybrid Model Definition -------------------
class FeatureFusionHybrid(nn.Module):
    def __init__(self, efficientnet_features, vit_model, vit_classifier, efficientnet_dim, vit_dim, num_classes):
        super().__init__()
        self.efficientnet_features = efficientnet_features
        self.vit = vit_model
        self.head = vit_classifier
        self.fusion = nn.Sequential(
            nn.Linear(efficientnet_dim + vit_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, efficientnet_x, vit_x):
        ef_feat = self.efficientnet_features(efficientnet_x)
        ef_feat = torch.flatten(ef_feat, 1)
        vit_out = self.vit(pixel_values=vit_x)
        vit_feat = vit_out.last_hidden_state[:, 0, :]
        combined = torch.cat([ef_feat, vit_feat], dim=1)
        return self.fusion(combined)

# Load hybrid model
checkpoint = torch.load(hybrid_model_path, map_location=device)
class_names = checkpoint['class_names']
num_classes = len(class_names)

efficientnet = models.efficientnet_b3(weights="IMAGENET1K_V1")
efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, num_classes)
efficientnet_features = nn.Sequential(*list(efficientnet.children())[:-1])
efficientnet_dim = 1536

vit_model = ViTModel.from_pretrained('google/vit-base-patch16-384')
vit_dim = vit_model.config.hidden_size
vit_classifier = nn.Linear(vit_dim, num_classes)

hybrid_model = FeatureFusionHybrid(
    efficientnet_features,
    vit_model,
    vit_classifier,
    efficientnet_dim,
    vit_dim,
    num_classes
)
hybrid_model.load_state_dict(checkpoint['model_state_dict'])
hybrid_model = hybrid_model.to(device)
hybrid_model.eval()
logger.info(f"Hybrid model loaded successfully from {hybrid_model_path}")

# ------------------- Translation Function -------------------
def translate_text(texts, target_language='en'):
    if not translator_key:
        logger.warning("Translator API key not available. Returning original texts.")
        return texts if isinstance(texts, list) else [texts]

    headers = {
        'Ocp-Apim-Subscription-Key': translator_key,
        'Ocp-Apim-Subscription-Region': translator_location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    params = {
        'api-version': '3.0',
        'to': target_language
    }

    body = [{'text': text} for text in texts] if isinstance(texts, list) else [{'text': texts}]
    logger.debug(f"Translation request: language={target_language}, texts={texts}")

    try:
        response = requests.post(
            translator_url,
            params=params,
            headers=headers,
            json=body
        )
        response.raise_for_status()
        translated = response.json()
        logger.debug(f"Translation response: {translated}")

        # Validate translations
        translated_texts = []
        for item in translated:
            translation = item['translations'][0]
            translated_text = translation['text']
            translated_to = translation['to']
            if translated_to != target_language:
                logger.warning(f"Translation returned language {translated_to} instead of {target_language}")
            translated_texts.append(translated_text)
        
        return translated_texts
    except requests.exceptions.HTTPError as e:
        logger.error(f"Translation HTTP error: {e}, status={response.status_code}, response={response.text}")
        return texts if isinstance(texts, list) else [texts]
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return texts if isinstance(texts, list) else [texts]

# ------------------- Prediction Endpoint -------------------
@app.route('/predict_pill', methods=['POST'])
def predict_pill():
    if 'image' not in request.files:
        logger.warning("No image provided in /predict_pill request")
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    try:
        image = Image.open(image_file.stream).convert('RGB')
    except Exception as e:
        logger.error(f"Invalid image file: {e}")
        return jsonify({'error': f'Invalid image file: {str(e)}'}), 400

    target_language = request.args.get('language', 'en')
    logger.info(f"Predict pill request with language: {target_language}")

    binary_img = binary_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        binary_output = binary_model(binary_img).squeeze()
        is_pill = binary_output.item() >= 0.5
        binary_confidence = binary_output.item() if is_pill else 1 - binary_output.item()

    if not is_pill:
        message = 'The uploaded image is not a pill. Please upload another image.'
        translated_message = translate_text([message], target_language)[0]
        return jsonify({
            'is_pill': False,
            'message': translated_message,
            'confidence': binary_confidence,
            'language': target_language
        }), 200

    efficientnet_img = efficientnet_transform(image).unsqueeze(0).to(device)
    vit_img = vit_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = hybrid_model(efficientnet_img, vit_img)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        class_idx = predicted.item()
        class_name = class_names[class_idx]

    translated_class_name = translate_text([class_name], target_language)[0]

    return jsonify({
        'is_pill': True,
        'predicted_class': translated_class_name,
        'class_index': class_idx,
        'confidence': confidence.item(),
        'binary_confidence': binary_confidence,
        'language': target_language
    })

# ------------------- Manual Translation Endpoint -------------------
@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    if not data or 'texts' not in data or 'language' not in data:
        logger.warning("Missing texts or language parameter in /translate request")
        return jsonify({'error': 'Missing texts or language parameter'}), 400

    texts = data['texts']
    target_language = data['language']
    logger.info(f"Translate request: language={target_language}, texts={texts}")

    if not isinstance(texts, dict):
        logger.warning("Texts must be an object with fields")
        return jsonify({'error': 'Texts must be an object with fields'}), 400

    text_list = []
    field_names = ['quickSummary', 'composition', 'pillUses', 'sideEffects']
    for field in field_names:
        text = texts.get(field, '')
        if text and text != 'Not available':
            text_list.append(text)
        else:
            text_list.append('')

    translated_texts = translate_text(text_list, target_language)

    result = {field: translated_texts[i] if text_list[i] else '' for i, field in enumerate(field_names)}

    return jsonify({
        'translations': result,
        'language': target_language
    })

# ------------------- Run Flask App -------------------
if __name__ == '__main__':
    app.run(port=5000, debug=True)