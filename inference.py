import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, AutoModel
from model import PhoBERTClassifier, ViTForSequenceClassification

LABELS = [
    'Chinh tri Xa hoi', 'Doi song', 'Khoa hoc', 'Kinh doanh', 'Phap luat',
    'Suc khoe', 'The gioi', 'The thao', 'Van hoa', 'Vi tinh'
]
MODEL_NAME_ROBERT = "vinai/phobert-base"
MODEL_NAME_VIT5 = "VietAI/vit5-base"
MODEL_PATH_ROBERT = "./phobert_finetuned_model1/pytorch_model.bin"
MODEL_PATH_VIT5 = "./vit5_model/model_weights.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_type):
    if model_type == 1: 
        model = PhoBERTClassifier(MODEL_NAME_ROBERT, num_labels=len(LABELS))
        model.load_state_dict(torch.load(MODEL_PATH_ROBERT, map_location=DEVICE))
    else:
        model = ViTForSequenceClassification(MODEL_NAME_VIT5, num_classes=len(LABELS))
        model.load_state_dict(torch.load(MODEL_PATH_VIT5, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def predict(model, text, model_type):
    
    model_name = MODEL_NAME_ROBERT if model_type == 1 else MODEL_NAME_VIT5
    
    fixed_text = " ".join(text.lower().split())
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(
        fixed_text,
        padding='max_length',
        max_length=128,
        truncation=True,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs["logits"], dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    return LABELS[pred_idx]

if __name__ == "__main__":
    model_2 = load_model(2)
    while 1:
        sample_text = input("Nhập: ")
        result = predict(model_2, sample_text, 2)
        print("Kết quả dự đoán:", result)
