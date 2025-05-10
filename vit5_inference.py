# import torch
# import torch.nn.functional as F
# from torch import nn
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# class ViTForSequenceClassification(nn.Module):
#     def __init__(self, model_name: str, num_classes: int):
#         super(ViTForSequenceClassification, self).__init__()
#         self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

#     def forward(self, input_ids, attention_mask):
#         return self.model(input_ids=input_ids, attention_mask=attention_mask)

# LABELS = ['Chinh tri Xa hoi', 'Doi song', 'Khoa hoc', 'Kinh doanh', 'Phap luat', 'Suc khoe', 'The gioi', 'The thao', 'Van hoa', 'Vi tinh']
# MODEL_NAME = 
# MODEL_PATH = 
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def load_model():
#     model = ViTForSequenceClassification(MODEL_NAME, len(LABELS))
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#     model.to(DEVICE)
#     model.eval()
#     return model

# def predict(model, text):
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     fixed_text = " ".join(text.lower().split())

#     inputs = tokenizer(
#         fixed_text,
#         padding='max_length',
#         max_length=128,
#         truncation=True,
#         return_tensors="pt"
#     )

#     input_ids = inputs["input_ids"].to(DEVICE)
#     attention_mask = inputs["attention_mask"].to(DEVICE)

#     with torch.no_grad():
#         outputs = model(input_ids, attention_mask=attention_mask)
#         probs = F.softmax(outputs.logits, dim=1)
#         pred_idx = torch.argmax(probs, dim=1).item()

#     return LABELS[pred_idx]


# if __name__ == "__main__":
#     text = "Công ty tôi đạt doanh thu 12 tỷ trong quý đầu tiên"
#     result = predict(text)
#     print("Output:", result)
