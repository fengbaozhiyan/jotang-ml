import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from train_cnn import CatDogCNN

model = CatDogCNN()
# 加载到cpu
model.load_state_dict(torch.load("catdog_best.pth", map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        prob = F.softmax(logits, dim=1)[0]
        pred_cls = torch.argmax(prob).item()
    label_name = "猫" if pred_cls ==0 else "狗"
    return label_name, prob[pred_cls].item(), img

result_label, conf, img = predict("test_pet.jpg")
print(f"预测结果：{result_label}，置信度：{conf:.3f}")
img.save("output/infer_result.jpg")

