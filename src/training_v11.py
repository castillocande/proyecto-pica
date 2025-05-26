from ultralytics import YOLO
import argparse
import torch

parser = argparse.ArgumentParser(description="Entrenamiento de modelo YOLOv11 para el proyecto PICA-40-B-883")
parser.add_argument("--model", default="yolo11n.pt", help="Modelo base para entrenar")
parser.add_argument("--dataset_path", default="yolov11/data_v11", help="Directorio del dataset en formato YOLOv11")
parser.add_argument("--epochs", default=150, help="Cantidad de epochs de entrenamiento")
parser.add_argument("--lr", default=0.015, help="Ritmo de aprendizaje")
parser.add_argument("--imgsz", default=640, help="Tamaño de imagen para el entrenamiento")
args = parser.parse_args()

if __name__ == "__main__":
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

    dataset = args.dataset_path
    model = YOLO("yolo11n.pt")

    results = model.train(data=f"{dataset}/data.yaml", epochs=args.epochs, imgsz=args.imgsz, batch=4, lr0=args.lr, plots=True, device=0)
    model.train()
