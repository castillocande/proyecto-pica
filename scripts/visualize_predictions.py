import argparse
import os
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

def box_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    unionArea = box1Area + box2Area - interArea

    if unionArea == 0:
        return 0
    return interArea / unionArea


def main():

    parser = argparse.ArgumentParser(description="YOLOv11 Evaluation")
    parser.add_argument("--model_path", default="yolov11/best.pt", help="Path to YOLOv11 model")
    args = parser.parse_args()

    model = YOLO(args.model_path)

    test_images = "yolov11/data_v11/valid/images"
    test_labels = "yolov11/data_v11/valid/labels"
    incorrect_folder = "results/prediction_visualization/incorrect_images"
    correct_folder = "results/prediction_visualization/correct_images"
    os.makedirs(incorrect_folder, exist_ok=True)
    os.makedirs(correct_folder, exist_ok=True)

    results = model.predict(source=test_images)
    # metrics = model.val(data="yolov11/data_v11/data.yaml", split="val", save=True, save_json=True)

    c = 0
    for result in results:
        image_path = result.path
        image_name = os.path.basename(image_path)
        label_path = os.path.join(test_labels, image_name.replace(".jpg", ".txt"))

        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        true_boxes = []
        for line in lines:
            parts = line.strip().split()
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            img_width, img_height = result.orig_shape[1], result.orig_shape[0]
            true_boxes.append((cls, [x1 * img_width, y1 * img_height, x2 * img_width, y2 * img_height]))

        pred_boxes = list(zip(result.boxes.cls.tolist(), result.boxes.xyxy.tolist()))

        matched_pred = set()
        matched_true = set()
        iou_threshold = 0.5

        for i, (t_cls, t_box) in enumerate(true_boxes):
            for j, (p_cls, p_box) in enumerate(pred_boxes):
                iou = box_iou(t_box, p_box)
                if iou > iou_threshold and t_cls == int(p_cls):
                    matched_pred.add(j)
                    matched_true.add(i)
                    break

        false_negatives = [true_boxes[i] for i in range(len(true_boxes)) if i not in matched_true]
        false_positives = [pred_boxes[j] for j in range(len(pred_boxes)) if j not in matched_pred]

        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        font_size = 20
        font = ImageFont.truetype("arial.ttf", font_size)

        for cls, box in pred_boxes:
            if cls == 0:
                color = "red"
                label = "avispa pred"
            if cls == 1:
                color = "blue"
                label = "reina pred"
            else:
                color = "yellow"      
                label = "zángano pred"  
            draw.rectangle(box, outline=color, width=5)
            draw.text((box[0], box[1] - font_size - 5), label, fill=color, font=font)

        for cls, box in true_boxes:
            if cls == 0:
                color = "hotpink"
                label = "avispa true"
            if cls == 1:
                color = "aqua"
                label = "reina true"
            else:
                color = "coral"  
                label = "zángano true" 
            draw.rectangle(box, outline= color, width=2)
            draw.text((box[0], box[3] + 5), label, fill=color, font=font)
        
        if false_negatives or false_positives:
            img.save(os.path.join(incorrect_folder, image_name))
        else: 
            img.save(os.path.join(correct_folder, image_name))

        print(f"Imagen: {image_name}")
        print(f"Falsos positivos: {len(false_positives)}")
        print(f"Falsos negativos: {len(false_negatives)}")

        c += len(false_negatives)
        c += len(false_positives)
        
    print(f"Predicciones incorrectas: {c}") 


if __name__ == "__main__":
    main()