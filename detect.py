import argparse
import warnings
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from ultralytics.nn.tasks import attempt_load_one_weight


warnings.filterwarnings("ignore")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(source: Path) -> List[Path]:
    if source.is_file():
        return [source]
    return sorted([p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def read_image_unicode(image_path: Path) -> np.ndarray:
    data = image_path.read_bytes()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    return image


def write_image_unicode(image_path: Path, image: np.ndarray, quality: int = 95) -> None:
    ext = image_path.suffix.lower()
    encode_ext = ".png" if ext == ".png" else ".jpg"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if encode_ext == ".jpg" else []
    ok, encoded = cv2.imencode(encode_ext, image, params)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {image_path}")
    image_path.write_bytes(encoded.tobytes())


def letterbox(image: np.ndarray, new_shape: int = 1024, color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    h, w = image.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = new_shape - new_unpad[0]
    dh = new_shape - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w, h) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, r, (dw, dh)


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    area1 = (box1[:, 2] - box1[:, 0]).clamp(0) * (box1[:, 3] - box1[:, 1]).clamp(0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(0) * (box2[:, 3] - box2[:, 1]).clamp(0)
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter / (area1[:, None] + area2 - inter + 1e-7)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= iou_thres]
    return torch.tensor(keep, device=boxes.device, dtype=torch.long)


def scale_boxes(boxes: torch.Tensor, orig_shape: Tuple[int, int], ratio: float, pad: Tuple[float, float]) -> torch.Tensor:
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= ratio
    h, w = orig_shape
    boxes[:, 0].clamp_(0, w)
    boxes[:, 2].clamp_(0, w)
    boxes[:, 1].clamp_(0, h)
    boxes[:, 3].clamp_(0, h)
    return boxes


def preprocess(image: np.ndarray, imgsz: int, device: torch.device) -> Tuple[torch.Tensor, float, Tuple[float, float]]:
    padded, ratio, pad = letterbox(image, imgsz)
    padded = padded[:, :, ::-1].transpose(2, 0, 1)
    padded = np.ascontiguousarray(padded)
    tensor = torch.from_numpy(padded).to(device).float() / 255.0
    return tensor.unsqueeze(0), ratio, pad


def infer_one(model, image: np.ndarray, imgsz: int, conf: float, iou: float, max_det: int, device: torch.device):
    tensor, ratio, pad = preprocess(image, imgsz, device)
    with torch.inference_mode():
        preds = model(tensor)
    preds = preds[0] if isinstance(preds, (list, tuple)) else preds
    preds = preds[0].transpose(0, 1) if preds.ndim == 3 else preds.transpose(0, 1)
    boxes = xywh2xyxy(preds[:, :4])
    cls_scores = preds[:, 4:]
    scores, cls_ids = cls_scores.max(dim=1)
    mask = scores >= conf
    boxes, scores, cls_ids = boxes[mask], scores[mask], cls_ids[mask]
    if boxes.numel() == 0:
        return torch.zeros((0, 6), dtype=torch.float32)

    offsets = cls_ids.float().unsqueeze(1).repeat(1, 4) * 4096.0
    keep = nms(boxes + offsets, scores, iou)[:max_det]
    det = torch.cat((boxes[keep], scores[keep, None], cls_ids[keep, None].float()), dim=1).cpu()
    det[:, :4] = scale_boxes(det[:, :4], image.shape[:2], ratio, pad)
    return det


def draw_detections(image: np.ndarray, detections: torch.Tensor, show_labels: bool, show_conf: bool, class_name: str) -> np.ndarray:
    canvas = image.copy()
    box_color = (40, 40, 198)
    text_color = (255, 255, 255)
    font_scale = 0.85
    font_thickness = 4

    for det in detections:
        x1, y1, x2, y2, score, _ = det.tolist()
        x1, y1, x2, y2 = [int(round(v)) for v in (x1, y1, x2, y2)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 5)

        parts = []
        if show_labels:
            parts.append(class_name)
        if show_conf:
            parts.append(f"{score:.2f}")
        text = " ".join(parts)
        if text:
            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            text_x = x1
            text_y = max(text_h + 6, y1 - 6)
            bg_top = max(0, text_y - text_h - baseline - 4)
            bg_bottom = min(canvas.shape[0], text_y + baseline)
            bg_right = min(canvas.shape[1], text_x + text_w + 6)
            cv2.rectangle(canvas, (text_x, bg_top), (bg_right, bg_bottom), box_color, -1)
            cv2.putText(
                canvas,
                text,
                (text_x + 3, text_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA,
            )
    return canvas


def parse_device(device: str) -> torch.device:
    text = str(device).strip().lower()
    if text in {"", "cpu"}:
        return torch.device("cpu")
    if torch.cuda.is_available():
        if text.startswith("cuda:"):
            return torch.device(text)
        if text.isdigit():
            return torch.device(f"cuda:{text}")
        return torch.device("cuda:0")
    return torch.device("cpu")


def build_args() -> argparse.Namespace:
    package_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="InfloClusterNet (FDPN-DGSI-DSRHead) detection runner")
    parser.add_argument("--model", type=Path, default=package_dir / "weight/InfloClusterNet.pt")
    parser.add_argument("--source", type=Path, default=package_dir / "samples/input")
    parser.add_argument("--output", type=Path, default=package_dir / "samples/output")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.4)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="0")
    parser.add_argument("--class-name", default="Cluster")
    parser.add_argument("--show-labels", action="store_true", default=True)
    parser.add_argument("--hide-labels", action="store_false", dest="show_labels")
    parser.add_argument("--show-conf", action="store_true")
    parser.add_argument("--save-quality", type=int, default=95)
    return parser.parse_args()


def main() -> None:
    args = build_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.source.exists():
        raise FileNotFoundError(f"Source not found: {args.source}")

    image_list = list_images(args.source)
    if not image_list:
        raise FileNotFoundError(f"No images found in: {args.source}")

    args.output.mkdir(parents=True, exist_ok=True)
    device = parse_device(args.device)
    model, _ = attempt_load_one_weight(args.model, device=device)

    for idx, image_path in enumerate(image_list, start=1):
        image = read_image_unicode(image_path)
        detections = infer_one(
            model=model,
            image=image,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=device,
        )
        visualized = draw_detections(
            image=image,
            detections=detections,
            show_labels=args.show_labels,
            show_conf=args.show_conf,
            class_name=args.class_name,
        )
        out_path = args.output / image_path.name
        write_image_unicode(out_path, visualized, quality=args.save_quality)
        print(f"processed={idx}/{len(image_list)} image_name={image_path.name} det_num={len(detections)}")

    print(f"Processed {len(image_list)} image(s) from: {args.source}")
    print(f"output_dir={args.output}")


if __name__ == "__main__":
    main()
