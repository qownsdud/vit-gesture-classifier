import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm

CLASSES = ['paper', 'rock', 'scissors']

CLASS_COLORS = {
    'rock': (0, 0, 255),       # Red
    'scissors': (0, 255, 0),   # Green
    'paper': (255, 0, 0)       # Blue
}

def load_model(weights_path='weights/vit_gesture_robust.pth', device='cpu'):
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, len(CLASSES))
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def make_square_letterbox(img, target_size=224, bg_color=(255, 255, 255)):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_size, target_size, 3), bg_color, dtype=np.uint8)

    # 0.90 scaling factor to match dataset hand ratio
    scale = (target_size * 0.90) / max(h, w)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_size, target_size, 3), bg_color, dtype=np.uint8)
    dx = (target_size - nw) // 2
    dy = (target_size - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = resized
    return canvas

def extract_hand_contour(roi):
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    max_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(max_contour) < 2500:
        return None

    bx, by, bw, bh = cv2.boundingRect(max_contour)
    wrist_cutoff_y = int(by + bh * 0.8)

    wrist_mask = np.zeros_like(mask)
    cv2.drawContours(wrist_mask, [max_contour], -1, 255, thickness=cv2.FILLED)
    wrist_mask[wrist_cutoff_y:, :] = 0

    final_contours, _ = cv2.findContours(wrist_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not final_contours:
        return None

    return max(final_contours, key=cv2.contourArea)

def draw_hud(img, probs_dict, fps):
    overlay = img.copy()
    panel_x1, panel_y1, panel_x2, panel_y2 = 10, 10, 360, 160
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (20, 20, 20), -1)
    alpha = 0.65
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (panel_x1, panel_y1), (panel_x2, panel_y2), (100, 100, 100), 1)

    cv2.putText(img, f"FPS: {fps:.1f}", (panel_x1 + 10, panel_y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    display_order = ['rock', 'scissors', 'paper']
    bar_start_x = 200
    max_bar_width = 130
    start_y = panel_y1 + 55

    for i, cls in enumerate(display_order):
        prob = probs_dict.get(cls, 0.0)
        color = CLASS_COLORS[cls]
        y = start_y + (i * 30)

        label_text = f"{cls.upper():<9} {prob * 100:4.1f}%"
        cv2.putText(img, label_text, (panel_x1 + 10, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.rectangle(img, (bar_start_x, y), (bar_start_x + max_bar_width, y + 12), (50, 50, 50), -1)
        current_bar_width = int(max_bar_width * prob)
        if current_bar_width > 0:
            cv2.rectangle(img, (bar_start_x, y), (bar_start_x + current_bar_width, y + 12), color, -1)

def main():
    video_path = 'tests/test_video.mp4'

    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return

    device = torch.device('cpu')
    print("Loading robust ViT model...")
    model = load_model('weights/vit_gesture_robust.pth', device)
    print("Model loaded successfully! (+/- keys to adjust ROI scale)")

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or video_fps is None:
        video_fps = 30.0
    frame_delay = max(1, int(1000 / video_fps))

    prev_time = 0
    scale_ratio = 0.85

    while cap.isOpened():
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Video playback completed.")
            break

        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2

        # 1. Calculate dynamic ROI box size
        box_size = int(min(h, w) * scale_ratio)
        x1 = max(0, cx - box_size // 2)
        y1 = max(0, cy - box_size // 2)
        x2 = min(w, cx + box_size // 2)
        y2 = min(h, cy + box_size // 2)

        roi = frame[y1:y2, x1:x2]

        # 2. Extract hand contour
        contour = extract_hand_contour(roi)

        # 3. Tight crop around the hand
        if contour is not None:
            bx, by, bw, bh = cv2.boundingRect(contour)
            pad = int(max(bw, bh) * 0.15)
            hx1 = max(0, bx - pad)
            hy1 = max(0, by - pad)
            hx2 = min(roi.shape[1], bx + bw + pad)
            hy2 = min(roi.shape[0], by + bh + pad)
            hand_target = roi[hy1:hy2, hx1:hx2]
        else:
            hand_target = roi

        if hand_target.size == 0:
            hand_target = roi

        # 4. ViT inference with square letterbox padding
        square_hand = make_square_letterbox(hand_target, target_size=224)
        rgb_hand = cv2.cvtColor(square_hand, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_hand)
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        probs_dict = {CLASSES[i]: probs[i].item() for i in range(len(CLASSES))}
        top_idx = torch.argmax(probs).item()
        top_class = CLASSES[top_idx]
        top_conf = probs[top_idx].item()
        active_color = CLASS_COLORS[top_class]

        # 5. Visualizations
        display = frame.copy()
        cv2.rectangle(display, (x1, y1), (x2, y2), (80, 80, 80), 1, cv2.LINE_AA)

        if contour is not None:
            contour_global = contour + np.array([x1, y1])
            glow_overlay = display.copy()
            cv2.drawContours(glow_overlay, [contour_global], -1, active_color, thickness=6, lineType=cv2.LINE_AA)
            cv2.addWeighted(glow_overlay, 0.45, display, 0.55, 0, display)
            cv2.drawContours(display, [contour_global], -1, active_color, thickness=2, lineType=cv2.LINE_AA)

            top_point = tuple(contour_global[contour_global[:, :, 1].argmin()][0])
            label_text = f"{top_class.upper()} ({top_conf*100:.1f}%)"
            cv2.putText(display, label_text, (max(10, top_point[0] - 50), max(30, top_point[1] - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, active_color, 2, cv2.LINE_AA)

        # 6. Draw HUD dashboard
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        draw_hud(display, probs_dict, fps)

        # 7. Render frames
        cv2.imshow("ViT Gesture Video Inference", display)
        # cv2.imshow("What ViT Sees", square_hand)

        elapsed_ms = int((time.time() - loop_start) * 1000)
        wait_time = max(1, frame_delay - elapsed_ms)
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('q'):
            break
        elif key in (ord('+'), ord('=')):
            scale_ratio = min(0.95, scale_ratio + 0.05)
        elif key in (ord('-'), ord('_')):
            scale_ratio = max(0.20, scale_ratio - 0.05)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()