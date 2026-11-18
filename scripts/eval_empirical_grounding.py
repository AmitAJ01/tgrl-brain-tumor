import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from src.dataset import get_dataloaders
from src.classifier import TumorClassifier
from src.segmentation import ResNetUNet
from src.metrics import compute_classification_metrics, compute_segmentation_metrics
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

_, val_loader, test_loader = get_dataloaders(cache_dir='data/cache', batch_size=32, num_workers=0)

def load_ckpt(model, path, device):
    data = torch.load(path, map_location=device)
    state = data['model_state_dict'] if isinstance(data, dict) and 'model_state_dict' in data else data
    model.load_state_dict(state)

cls_model = TumorClassifier().to(device)
load_ckpt(cls_model, 'checkpoints/best_baseline_classifier.pth', device)
cls_model.eval()

seg_model = ResNetUNet().to(device)
load_ckpt(seg_model, 'checkpoints/best_segmentation.pth', device)
seg_model.eval()

def get_preds(loader):
    p_full_list, p_roi_list, y_list, masks_list, targets_list = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch['image'].to(device)
            y = batch['label'].cpu().numpy()
            target_mask = batch['mask'].cpu().numpy()
            
            p_full = torch.softmax(cls_model(x), dim=1)
            pred_mask = seg_model(x)
            x_roi = x * pred_mask
            p_roi = torch.softmax(cls_model(x_roi), dim=1)
            
            p_full_list.append(p_full.cpu().numpy())
            p_roi_list.append(p_roi.cpu().numpy())
            y_list.append(y)
            masks_list.append(pred_mask.cpu().numpy())
            targets_list.append(target_mask)
    return (np.concatenate(p_full_list), np.concatenate(p_roi_list), 
            np.concatenate(y_list), np.concatenate(masks_list), np.concatenate(targets_list))

val_full, val_roi, val_y, val_m, val_t = get_preds(val_loader)
test_full, test_roi, test_y, test_m, test_t = get_preds(test_loader)

# Find optimal alpha on validation set strictly!
best_alpha = 0.0
best_val_acc = 0.0
for a in np.linspace(0.0, 1.0, 101):
    fused_val = (1 - a) * val_full + a * val_roi
    acc = np.mean(np.argmax(fused_val, axis=1) == val_y)
    if acc > best_val_acc:
        best_val_acc = acc
        best_alpha = a

print(f"Optimal alpha tuned strictly on validation set: {best_alpha:.2f} (Val Acc: {best_val_acc*100:.2f}%)")

base_test_acc = np.mean(np.argmax(test_full, axis=1) == test_y)
fused_test = (1 - best_alpha) * test_full + best_alpha * test_roi
fused_test_acc = np.mean(np.argmax(fused_test, axis=1) == test_y)

print(f"Baseline Test Acc: {base_test_acc*100:.2f}% ({np.sum(np.argmax(test_full, axis=1) == test_y)}/643 correct)")
print(f"Fused TGRL Test Acc: {fused_test_acc*100:.2f}% ({np.sum(np.argmax(fused_test, axis=1) == test_y)}/643 correct)")

# Per class breakdown
class_names = ["Meningioma", "Glioma", "Pituitary"]
base_p, base_r, base_f1, _ = precision_recall_fscore_support(test_y, np.argmax(test_full, axis=1))
fused_p, fused_r, fused_f1, _ = precision_recall_fscore_support(test_y, np.argmax(fused_test, axis=1))

print("\n--- Per-Class Recall & F1 Comparison ---")
for i, cname in enumerate(class_names):
    print(f"{cname:<12} | Base Recall: {base_r[i]*100:.2f}%, F1: {base_f1[i]*100:.2f}% | TGRL Recall: {fused_r[i]*100:.2f}%, F1: {fused_f1[i]*100:.2f}%")

print(f"\nMacro F1 -> Baseline: {np.mean(base_f1)*100:.2f}% | TGRL: {np.mean(fused_f1)*100:.2f}%")

test_seg = compute_segmentation_metrics(test_m, test_t)
print(f"\n2D Test Segmentation -> Dice: {test_seg['mean_dice']*100:.2f}% | IoU: {test_seg['mean_iou']*100:.2f}%")
