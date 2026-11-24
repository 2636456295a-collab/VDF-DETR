"""Evaluate saved prediction records against the manuscript test references."""
import argparse
import contextlib
import io
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def iou(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = max(0, a[2]-a[0])*max(0, a[3]-a[1]) + max(0, b[2]-b[0])*max(0, b[3]-b[1]) - intersection
    return intersection / union if union else 0

def score(rows, threshold):
    tp = fp = fn = 0
    for row in rows:
        used = set()
        for prediction in sorted(row['predictions'], key=lambda p: -p['confidence']):
            if prediction['confidence'] < threshold:
                continue
            overlap, index = max(((iou(prediction['xyxy'], box), j) for j, box in enumerate(row['reference_xyxy']) if j not in used), default=(0, -1))
            if overlap >= 0.5:
                tp += 1
                used.add(index)
            else:
                fp += 1
        fn += len(row['reference_xyxy']) - len(used)
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return {'threshold': threshold, 'P': p, 'R': r, 'F1': 2*p*r/(p+r) if p+r else 0, 'TP': tp, 'FP': fp, 'FN': fn}

parser = argparse.ArgumentParser()
parser.add_argument('--predictions', required=True, help='JSON list of per-image prediction records')
parser.add_argument('--references', required=True, help='JSON mapping image IDs to xyxy reference boxes')
parser.add_argument('--confidence', required=True, type=float, help='confidence threshold, 0.50 for the paper tables')
args = parser.parse_args()
with open(args.predictions, encoding='utf-8-sig') as f:
    rows = json.load(f)
with open(args.references, encoding='utf-8-sig') as f:
    references = json.load(f)
assert len(rows) == len(references)
assert len({r['id'] for r in rows}) == len(rows) and set(references) == {r['id'] for r in rows}
gt = {'info': {}, 'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'surfacing'}]}
predictions = []
for i, row in enumerate(rows, 1):
    row['reference_xyxy'] = references[row['id']]
    gt['images'].append({'id': i, 'width': row['width'], 'height': row['height']})
    for x1, y1, x2, y2 in row['reference_xyxy']:
        width, height = max(0, x2-x1), max(0, y2-y1)
        gt['annotations'].append({'id': len(gt['annotations'])+1, 'image_id': i, 'category_id': 1, 'bbox': [x1,y1,width,height], 'area': width*height, 'iscrowd': 0})
    for prediction in row['predictions']:
        x1, y1, x2, y2 = prediction['xyxy']
        predictions.append({'image_id': i, 'category_id': 1, 'bbox': [x1,y1,max(0,x2-x1),max(0,y2-y1)], 'score': float(prediction['confidence'])})
with contextlib.redirect_stdout(io.StringIO()):
    coco = COCO()
    coco.dataset = gt
    coco.createIndex()
    evaluator = COCOeval(coco, coco.loadRes(predictions), 'bbox')
    evaluator.params.maxDets = [1, 10, 300]
    evaluator.evaluate()
    evaluator.accumulate()
precision = evaluator.eval['precision'][:, :, 0, 0, 2]
def mean_valid(values):
    values = values[values >= 0]
    return float(values.mean())
print(json.dumps({'images': len(rows), 'targets': len(gt['annotations']), 'fixed_confidence': score(rows, args.confidence), 'AP50': mean_valid(precision[0]), 'AP50_95': mean_valid(precision)}, indent=2))
