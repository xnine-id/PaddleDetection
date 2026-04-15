import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import re
from fuzzywuzzy import fuzz
import Levenshtein

def clean_text(text):
    """Remove spaces and non-alphanumeric characters, and convert to uppercase."""
    if not text:
        return ""
    return re.sub(r'\W+', '', str(text)).upper()

def evaluate_predictions(ground_truth_file, predictions_file, score_threshold=0.5, 
                         fuzzy_threshold=80, min_len=4, max_len=12, verbose=True):
    # Load ground truth
    with open(ground_truth_file, 'r') as f:
        gt_data = json.load(f)
    
    # Map ground truth by image basename
    gt_map = {}
    for item in gt_data:
        image_name = os.path.basename(item['image'])
        plate_text = item.get('plate_text')
        texts = []
        if isinstance(plate_text, str):
            if plate_text.strip():
                texts.append(clean_text(plate_text))
        elif isinstance(plate_text, list):
            for pt in plate_text:
                if pt and str(pt).strip():
                    texts.append(clean_text(pt))
        
        # Only add if there's actually some text
        if texts:
            if image_name not in gt_map:
                gt_map[image_name] = []
            gt_map[image_name].extend(texts)

    # Load predictions
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)

    total_gt = 0
    total_tp = 0
    total_cer = 0.0
    ignored_preds = 0

    for pred in predictions:
        image_name = pred['image_name']
        gt_texts = gt_map.get(image_name, [])
        if not gt_texts:
            continue # Skip images with no GT text
        
        # Get predictions above threshold that have valid text length
        pred_texts = []
        plates = pred.get('plates', [])
        scores = pred.get('scores', [])
        
        for i, p in enumerate(plates):
            cl_p = clean_text(p)
            score = scores[i] if i < len(scores) else 0.0
            if score >= score_threshold and min_len <= len(cl_p) <= max_len:
                pred_texts.append(cl_p)
        
        gt_matched = [False] * len(gt_texts)
        pred_matched = [False] * len(pred_texts)
        
        # Calculate fuzzy similarity for all combinations
        match_pairs = []
        for g_idx, g_text in enumerate(gt_texts):
            for p_idx, p_text in enumerate(pred_texts):
                ratio = fuzz.ratio(g_text, p_text)
                match_pairs.append((ratio, g_idx, p_idx))
        
        # Sort by highest similarity ratio
        match_pairs.sort(reverse=True, key=lambda x: x[0])
        
        image_tp = 0
        image_cer_sum = 0.0

        # Greedy match
        for ratio, g_idx, p_idx in match_pairs:
            if not gt_matched[g_idx] and not pred_matched[p_idx]:
                # If fuzzy similarity is >= fuzzy_threshold, consider it a successful read
                if ratio >= fuzzy_threshold:
                    gt_matched[g_idx] = True
                    pred_matched[p_idx] = True
                    image_tp += 1
                    
                    # Calculate Character Error Rate (CER) for the matched text
                    dist = Levenshtein.distance(gt_texts[g_idx], pred_texts[p_idx])
                    cer = dist / max(len(gt_texts[g_idx]), 1)
                    image_cer_sum += cer
        
        # For any unmatched ground truth, the CER is considered 1.0 (100% error)
        unmatched_gt = len(gt_texts) - image_tp
        image_cer_sum += unmatched_gt * 1.0

        total_gt += len(gt_texts)
        total_tp += image_tp
        total_cer += image_cer_sum
        ignored_preds += (len(pred_texts) - image_tp)

    # Calculate metrics
    accuracy = total_tp / total_gt if total_gt > 0 else 0
    avg_cer = total_cer / total_gt if total_gt > 0 else 0

    if verbose:
        print(f"=== Enhanced OCR Evaluation (Confidence Threshold: {score_threshold:.2f}) ===")
        print(f"Total Ground Truth Plates : {total_gt}")
        print(f"Successfully Read (TP)    : {total_tp} (Fuzzy Match >= {fuzzy_threshold}%)")
        print(f"Failed to Read (FN)       : {total_gt - total_tp}")
        print(f"Ignored Extra Preds       : {ignored_preds} (Filtered out/Not penalized)")
        print("-" * 45)
        print(f"OCR Accuracy (Recall)     : {accuracy:.4f} ({(accuracy*100):.2f}%)")
        print(f"Average CER               : {avg_cer:.4f} ({(avg_cer*100):.2f}%)")
        print("==================================================")
    
    return {
        'threshold': score_threshold,
        'accuracy': accuracy,
        'cer': avg_cer
    }

def generate_plots(ground_truth_file, predictions_file, output_dir):
    print("Mengevaluasi performa OCR dengan threshold 0.5...")
    evaluate_predictions(ground_truth_file, predictions_file, score_threshold=0.5, verbose=True)

    print("\nMenghasilkan grafik untuk berbagai threshold...")
    thresholds = np.arange(0.1, 1.0, 0.1)
    accuracies = []
    cers = []

    for t in thresholds:
        metrics = evaluate_predictions(ground_truth_file, predictions_file, score_threshold=t, verbose=False)
        accuracies.append(metrics['accuracy'])
        cers.append(metrics['cer'])

    # Plot Accuracy and CER vs Threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, marker='o', label='OCR Accuracy', color='blue', linewidth=2)
    plt.plot(thresholds, cers, marker='s', label='Average CER', color='red', linewidth=2)
    
    plt.title('OCR Performance vs Confidence Threshold\n(Fuzzy Match, Filtered Length, No FP Penalty)')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Score / Error Rate')
    plt.xticks(thresholds)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='center right')
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'ocr_metrics_vs_threshold.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Grafik berhasil disimpan di: {plot_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate vehicle plate TEXT predictions with fuzzy matching and CER.")
    parser.add_argument('--ground_truth', type=str, default=r"/home/jeremyjfn/app/skyure-df/PaddleDetection/storage/private/eval/results/vehicle_plate/ground_truth.json", help="Path to ground truth JSON file")
    parser.add_argument('--predictions', type=str, default=r"/home/jeremyjfn/app/skyure-df/PaddleDetection/storage/private/eval/results/vehicle_plate/predictions_vehicle_plate.json", help="Path to predictions JSON file")
    parser.add_argument('--output_dir', type=str, default=r"/home/jeremyjfn/app/skyure-df/PaddleDetection/storage/private/eval/results/vehicle_plate/", help="Directory to save plots")
    
    args = parser.parse_args()
    
    generate_plots(args.ground_truth, args.predictions, args.output_dir)
