import os
import sys
import re
from torch_fidelity import calculate_metrics

def get_available_epochs():
    """Scans the directory to find all available fake image folders from training."""
    epochs = []
    pattern = re.compile(r"^fake_images_epoch_(\d+)$")
    
    for item in os.listdir("."):
        if os.path.isdir(item):
            match = pattern.match(item)
            if match:
                epochs.append(int(match.group(1)))
    return sorted(epochs)

def evaluate_single_epoch(epoch_num, real_dir):
    """Runs the FID evaluation for a single specified epoch folder."""
    fake_dir = f"fake_images_epoch_{epoch_num}"
    
    print(f"\n🔬 Evaluating Epoch {epoch_num}...")
    
    # Guardrail: Check if the folder exists
    if not os.path.exists(fake_dir):
        print(f"❌ Error: '{fake_dir}' does not exist.")
        return None

    # Guardrail: Check image counts
    images = [f for f in os.listdir(fake_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    num_images = len(images)
    
    if num_images == 0:
        print(f"⚠️ Warning: '{fake_dir}' is empty. Skipping.")
        return None
    elif num_images < 10000:
        print(f"⏳ Notice: '{fake_dir}' only has {num_images}/10000 images. Still generating? Skipping for now.")
        return None

    # Calculate metrics
    try:
        metrics = calculate_metrics(
            input1=real_dir,
            input2=fake_dir,
            fid=True,
            cuda=True
        )
        fid_score = metrics["frechet_inception_distance"]
        print(f"✅ Epoch {epoch_num} FID Score: {fid_score:.4f}")
        return fid_score
    except Exception as e:
        print(f"❌ Error during metrics calculation for Epoch {epoch_num}: {e}")
        return None

def main():
    real_dir = "real_images"
    
    print("=" * 60)
    print("🚀 AUTOMATED DDPM FID EVALUATION MODULE")
    print("=" * 60)

    # Guardrail: Ensure ground truth real images exist
    if not os.path.exists(real_dir) or len(os.listdir(real_dir)) == 0:
        print(f"❌ Error: Ground truth folder '{real_dir}' not found or empty.")
        print("Run the training script first to save the baseline real images.")
        sys.exit(1)

    # Determine execution mode (Single Target vs Batch Scan)
    fid_results = {}
    
    if len(sys.argv) >= 2:
        # Mode A: User requested a specific epoch folder
        target_epoch = int(sys.argv[1])
        epochs_to_test = [target_epoch]
        print(f"🎯 Target Mode: Evaluating specific epoch [{target_epoch}]")
    else:
        # Mode B: Automated scanning of all generated 50-epoch checkpoints
        epochs_to_test = get_available_epochs()
        if not epochs_to_test:
            print("❌ No matching 'fake_images_epoch_X' folders found in this directory.")
            print("Usage for single check: python FID.py 50")
            sys.exit(1)
        print(f"🔍 Scan Mode: Detected {len(epochs_to_test)} evaluation folders: {epochs_to_test}")

    # Process target folders
    print(f"🔄 Extracting features and matching distribution tracking against '{real_dir}'...")
    for ep in epochs_to_test:
        score = evaluate_single_epoch(ep, real_dir)
        if score is not None:
            fid_results[ep] = score

    # =====================================================================
    # 6. FINAL PERFORMANCE SUMMARY REPORT
    # =====================================================================
    if fid_results:
        print("\n" + "=" * 50)
        print("📊 FID EVOLUTION SUMMARY REPORT")
        print("-" * 50)
        print(f"{'EPOCH':<15}{'FID SCORE':<20}{'STATUS'}")
        print("-" * 50)
        
        # Sort key entries chronologically to read easily
        for ep in sorted(fid_results.keys()):
            score = fid_results[ep]
            # Provide visual guidance indicators on generation quality
            if score > 200:
                status = "❌ High Distortion / Chaos"
            elif score > 100:
                status = "⚠️ Improving Structure"
            elif score > 50:
                status = "✨ Good / Coherent"
            else:
                status = "🔥 Excellent / High Quality"
                
            print(f"Epoch {ep:<9}{score:<20.4f}{status}")
        print("=" * 50 + "\n")
    else:
        print("\n❌ No valid evaluation folders were processed successfully.")

if __name__ == "__main__":
    main()