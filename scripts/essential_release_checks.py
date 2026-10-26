import os
import sys
import json
import traceback
import io

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def run_essential_checks():
    results = {
        "flask_app_startup": False,
        "models_loaded": {},
        "genuine_image_inference": None,
        "deepfake_image_inference": None,
        "video_inference": None,
        "consistency_check": False,
        "errors": []
    }

    print("=== [1/5] Checking Backend App Startup & Routes ===")
    try:
        from backend.app import create_app
        app = create_app()
        app.testing = True
        client = app.test_client()
        resp = client.get("/api/health")
        if resp.status_code == 200:
            print(f" -> Health check endpoint returned 200 OK: {resp.get_json()}")
            results["flask_app_startup"] = True
        else:
            print(f" -> Health check returned status {resp.status_code}")
            results["errors"].append(f"Health check failed: {resp.status_code}")
    except Exception as e:
        print(f" -> Flask startup error: {e}")
        results["errors"].append(f"Flask startup exception: {e}")
        traceback.print_exc()

    print("\n=== [2/5] Checking Model Checkpoints & Device Manager ===")
    try:
        from backend.utils.device_manager import DeviceManager
        healthy, reason = DeviceManager.probe_cuda()
        print(f" -> DeviceManager CUDA probe: healthy={healthy}, reason={reason}")

        from backend.agents.image_agent import ImageAgent
        img_agent = ImageAgent()
        print(" -> ImageAgent initialized successfully.")
        
        # Verify component models
        has_baseline = img_agent.model is not None
        has_xcep = img_agent.xception_detector is not None
        has_vit = img_agent.vit_detector is not None
        results["models_loaded"]["image_resnet18"] = has_baseline
        results["models_loaded"]["image_xception"] = has_xcep
        results["models_loaded"]["image_vit"] = has_vit
        print(f" -> Models status: ResNet18={has_baseline}, Xception={has_xcep}, ViT={has_vit}")

        from backend.agents.video_agent import VideoAgent
        vid_agent = VideoAgent()
        has_vid_model = vid_agent.model is not None
        results["models_loaded"]["video_model"] = has_vid_model
        print(f" -> VideoAgent initialized. VideoModel={has_vid_model}")
    except Exception as e:
        print(f" -> Model loading error: {e}")
        results["errors"].append(f"Model loading exception: {e}")
        traceback.print_exc()

    print("\n=== [3/5] End-to-End Genuine Image Inference ===")
    genuine_path = os.path.join(PROJECT_ROOT, "scratch", "image_eval_set", "real_celeb_portrait_01.jpg")
    if os.path.exists(genuine_path):
        try:
            with open(genuine_path, "rb") as f:
                data = {"file": (f, "real_celeb_portrait_01.jpg")}
                resp = client.post("/api/analyze", data=data, content_type="multipart/form-data")
            if resp.status_code == 200:
                payload = resp.get_json()
                fusion = payload.get("fusion_result", {}) or {}
                cv = fusion.get("canonical_verdict") or fusion.get("verdict")
                prob = fusion.get("fake_probability")
                explanation = payload.get("explanation", {}) or fusion.get("explanation", {})
                evidence_count = len(explanation.get("evidence", []))
                cards_count = len(fusion.get("evidence_cards", []))
                print(f" -> Genuine E2E Result: verdict={cv}, fake_p={prob}, evidence_items={evidence_count}, cards={cards_count}")
                results["genuine_image_inference"] = {
                    "sample": os.path.basename(genuine_path),
                    "verdict": cv,
                    "fake_probability": prob,
                    "evidence_count": evidence_count,
                    "cards_count": cards_count,
                    "recommendation": str(explanation.get("recommendation", ""))[:80] + "..."
                }
            else:
                print(f" -> /api/analyze failed with status {resp.status_code}: {resp.data}")
                results["errors"].append(f"/api/analyze genuine failed: {resp.status_code}")
        except Exception as e:
            print(f" -> Genuine image inference failed: {e}")
            results["errors"].append(f"Genuine image inference exception: {e}")
            traceback.print_exc()
    else:
        print(f" -> Sample missing: {genuine_path}")
        results["errors"].append(f"Missing sample: {genuine_path}")

    print("\n=== [4/5] End-to-End Synthetic / Deepfake Image Inference ===")
    fake_path = os.path.join(PROJECT_ROOT, "scratch", "image_eval_set", "fake_celeb_synth_00.jpg")
    if os.path.exists(fake_path):
        try:
            with open(fake_path, "rb") as f:
                data = {"file": (f, "fake_celeb_synth_00.jpg")}
                resp = client.post("/api/analyze", data=data, content_type="multipart/form-data")
            if resp.status_code == 200:
                payload = resp.get_json()
                fusion = payload.get("fusion_result", {}) or {}
                cv = fusion.get("canonical_verdict") or fusion.get("verdict")
                prob = fusion.get("fake_probability")
                explanation = payload.get("explanation", {}) or fusion.get("explanation", {})
                evidence_count = len(explanation.get("evidence", []))
                cards_count = len(fusion.get("evidence_cards", []))
                print(f" -> Deepfake E2E Result: verdict={cv}, fake_p={prob}, evidence_items={evidence_count}, cards={cards_count}")
                results["deepfake_image_inference"] = {
                    "sample": os.path.basename(fake_path),
                    "verdict": cv,
                    "fake_probability": prob,
                    "evidence_count": evidence_count,
                    "cards_count": cards_count,
                    "recommendation": str(explanation.get("recommendation", ""))[:80] + "..."
                }
            else:
                print(f" -> /api/analyze failed with status {resp.status_code}: {resp.data}")
                results["errors"].append(f"/api/analyze fake failed: {resp.status_code}")
        except Exception as e:
            print(f" -> Deepfake image inference failed: {e}")
            results["errors"].append(f"Deepfake image inference exception: {e}")
            traceback.print_exc()
    else:
        print(f" -> Sample missing: {fake_path}")
        results["errors"].append(f"Missing sample: {fake_path}")

    print("\n=== [5/5] End-to-End Video Inference ===")
    vid_path = os.path.join(PROJECT_ROOT, "backend", "uploads", "ff_deepfake_sample_01.mp4")
    if os.path.exists(vid_path):
        try:
            with open(vid_path, "rb") as f:
                data = {"file": (f, "ff_deepfake_sample_01.mp4")}
                resp = client.post("/api/analyze", data=data, content_type="multipart/form-data")
            if resp.status_code == 200:
                payload = resp.get_json()
                fusion = payload.get("fusion_result", {}) or {}
                cv = fusion.get("canonical_verdict") or fusion.get("verdict")
                prob = fusion.get("fake_probability")
                print(f" -> Video E2E Result: verdict={cv}, fake_p={prob}")
                results["video_inference"] = {
                    "sample": os.path.basename(vid_path),
                    "verdict": cv,
                    "fake_probability": prob
                }
            else:
                print(f" -> /api/analyze video failed with status {resp.status_code}: {resp.data}")
                results["errors"].append(f"/api/analyze video failed: {resp.status_code}")
        except Exception as e:
            print(f" -> Video inference failed: {e}")
            results["errors"].append(f"Video inference exception: {e}")
            traceback.print_exc()
    else:
        print(f" -> Video sample missing: {vid_path}")
        results["errors"].append(f"Missing sample: {vid_path}")

    # Consistency Check
    if results["genuine_image_inference"] and results["deepfake_image_inference"]:
        real_v = results["genuine_image_inference"].get("verdict")
        fake_v = results["deepfake_image_inference"].get("verdict")
        real_p = results["genuine_image_inference"].get("fake_probability")
        fake_p = results["deepfake_image_inference"].get("fake_probability")
        if real_p is not None and fake_p is not None:
            if real_p < fake_p and real_v in ("REAL", "UNCERTAIN") and fake_v == "FAKE":
                results["consistency_check"] = True
                print("\n -> Consistency Check PASSED: Real and Fake samples produce consistent ordinal probabilities and canonical verdicts.")

    out_file = os.path.join(PROJECT_ROOT, "research", "results", "essential_release_checks.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport written to: {out_file}")

if __name__ == "__main__":
    run_essential_checks()
