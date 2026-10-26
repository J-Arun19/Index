# Model Inventory & Artifact Registry (SatyaDrishti)

## Executive Summary

An exhaustive, non-destructive audit of all machine learning models, weights files, checkpoints, configurations, training logs, and local model caches was conducted within the SatyaDrishti repository (`D:\Download\satya-vision-guard-main\satya-vision-guard-main`) and local cache environments (`C:\Users\HP\.cache`).

**Key Takeaways:**
1. **Production Image Model Identity Verified**: `backend/models/weights/image_model.pth` has SHA-256 hash `cc411dafbc845e1fb65197487a87fe409bc0c7ff1520c8e4e83b4aceb4fa8aeb`. It is a byte-for-byte exact clone of `backend/models/weights/runs/image_both_20260815_233645.pth`. This checkpoint originated from an early **2-epoch pilot run** on a FaceForensics++ C23 video frame subset.
2. **Ground Truth Class Index Mapping**: **Index 0 = REAL, Index 1 = FAKE**. Verified unequivocally across training losses, datasets, inference heads, and evaluation scripts. There is no inverted label bug.
3. **Existing Trained Alternative Candidates**:
   - **SBI EfficientNet-B4** (`backend/models/experimental/weights/sbi_ffc23.tar`, 134.75 MB, SHA-256 `2b05c2ba36ccb9e9f4f9e1aae9acd443ae2e6400ce725f56104fdb175d3c3267`): Self-Blended Images model from CVPR 2022 (Shiohara & Yamasaki, Univ of Tokyo).
   - **XceptionNet** (`backend/models/experimental/weights/xception_best.pth`, 83.70 MB, SHA-256 `dd61186f9ac8e99aae85982cae338b555dfaeef042f6b699e5a55cae3998ba47`): DeepfakeBench v1.0.1 FaceForensics++ C23 detector.
   - **Vision Transformer (ViT)** (Local HuggingFace Cache `models--dima806--deepfake_vs_real_image_detection`, 343.2 MB): Google ViT-Base fine-tuned on Real vs Fake faces.
   - **Wang et al. ResNet-50** (`scratch/models/blur_jpg_prob0.5.pth`, 269.36 MB, SHA-256 `3533f3e58a3b82215f21197f2faf4e99b911ec42a4a6e2d7823e04d5f3fd245a`): UC Berkeley / Adobe CVPR 2020 CNN artifact detector.
   - **Full Image ResNet-18** (`backend/models/weights/runs/full_image_I1/model_best.pth`, 42.72 MB, SHA-256 `f9fed25debad260a276a010be5b45d1d7bd20108009d8312acab9544cb4090b7`).

---

## 1. Discovered Model Checkpoint Registry

| Model Name | Backbone Architecture | Checkpoint Path | File Size | SHA-256 Hash | Status | Media Support |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Production Image Model** | ResNet-18 (`Linear(512, 2)`) | `backend/models/weights/image_model.pth` | 42.72 MB | `cc411dafbc845e1fb65197487a87fe409bc0c7ff1520c8e4e83b4aceb4fa8aeb` | **Active Prod** | Still Images |
| **Production Video Model** | ResNet-18 + BiLSTM (10 frames) | `backend/models/weights/video_model.pth` | 45.72 MB | `a1c069bf6889b8306c398c306843f134e244187cbd8f86d66a7540d1b3e8db81` | **Active Prod** | Video (Sequences) |
| **Pilot Run Archive (Clone)** | ResNet-18 (`Linear(512, 2)`) | `backend/models/weights/runs/image_both_20260815_233645.pth` | 42.72 MB | `cc411dafbc845e1fb65197487a87fe409bc0c7ff1520c8e4e83b4aceb4fa8aeb` | Archived | Still Images |
| **Full Image I1 Best** | ResNet-18 (`Linear(512, 2)`) | `backend/models/weights/runs/full_image_I1/model_best.pth` | 42.72 MB | `f9fed25debad260a276a010be5b45d1d7bd20108009d8312acab9544cb4090b7` | Research Run | Still Images |
| **Exp C Generalization** | ResNet-18 (`Linear(512, 2)`) | `backend/models/weights/runs/exp_c_generalization/best_checkpoint.pth` | 122.82 MB | `f186c63614a0663b7009b85053a31a3e480abc8db50ce6f8ad8baff432cf5d28` | Research Run | Still Images |
| **Controlled Val Backup** | ResNet-18 (`Linear(512, 2)`) | `backend/backup_ml_controlled_validation/weights/image_model.pth` | 42.72 MB | `a4688649a6be5aa053de413f8713ebc999829e8cc6e100e3075f0df2bf33de24` | Backup | Still Images |
| **SBI EfficientNet-B4** | EfficientNet-B4 | `backend/models/experimental/weights/sbi_ffc23.tar` | 134.75 MB | `2b05c2ba36ccb9e9f4f9e1aae9acd443ae2e6400ce725f56104fdb175d3c3267` | Experimental | Still / Video frames |
| **XceptionNet** | Xception (`Linear(2048, 2)`) | `backend/models/experimental/weights/xception_best.pth` | 83.70 MB | `dd61186f9ac8e99aae85982cae338b555dfaeef042f6b699e5a55cae3998ba47` | Experimental | Still / Video frames |
| **Vision Transformer (ViT)** | ViT-Base-Patch16-224 | `C:\Users\HP\.cache\huggingface\hub\models--dima806...` | 343.25 MB | Cached HF Directory | Experimental | Still Images |
| **Wang et al. ResNet-50** | ResNet-50 (`Linear(2048, 1)`) | `scratch/models/blur_jpg_prob0.5.pth` | 269.36 MB | `3533f3e58a3b82215f21197f2faf4e99b911ec42a4a6e2d7823e04d5f3fd245a` | Offline Diagnostic | Still Images |
| **MesoInception-4** | MesoInception-4 | `scratch/models/meso4Incep_best.pth` | 0.12 MB | `16a2d7cec846e5bcbcb7f4826a1691620b80dcd7c1c63bafe040f1e811991f66` | Offline Diagnostic | Still / Video frames |
| **Torch Hub ResNet-18** | ResNet-18 ImageNet weights | `C:\Users\HP\.cache\torch\hub\checkpoints\resnet18-f37072fd.pth` | 44.66 MB | Cached ImageNet Checkpoint | Pretrained Cache | Feature Extractor |
| **Torch Hub ResNet-50** | ResNet-50 ImageNet weights | `C:\Users\HP\.cache\torch\hub\checkpoints\resnet50-11ad3fa6.pth` | 97.78 MB | Cached ImageNet Checkpoint | Pretrained Cache | Feature Extractor |

---

## 2. In-Depth Model Profiles

### Model 1: Production ResNet-18 (`backend/models/weights/image_model.pth`)
- **Architecture**: Modified ResNet-18 from `torchvision.models.resnet18`. Final fully connected layer is replaced with `nn.Linear(512, 2)` followed by softmax during inference.
- **Tensors / Parameters**: 122 parameter tensors, 11,177,538 total parameters.
- **Input Dimensions**: `(3, 224, 224)`.
- **Color Format**: RGB (read as BGR via OpenCV `cv2.imread`, converted via `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`).
- **Pixel Scaling & Normalization**: Scaled to `[0.0, 1.0]` by dividing by `255.0`. Standard ImageNet normalization:
  $$\text{Mean} = [0.485, 0.456, 0.406], \quad \text{Std} = [0.229, 0.224, 0.225]$$
  Resulting input range: approximately `[-2.118, +2.509]`.
- **Face Cropping**: Haar cascade face detector (`cv2.CascadeClassifier(haarcascade_frontalface_default.xml)`). First detected face bounding box `(x, y, w, h)` cropped and resized to `(224, 224)`. If no face is detected, full image is resized.
- **Class Index Mapping**:
  - `Index 0`: **REAL / PRISTINE**
  - `Index 1`: **FAKE / MANIPULATED**
- **Original Training Dataset**: FaceForensics++ C23 video frames (Deepfakes, Face2Face, FaceSwap, NeuralTextures) + Celeb-DF v2. 96.2% of the training images were video-extracted frames. Zero static diffusion or modern GAN portraits were included in the training set.
- **Training Duration & Hyperparameters**:
  - 2 epochs (Pilot Run `image_both_20260815_233645`). Batch size: 128. Learning rate: 0.0001 with Adam optimizer.
  - Final Training Loss: 0.3186. Training Accuracy: 62.5%.
  - Validation Loss: 1.8859. High Expected Calibration Error (ECE): 0.6838.
- **Historical Validation Threshold**: `0.04` (derived via grid search maximizing macro-F1 on the validation set).
- **License**: Non-Commercial Research Use (FaceForensics++ / Celeb-DF academic restrictions).

---

### Model 2: Self-Blended Images (SBI) EfficientNet-B4 (`backend/models/experimental/weights/sbi_ffc23.tar`)
- **Architecture**: EfficientNet-B4 (`efficientnet-pytorch` release) initialized with 2 classes (`Linear(1792, 2)`).
- **Parameters**: 17,551,714 parameters.
- **Input Dimensions**: `(3, 380, 380)`.
- **Color Format**: RGB.
- **Pixel Scaling & Normalization**: Scaled to `[0.0, 1.0]` by dividing by `255.0`. ImageNet mean and std applied.
- **Face Cropping**: Official SBI boundary margin extraction. Unlike tight face cropping, SBI expands the face bounding box by $w/4$ and $h/4$ (25% margin ratio) to preserve the boundary seams (jawline, hairline, forehead blending masks) where forgery artifacts reside.
- **Class Index Mapping**:
  - `Index 0`: **REAL**
  - `Index 1`: **FAKE**
- **Original Training Dataset**: FaceForensics++ C23 with Self-Blended Image synthesis. SBI generates synthetic pseudo-forgeries dynamically from pristine source faces by blending warped facial components back onto the same face, teaching the network boundary artifact detection without overfitting to specific GAN generators.
- **License**: Academic Non-Commercial Research License (Yamasaki Lab, The University of Tokyo).

---

### Model 3: DeepfakeBench Xception (`backend/models/experimental/weights/xception_best.pth`)
- **Architecture**: Xception backbone based on DeepfakeBench v1.0.1 (modified from Rossler et al. / Chollet). Entry, middle (8 blocks), and exit flows with depthwise separable convolutions, followed by `nn.Linear(2048, 2)`.
- **Parameters**: 20,815,170 parameters.
- **Input Dimensions**: `(3, 256, 256)`.
- **Color Format**: RGB.
- **Pixel Scaling & Normalization**: Scaled to `[-1.0, 1.0]` using transformation:
  $$\text{Input} = \frac{\text{Pixel}}{127.5} - 1.0$$
- **Face Cropping**: Tight facial bounding box crop.
- **Class Index Mapping**:
  - `Index 0`: **REAL**
  - `Index 1`: **FAKE**
- **Original Training Dataset**: FaceForensics++ C23 (Deepfakes, Face2Face, FaceSwap, NeuralTextures).
- **License**: MIT / Academic Non-Commercial Research License.

---

### Model 4: Vision Transformer (ViT) Detector (`models--dima806--deepfake_vs_real_image_detection`)
- **Architecture**: Vision Transformer Base (`google/vit-base-patch16-224-in21k`) with classification head fine-tuned for binary deepfake detection.
- **Parameters**: 85,800,194 parameters.
- **Input Dimensions**: `(3, 224, 224)`.
- **Preprocessing**: Processed via Hugging Face `AutoImageProcessor`.
- **Class Index Mapping**:
  - `Index 0`: **REAL** (`id2label[0] = 'Real'`)
  - `Index 1`: **FAKE** (`id2label[1] = 'Fake'`)
- **Original Training Dataset**: Kaggle Real and Fake Face Detection dataset.
- **License**: Apache 2.0 (Google ViT base).

---

### Model 5: Wang et al. ResNet-50 (`scratch/models/blur_jpg_prob0.5.pth`)
- **Architecture**: Standard ResNet-50 from `torchvision.models.resnet50` with final linear layer modified to `nn.Linear(2048, 1)`.
- **Parameters**: 23,510,081 parameters.
- **Input Dimensions**: `(3, 224, 224)`.
- **Normalization**: ImageNet mean and std.
- **Output Semantics**: Single scalar output logit. Sigmoid $\sigma(z)$ outputs the probability that the image was synthesized by a generative model.
- **Original Training Dataset**: ProGAN (100,000 generated images across 20 categories) trained with aggressive Gaussian Blur ($p=0.50$) and JPEG Compression ($p=0.50$) augmentations to isolate frequency-domain CNN generation fingerprints.
- **License**: Non-Commercial Research License (UC Berkeley / Adobe Research).

---

## 3. Training & Validation Datasets Present Locally

The local filesystem houses the following labelled dataset assets:

1. **`D:\Download\SatyaDrishti_Datasets\Celeb DF`**:
   - `Celeb-real` (590 genuine celebrity YouTube interviews)
   - `YouTube-real` (300 genuine YouTube videos)
   - `Celeb-synthesis` (5,639 high-quality deepfake face swaps)
   - Ground truth test split: `List_of_testing_videos.txt` (518 test videos)
2. **`D:\Download\SatyaDrishti_Datasets\FaceForensics++_C23`**:
   - `original` (1,000 pristine YouTube video sequences)
   - `Deepfakes` (1,000 autoencoder face swap videos)
   - `Face2Face` (1,000 facial reenactment videos)
   - `FaceSwap` (1,000 classical graphics face swap videos)
   - `NeuralTextures` (1,000 neural rendering reenactment videos)
   - Splits: `backend/training/splits/train.json` (360 video pairs), `val.json` (70 video pairs), `test.json` (70 video pairs).
3. **`scratch/image_eval_set/` (55 Still Images)**:
   - 20 genuine video-extracted portraits (10 FF++, 5 Celeb-real, 5 YouTube-real)
   - 4 in-the-wild genuine portraits (`profile.jpeg`, `picture.jpeg`, `HON1.jpg`, `she.jpeg`)
   - 27 manipulated video-extracted faces (Celeb-synthesis, Deepfakes, FaceSwap, Face2Face, NeuralTextures)
   - 2 in-the-wild manipulated portraits (`deepy.jpeg` [face swap], `professional.jpeg` [synthetic face])
   - 2 non-face safety edge cases (`document_contract_sample.png`, `deepfake_logo.png`)
4. **`research/experimental_fusion/datasets/validation_manifest_v2.json` (60 Frames)**:
   - 30 genuine middle-frame extractions from `splits/val.json`
   - 30 fake middle-frame extractions from `splits/val.json`
5. **`research/experimental_fusion/datasets/locked_test_manifest_v2.json`**:
   - Locked evaluation manifest strictly preserved and NOT used for model selection or threshold tuning.

---

## 4. Usage & License Summary

| Model | Permitted for Commercial Deployment? | License Classification |
| :--- | :--- | :--- |
| **Production ResNet-18** | **NO** (Trained on academic FaceForensics++ & Celeb-DF) | Non-Commercial Academic Research Only |
| **Production Video BiLSTM** | **NO** (Trained on FaceForensics++) | Non-Commercial Academic Research Only |
| **SBI EfficientNet-B4** | **NO** (Yamasaki Lab Academic License) | Non-Commercial Academic Research Only |
| **XceptionNet** | **NO** (Trained on FaceForensics++) | Non-Commercial Academic Research Only |
| **ViT Base** | **POTENTIALLY** (Apache 2.0 base, Kaggle dataset dependencies) | Open Research / Restrictive depending on Kaggle source |
| **Wang et al. ResNet-50** | **NO** (UC Berkeley / Adobe Academic License) | Non-Commercial Academic Research Only |

*Note: For commercial production deployment, models must be retrained on openly licensed or commercially cleared datasets (e.g., CC-BY-4.0, synthetic datasets with commercial rights).*
