# Human Face Detection & Content Screening

Two small computer-vision utilities for screening uploaded images before they're allowed into a system: is there actually a real human face in the shot, and separately, is the image safe (no explicit or violent content)?

## Face presence check — `face_detection_sys_HF.ipynb`

Takes an uploaded image and decides ACCEPT or REJECT based on whether it contains a real, clearly visible human face — the kind of gate you'd put in front of a profile-picture upload or a selfie verification flow.

- Runs face detection using InsightFace's `buffalo_l` model bundle (RetinaFace-style detector) on ONNX Runtime.
- Filters out low-confidence detections and faces that are too small in the frame, so a blurry background face or a face in a poster doesn't pass.
- Returns a simple accept/reject decision with the detected bounding boxes drawn on the image.

**Tech stack:** InsightFace, ONNX Runtime, OpenCV, Pillow, NumPy, Matplotlib

## Content safety check — `NSFW_detection.ipynb`

Screens an uploaded image for explicit or violent content and returns a single ACCEPT/REJECT decision.

- Runs two independent Hugging Face image-classification pipelines side by side — one for NSFW content, one for violence — rather than relying on a single classifier.
- Rejects the image if either model crosses its own threshold, so a false negative on one check can still be caught by the other.
- Supports both direct file upload and loading an image from a URL.

**Tech stack:** Hugging Face Transformers, PyTorch, Pillow, Requests

## How they fit together

Both notebooks follow the same shape — load a pretrained model, run inference on an uploaded image, apply a threshold, return a decision — and are meant to be used as two stages of one screening pipeline: confirm there's a real face in the image, then confirm the image itself is safe.

---

**Ankit Singh**
[LinkedIn](https://www.linkedin.com/in/ankit-singh-117925249/) · [Kaggle](https://www.kaggle.com/ankit1904singh)
