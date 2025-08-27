import os
import json
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Tests
from .serializers import TestsSerializer
from rest_framework import generics
import cloudinary.uploader
from django.conf import settings
from openai import OpenAI

client = OpenAI(api_key=settings.OPENAI_API_KEY)


def clean_gpt_json(raw_text: str) -> str:
    """
    Removes markdown code fences if present, leaving only the JSON string.
    """
    # Remove ```json and ``` markers
    if raw_text.startswith("```"):
        raw_text = "\n".join(raw_text.split("\n")[1:-1])
    return raw_text.strip()


import cv2
import numpy as np
import tempfile
import requests
import os

@api_view(["POST"])
def analyze_test(request):
    image_file = request.FILES.get("image")
    if not image_file:
        return Response({"status": "error", "message": "No image provided"})

    print(f"[DEBUG] Received image: {image_file.name} ({image_file.size} bytes)")

    # Step 1: Upload image to Cloudinary
    try:
        upload_result = cloudinary.uploader.upload(image_file, folder="tests")
        image_url = upload_result.get("secure_url")
        print(f"[DEBUG] Cloudinary Image URL: {image_url}")
    except Exception as e:
        print(f"[ERROR] Cloudinary upload failed: {e}")
        return Response({"status": "error", "message": f"Cloudinary upload failed: {str(e)}"})

    # Step 2: Save initial DB record
    test_data = {
        "device_id": request.data.get("device_id"),
        "test_type": request.data.get("test_type"),
        "image_url": image_url,
        "result": "",
        "description": "",
    }

    serializer = TestsSerializer(data=test_data)
    if not serializer.is_valid():
        return Response({
            "status": "error",
            "message": "Failed to save initial test record",
            "errors": serializer.errors
        })

    test_instance = serializer.save()

    try:
        # === NEW STEP: OpenCV-based line detection ===
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tmp_path = tmp_file.name
        tmp_file.write(requests.get(image_url).content)
        tmp_file.close()

        img = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise Exception("Failed to load image with OpenCV")

        # Crop middle region (adjust values to fit your test strip area)
        h, w = img.shape
        roi = img[int(h*0.3):int(h*0.7), int(w*0.2):int(w*0.8)]

        blurred = cv2.GaussianBlur(roi, (5,5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Sum pixel intensities row-wise
        row_sum = np.sum(edges, axis=1)
        threshold = np.mean(row_sum) * 2
        peaks = np.where(row_sum > threshold)[0]

        line_count = len(peaks)
        print(f"[DEBUG] OpenCV detected approx {line_count} lines")

        # Remove tmp
        os.remove(tmp_path)

        # Save OpenCV raw result in description
        test_instance.description = f"OpenCV detected {line_count} potential bands."
        test_instance.save()

        # Step 3: GPT Phase 1 – Verify test strip
        phase1_prompt = f"""
        You are analyzing an image. Confirm ONLY if this image contains a valid HIV test strip.
        Return JSON only:
        {{
            "is_test_image": true/false,
            "reason": "Explain why it is or isn't a test image."
        }}
        Image URL: {image_url}
        """

        phase1_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": phase1_prompt}],
            max_tokens=150
        )

        phase1_raw = phase1_response.choices[0].message.content.strip()
        print(f"[DEBUG] Phase1 GPT raw: {phase1_raw}")

        try:
            phase1_cleaned = clean_gpt_json(phase1_raw)
            phase1_data = json.loads(phase1_cleaned)
        except json.JSONDecodeError:
            phase1_data = {"is_test_image": False, "reason": "Invalid GPT JSON"}

        if not phase1_data.get("is_test_image", False):
            test_instance.result = "invalid"
            test_instance.description += f" | Invalid test image: {phase1_data.get('reason', '')}"
            test_instance.save()
            return Response({
                "status": "success",
                "result": test_instance.result,
                "description": test_instance.description,
            })

        # Step 4: GPT Phase 2 – Detect control & test line separately
        phase2_prompt = f"""
        You are analyzing a valid HIV test strip image. 
        The system detected approx {line_count} bands with OpenCV.
        Detect if the CONTROL line and TEST line are visible.
        Return JSON only in this exact structure:
        {{
            "control_line_visible": true/false,
            "test_line_visible": true/false,
            "confidence": 0-100,
            "reason": "Explain what lines are visible and why."
        }}
        Image URL: {image_url}
        """

        phase2_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": phase2_prompt}],
            max_tokens=200
        )

        phase2_raw = phase2_response.choices[0].message.content.strip()
        print(f"[DEBUG] Phase2 GPT raw: {phase2_raw}")

        try:
            phase2_cleaned = clean_gpt_json(phase2_raw)
            phase2_data = json.loads(phase2_cleaned)
        except json.JSONDecodeError:
            phase2_data = {
                "control_line_visible": False,
                "test_line_visible": False,
                "confidence": 0,
                "reason": "Invalid GPT JSON"
            }

        control = phase2_data.get("control_line_visible", False)
        test = phase2_data.get("test_line_visible", False)

        # Step 5: Determine result
        if control and not test:
            test_instance.result = "negative"
            test_instance.description += " | Only control line detected → Negative."
        elif control and test:
            test_instance.result = "positive"
            test_instance.description += " | Control + test line detected → Positive."
        else:
            test_instance.result = "invalid"
            test_instance.description += " | No valid control line detected → Invalid test."

        # Save updates
        test_instance.save()

        return Response({
            "status": "success",
            "id": test_instance.id,
            "device_id": test_instance.device_id,
            "test_type": test_instance.test_type,
            "image_url": test_instance.image_url,
            "opencv_line_count": line_count,
            "result": test_instance.result,
            "description": test_instance.description,
            "phase1_raw": phase1_raw,
            "phase2_raw": phase2_raw
        })

    except Exception as e:
        print(f"[ERROR] GPT analysis failed: {e}")
        return Response({"status": "error", "message": f"Analysis failed: {str(e)}"})


class ListHistoryView(generics.ListAPIView):
    serializer_class = TestsSerializer

    def get_queryset(self):
        device_id = self.request.query_params.get("device_id")  # GET /tests/?device_id=123
        if device_id:
            return Tests.objects.filter(device_id=device_id).order_by('-created_at')
        return Tests.objects.none()  # empty queryset if no device_id
