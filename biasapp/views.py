from django.shortcuts import render
from .models import Dataset, Result
from .ml_model import analyze_and_train
import traceback

def upload_file(request):
    if request.method == "POST":
        try:
            # ✅ 1. Safely get uploaded file
            file = request.FILES.get("file")

            if not file:
                return render(request, "upload.html", {
                    "error": "No file uploaded. Please choose a CSV file."
                })

            # ✅ 2. Save dataset to database
            dataset = Dataset.objects.create(file=file)

            print("FILE UPLOADED PATH:", dataset.file.path)

            # ✅ 3. Run ML model safely
            result = analyze_and_train(dataset.file.path)

            print("ML RESULT:", result)

            # ❌ Handle ML failure
            if not result or "error" in result:
                return render(request, "upload.html", {
                    "error": result.get("error", "ML processing failed.")
                })

            # ❌ Validate output
            bias_score = result.get("bias_score")

            if bias_score is None:
                return render(request, "upload.html", {
                    "error": "Model did not return bias_score."
                })

            # ✅ 4. Save result in DB
            Result.objects.create(
                bias_score=bias_score
            )

            # ✅ 5. Return result page
            return render(request, "result.html", result)

        except Exception:
            # 🔥 FULL ERROR LOG (VERY IMPORTANT for Render debugging)
            print("🔥 FULL ERROR TRACEBACK:")
            print(traceback.format_exc())

            return render(request, "upload.html", {
                "error": "Internal server error occurred. Check logs."
            })

    # GET request → show upload page
    return render(request, "upload.html")