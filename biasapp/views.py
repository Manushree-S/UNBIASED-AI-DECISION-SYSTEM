from django.shortcuts import render
from .models import Dataset, Result
from .ml_model import analyze_and_train

def upload_file(request):
    if request.method == "POST":
        file = request.FILES["file"]
        dataset = Dataset.objects.create(file=file)

        # AI processing (Pandas + scikit-learn)
        result = analyze_and_train(dataset.file.path)

        # Handle error
        if "error" in result:
            return render(request, "upload.html", {"error": result["error"]})

        # Save to database (only existing field)
        Result.objects.create(
            bias_score=result["bias_score"]
        )

        # Send to frontend
        return render(request, "result.html", result)

    return render(request, "upload.html")