from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
from pathlib import Path
import time
import uuid
from detect import zero_shot_detect, one_shot_detect, few_shot_detect

app = FastAPI()

# Create directories
os.makedirs("static", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("examples", exist_ok=True)

# serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home():
    index_path = Path("static/index.html")
    if not index_path.exists():
        return HTMLResponse("<h1>index.html missing!</h1>")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/detect")
async def detect(
        file: UploadFile = File(...),
        prompt: str = Form(...),
        shot_type: str = Form("zero"),
        example_images: str = Form("")
):
    try:
        start_time = time.time()

        # Save uploaded image
        file_id = str(uuid.uuid4())[:8]
        input_path = f"static/input_{file_id}.png"
        output_path = f"static/output_{file_id}.png"

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"\n{'=' * 50}")
        print(f"📸 Processing: {file.filename}")
        print(f"🎯 Prompt: {prompt}")
        print(f"⚡ Shot type: {shot_type}")

        # Call appropriate function
        if shot_type == "zero":
            boxes, masks, labels, out_path = zero_shot_detect(input_path, prompt, output_path)
            shot_info = "Zero-shot"
        elif shot_type == "one":
            # Parse example for one-shot
            example_path = f"examples/{example_images}" if example_images else "examples/cat1.jpg"
            boxes, masks, labels, out_path = one_shot_detect(input_path, example_path, prompt, output_path)
            shot_info = "One-shot"
        else:  # few-shot
            # Parse multiple examples
            example_files = [f"examples/{f.strip()}" for f in example_images.split(",") if f.strip()]
            if not example_files:
                example_files = ["examples/cat1.jpg", "examples/cat2.jpg"]
            boxes, masks, labels, out_path = few_shot_detect(input_path, example_files, prompt, output_path)
            shot_info = "Few-shot"

        elapsed = time.time() - start_time

        # Return correct URL
        output_filename = os.path.basename(out_path)
        output_url = f"/static/{output_filename}?t={int(time.time())}"

        return JSONResponse(content={
            "boxes": boxes,
            "labels": labels,
            "masks_img": output_url,
            "shot_info": shot_info,
            "num_objects": len(boxes),
            "time_taken": f"{elapsed:.2f}s"
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/examples")
async def list_examples():
    """List example images"""
    examples = []
    if os.path.exists("examples"):
        for f in os.listdir("examples"):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                examples.append({
                    "filename": f,
                    "path": f"/examples/{f}"
                })
    return {"examples": examples}


if __name__ == "__main__":
    import uvicorn

    print("\n Server starting at http://localhost:9000")
    uvicorn.run(app, host="0.0.0.0", port=9000)