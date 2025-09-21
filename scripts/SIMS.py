import gradio as gr
from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import time

# --- 1. CONFIGURATION & MODEL LOADING ---
MODEL_PATH = Path("../runs/train/exp2/weights/best.pt")
CLASS_NAMES = ['OxygenTank', 'NitrogenTank', 'FirstAidBox', 'FireAlarm', 'SafetySwitchPanel', 'EmergencyPhone', 'FireExtinguisher']

# Load the model once to be efficient
try:
    print("Loading model...")
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- 2. PROCESSING FUNCTIONS (The "Backend") ---

def process_image(image_input):
    """
    Processes a single image, returns the annotated image and a description of detections.
    """
    if model is None:
        return None, "Error: Model not loaded."

    # Perform prediction
    results = model(image_input)
    annotated_image = results[0].plot() # Get the annotated image
    
    # Create the description
    description = "--- Objects Detected ---\n"
    if len(results[0].boxes) == 0:
        description += "No objects detected."
    else:
        for box in results[0].boxes:
            class_name = CLASS_NAMES[int(box.cls)]
            confidence = box.conf.item() * 100
            description += f"- Class: {class_name}, Confidence: {confidence:.2f}%\n"
            
    return annotated_image, description

def process_video(video_input_path):
    """
    Processes a video, returns the path to the annotated video and a summary.
    """
    if model is None:
        return None, "Error: Model not loaded."

    # Define a unique output path for the processed video
    timestamp = int(time.time())
    output_video_path = f"temp_video_output_{timestamp}.mp4"
    
    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        return None, "Error: Could not open video file."
        
    # Get video properties for the output file
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    all_detections = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform detection
        results = model(frame)
        annotated_frame = results[0].plot()

        # Log unique detections for summary
        for box in results[0].boxes:
            all_detections.add(CLASS_NAMES[int(box.cls)])
        
        out.write(annotated_frame)

    cap.release()
    out.release()
    
    # Create summary description
    summary = "--- Video Processing Complete ---\n"
    if not all_detections:
        summary += "No objects were detected in the video."
    else:
        summary += "The following object types were detected:\n"
        for obj in sorted(list(all_detections)):
            summary += f"- {obj}\n"

    return output_video_path, summary

# --- 3. GRADIO UI (The "Frontend") ---

with gr.Blocks(theme=gr.themes.Soft(), title="SIMS - Safety & Inventory Management System") as demo:
    gr.Markdown("# SIMS: Safety & Inventory Management System")
    gr.Markdown("A real-time monitoring system for critical safety equipment, powered by YOLOv8.")

    with gr.Tabs():
        with gr.TabItem("🖼️ Image Analysis"):
            gr.Markdown("Upload an image to detect safety equipment.")
            with gr.Row():
                image_input = gr.Image(type="pil", label="Upload Image")
                with gr.Column():
                    image_output = gr.Image(label="Processed Image")
                    description_output_image = gr.Textbox(label="Detection Results")
            image_button = gr.Button("Analyze Image")

        with gr.TabItem("🎬 Video Analysis"):
            gr.Markdown("Upload a video to detect safety equipment. Processing may take a few minutes.")
            with gr.Row():
                video_input = gr.Video(label="Upload Video")
                with gr.Column():
                    video_output = gr.Video(label="Processed Video (Downloadable)")
                    description_output_video = gr.Textbox(label="Processing Summary")
            video_button = gr.Button("Analyze Video")
            
    # Define button actions
    image_button.click(
        fn=process_image,
        inputs=image_input,
        outputs=[image_output, description_output_image]
    )
    
    video_button.click(
        fn=process_video,
        inputs=video_input,
        outputs=[video_output, description_output_video]
    )

if __name__ == "__main__":
    if model:
        print("Launching Gradio UI...")
        demo.launch(debug=True) # debug=True allows for easier troubleshooting