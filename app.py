import gradio as gr
from ben2 import BEN_Base
from PIL import Image
import torch

# Setup device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
    # Test a simple tensor computation on the GPU
    a = torch.tensor([1.0, 2.0]).to(device)
    b = torch.tensor([3.0, 4.0]).to(device)
    print("Test computation result:", a + b)
else:
    print("CUDA is not available. Running on CPU.")

model = BEN_Base.from_pretrained("PramaLLC/BEN2")
model.to(device).eval()

def remove_background_image(image, refine_foreground):
    if image is None:
        return None
    # Inference for image background removal
    foreground = model.inference(image, refine_foreground=refine_foreground)
    return foreground

def remove_background_video(video, refine_foreground):
    if video is None:
        return "No video provided."
    # Use the provided video file path directly; ensure the file exists for processing.
    video_path = video
    output_path = "./"  # Outputs will be saved in the current directory
    model.segment_video(
        video_path=video_path,
        output_path=output_path,
        fps=0,  # 0 lets CV2 detect the frame rate
        refine_foreground=refine_foreground,
        batch=1,
        print_frames_processed=True,
        webm=False,
        rgb_value=(0, 255, 0)
    )
    return "Video segmentation complete. Check the output in the current directory."

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.Tab("Image Background Removal"):
            image_input = gr.Image(label="Input Image", type="pil")
            refine_checkbox = gr.Checkbox(label="Refine Foreground", value=False)
            image_output = gr.Image(label="Output Image")
            image_button = gr.Button("Remove Background")
            image_button.click(fn=remove_background_image, inputs=[image_input, refine_checkbox], outputs=image_output)
        with gr.Tab("Video Background Removal"):
            video_input = gr.Video(label="Input Video")
            refine_checkbox_video = gr.Checkbox(label="Refine Foreground", value=False)
            video_button = gr.Button("Remove Background")
            video_output = gr.Textbox(label="Status")
            video_button.click(fn=remove_background_video, inputs=[video_input, refine_checkbox_video], outputs=video_output)
            
demo.launch()
