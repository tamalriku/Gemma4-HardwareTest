import gradio as gr
import torch
import spaces
import tempfile
import os
import re
from transformers import pipeline, BitsAndBytesConfig

# Configuration for 2026 Models
MODELS = {
    "Gemma 4 31B (Dense) - Max Logic": "google/gemma-4-31B-it",
    "Gemma 4 26B (MoE) - Fast Iteration": "google/gemma-4-26B-A4B-it",
    "Gemma 4 4B (Medium) - Budget-Friendly": "google/gemma-4-E4B-it",
    "Gemma 4 2B (Light) - Budget-Friendly": "google/gemma-4-E2B-it"
    
}

# --- GPU Inference Logic ---
@spaces.GPU(duration=150)
def agentic_workflow(idea, model_id, progress=gr.Progress()):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    progress(0, desc="Waking up the Agentic Engine...")
    pipe = pipeline("text-generation", model=model_id, model_kwargs={"quantization_config": quant_config}, device_map="auto")

    # PHASE 1: ARCHITECT
    progress(0.2, desc="Phase 1: Architecting Hardware...")
    arch_msg = [{"role": "system", "content": "You are a Senior Hardware Architect. Create a technical spec for this idea."},
                {"role": "user", "content": idea}]
    spec = pipe(arch_msg, max_new_tokens=512)[0]['generated_text'][-1]['content']

    # PHASE 2: ENGINEER
    progress(0.5, desc="Phase 2: Lead Engineer Coding...")
    eng_msg = [{"role": "system", "content": "Lead Firmware Engineer. Write optimized .ino code based on this spec. Output ONLY code."},
               {"role": "user", "content": spec}]
    code_raw = pipe(eng_msg, max_new_tokens=1536)[0]['generated_text'][-1]['content']
    code = re.sub(r"```(cpp|ino|arduino)?", "", code_raw).replace("```", "").strip()

    # PHASE 3: DESIGNER
    progress(0.8, desc="Phase 3: Generating Wiring Guide...")
    hw_msg = [{"role": "system", "content": "Hardware Designer. Create a point-to-point wiring table for this code."},
              {"role": "user", "content": f"Source: {code}"}]
    wiring = pipe(hw_msg, max_new_tokens=512)[0]['generated_text'][-1]['content']

    # Create temporary file for download
    temp_dir = tempfile.gettempdir()
    path = os.path.join(temp_dir, "project_sketch.ino")
    with open(path, "w") as f:
        f.write(code)

    return spec, code, wiring, path

# --- Custom Theme & UI ---
theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate").set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_500",
)

with gr.Blocks(title="AI Hardware Lab") as demo:
    gr.Markdown("""
    # 🤖 Gemma 4 Hardware Agent
    ### Professional-grade Arduino Project Generation for Robotics & Engineering
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            idea_input = gr.Textbox(
                label="What's the project idea?", 
                placeholder="Describe your vision (e.g. A 3D-printed smart planter with soil moisture triggers...)",
                lines=3
            )
            model_sel = gr.Dropdown(choices=list(MODELS.values()), value=MODELS["Gemma 4 31B (Dense) - Max Logic"], label="Select Intelligence Level")
            build_btn = gr.Button("🚀 Build Hardware Package", variant="primary")
            
            gr.Examples(
                examples=[
                    ["An automated hydroponics controller using ESP32, DHT22, and a 12V water pump."],
                    ["A distance-controlled MIDI controller for my Bass Guitar using VL53L1X sensors."],
                    ["A PID-controlled humanoid leg movement script for a servo-based robot."]
                ],
                inputs=idea_input
            )

        with gr.Column(scale=1):
            gr.Markdown("### Project Exports")
            download_btn = gr.DownloadButton("📥 Download .ino File", visible=False)
            status_box = gr.Markdown("*Ready for new instructions...*")

    with gr.Tabs():
        with gr.TabItem("🛠️ Firmware Code"):
            code_out = gr.Code(language="cpp", show_label=False)
        with gr.TabItem("🔌 Wiring Guide"):
            wiring_out = gr.Markdown()
        with gr.TabItem("📋 Technical Spec"):
            spec_out = gr.Markdown()

    # Interaction Logic
    def update_ui_on_success(spec, code, wiring, file_path):
        return spec, code, wiring, gr.update(value=file_path, visible=True), "✅ Project Generated Successfully!"

    build_btn.click(
        fn=agentic_workflow,
        inputs=[idea_input, model_sel],
        outputs=[spec_out, code_out, wiring_out, download_btn],
        api_name="generate"
    )

demo.launch(theme=theme, ssr_mode=False)