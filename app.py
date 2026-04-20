import gradio as gr
import torch
import spaces
import tempfile
import os
import re
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time

# ─────────────────────────────────────────────
#  CONFIGURATION: Small models that fit in 2 min
#  All models are free-to-use, no API key needed
# ─────────────────────────────────────────────
MODELS = {
    "Qwen2.5-Coder-1.5B (Fast)": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Qwen2.5-Coder-3B (Better)": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "Phi-3-Mini (Lightweight)": "microsoft/phi-3-mini-4k-instruct",
    "Gemma 4 4B (Medium) - Budget-Friendly": "google/gemma-4-E4B-it",
    "Gemma 4 2B (Light) - Budget-Friendly": "google/gemma-4-E2B-it"
}

# Cache models at module level — load once, reuse
_model_cache = {}
_tokenizer_cache = {}

@spaces.GPU(duration=120)
def load_model_cached(model_id: str):
    """Load model once and cache it."""
    if model_id not in _model_cache:
        print(f"Loading {model_id}...")
        # Use 8-bit quantization to fit smaller VRAM
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quant_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        _model_cache[model_id] = model
        _tokenizer_cache[model_id] = tokenizer
        print(f"✓ {model_id} loaded successfully")
    
    return _model_cache[model_id], _tokenizer_cache[model_id]


def call_llm(model, tokenizer, system: str, user: str, max_tokens: int = 256) -> str:
    """Single LLM call with strict token limit."""
    # Build messages
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    
    # Format for model
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer.encode(text, return_tensors="pt").to(model.device)
    
    # Generate with strict token limit
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response (remove system + user)
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    return response


def clean_code(raw: str) -> str:
    """Strip markdown fences from generated code."""
    return re.sub(r"```(cpp|ino|arduino|c\+\+)?", "", raw).replace("```", "").strip()


# ─────────────────────────────────────────────
#  CORE AGENTIC WORKFLOW
#  All inference is LOCAL on Spaces GPU
#  Model is cached so it loads only once
# ─────────────────────────────────────────────
@spaces.GPU(duration=120)
def agentic_workflow(idea: str, model_id: str, progress=gr.Progress()):
    """3-phase agentic workflow with local inference."""
    if not idea.strip():
        raise gr.Error("Please describe your project idea first.")

    # Load model once (cached)
    model, tokenizer = load_model_cached(model_id)

    # ── PHASE 1: ARCHITECT ──────────────────────────────────────
    progress(0.05, desc="Phase 1/3 — Architecting hardware spec...")
    try:
        spec = call_llm(
            model, tokenizer,
            system=(
                "You are a Senior Hardware Architect. Write a concise technical spec: "
                "list microcontroller, sensors, actuators, power requirements. Use bullet points."
            ),
            user=idea,
            max_tokens=300,  # Short spec
        )
    except Exception as e:
        raise gr.Error(f"Phase 1 failed: {str(e)}")

    # ── PHASE 2: ENGINEER ───────────────────────────────────────
    progress(0.40, desc="Phase 2/3 — Writing firmware code...")
    try:
        code_raw = call_llm(
            model, tokenizer,
            system=(
                "You are a Lead Firmware Engineer. Write complete Arduino .ino code "
                "based on this spec. Include #include, pins, setup(), loop(). "
                "Output ONLY code — no explanation."
            ),
            user=f"Spec:\n{spec}",
            max_tokens=600,  # Larger for code
        )
        code = clean_code(code_raw)
    except Exception as e:
        raise gr.Error(f"Phase 2 failed: {str(e)}")

    # ── PHASE 3: DESIGNER ───────────────────────────────────────
    progress(0.75, desc="Phase 3/3 — Generating wiring guide...")
    try:
        wiring = call_llm(
            model, tokenizer,
            system=(
                "You are a Hardware Designer. Given Arduino code, produce a wiring table "
                "with columns: Component | Pin | Arduino Pin | Notes. Be precise."
            ),
            user=f"Code:\n{code[:500]}",  # Use first 500 chars of code to save tokens
            max_tokens=300,
        )
    except Exception as e:
        raise gr.Error(f"Phase 3 failed: {str(e)}")

    # Save .ino file
    progress(0.95, desc="Packaging files...")
    path = os.path.join(tempfile.gettempdir(), "project_sketch.ino")
    with open(path, "w") as f:
        f.write(code)

    progress(1.0, desc="✅ Done!")
    return spec, code, wiring, path


# ─────────────────────────────────────────────
#  GRADIO UI
# ─────────────────────────────────────────────
theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate").set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_500",
)

with gr.Blocks(title="AI Hardware Lab", theme=theme) as demo:
    gr.Markdown("""
    # 🤖 AI Hardware Agent
    ### Local inference on free HF Spaces GPU — no credit card needed!
    
    ⚡ **3-phase pipeline:** Architect → Engineer → Wiring Designer
    
    💡 **Models:** Small, fast, local — runs entirely on Spaces free GPU
    """)

    with gr.Row():
        with gr.Column(scale=2):
            idea_input = gr.Textbox(
                label="📝 Describe your project",
                placeholder="e.g. A PID-controlled balancing robot using MPU6050 and DC motors...",
                lines=3,
            )
            model_sel = gr.Dropdown(
                choices=list(MODELS.values()),
                value=list(MODELS.values())[1],  # Default to 3B
                label="🧠 Model (larger = better but slower)",
            )
            build_btn = gr.Button("🚀 Build Hardware Package", variant="primary", size="lg")

            gr.Examples(
                examples=[
                    ["A soil moisture sensor with ESP32 that triggers a water pump when soil is dry"],
                    ["A distance-controlled synthesizer using ultrasonic sensors and tone library"],
                    ["A temperature-logged data logger with SD card using Arduino and DHT22"],
                    ["A servo-based robotic arm with 4 joints controlled via joystick"],
                ],
                inputs=idea_input,
            )

        with gr.Column(scale=1):
            gr.Markdown("""
            ### 📥 Export
            
            **💚 Free models** — no API key needed
            
            **⏱️ Fits in 2 min:** Optimized for Spaces free GPU
            
            **🔗 Tip:** Start with Qwen-3B for best quality/speed balance
            """)
            download_btn = gr.DownloadButton("📥 Download .ino", visible=False)

    with gr.Tabs():
        with gr.TabItem("🛠️ Firmware Code"):
            code_out = gr.Code(language="cpp", show_label=False)
        with gr.TabItem("🔌 Wiring Guide"):
            wiring_out = gr.Markdown()
        with gr.TabItem("📋 Technical Spec"):
            spec_out = gr.Markdown()

    # Event handler
    def on_success(spec, code, wiring, file_path):
        return spec, code, wiring, gr.update(value=file_path, visible=True)

    build_btn.click(
        fn=agentic_workflow,
        inputs=[idea_input, model_sel],
        outputs=[spec_out, code_out, wiring_out, download_btn],
        api_name="generate",
    )

demo.launch()