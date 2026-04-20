import gradio as gr
import tempfile
import os
import re
from huggingface_hub import InferenceClient

# ─────────────────────────────────────────────
#  OPTIMIZATION 1: Models served via HF Inference API
#  → Zero GPU time on your Space for large models.
#  → No model loading delay. No VRAM limit.
#  → Switch between models without restarting.
# ─────────────────────────────────────────────
MODELS = {
    "Qwen2.5-Coder-32B  (Best quality)":   "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen2.5-Coder-7B   (Fast + strong)":  "Qwen/Qwen2.5-Coder-7B-Instruct",
    "DeepSeek-Coder-V2  (Embedded C++)":   "deepseek-ai/DeepSeek-Coder-V2-Instruct",
    "Llama-3.3-70B       (Best reasoning)": "meta-llama/Llama-3.3-70B-Instruct",
    "Mixtral-8x7B        (Fast MoE)":       "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Mistral-7B          (Budget)":         "mistralai/Mistral-7B-Instruct-v0.3",
}

# ─────────────────────────────────────────────
#  OPTIMIZATION 2: Cache InferenceClient at module level.
#  Created once, reused across all user calls.
#  Avoids re-instantiating HTTP sessions per request.
# ─────────────────────────────────────────────
_client_cache: dict[str, InferenceClient] = {}

def get_client(model_id: str) -> InferenceClient:
    if model_id not in _client_cache:
        _client_cache[model_id] = InferenceClient(model_id)
    return _client_cache[model_id]


def call_llm(client: InferenceClient, system: str, user: str, max_tokens: int = 512) -> str:
    """Single reusable LLM call with strict token cap."""
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        # OPTIMIZATION 3: Strict max_new_tokens — generation time is linear in tokens.
        # Keep Architect + Wiring short; only code gets a larger budget.
        max_tokens=max_tokens,
        temperature=0.2,      # Low temp = more deterministic code output
        top_p=0.9,
    )
    return response.choices[0].message.content.strip()


def clean_code(raw: str) -> str:
    """Strip markdown fences from generated code."""
    return re.sub(r"```(cpp|ino|arduino|c\+\+)?", "", raw).replace("```", "").strip()


# ─────────────────────────────────────────────
#  CORE AGENTIC WORKFLOW
#  No @spaces.GPU needed — inference is remote via HF API.
#  The 3 phases run sequentially but each call is fast
#  because you're hitting a hosted endpoint, not loading weights.
# ─────────────────────────────────────────────
def agentic_workflow(idea: str, model_id: str, progress=gr.Progress()):
    if not idea.strip():
        raise gr.Error("Please describe your project idea first.")

    client = get_client(model_id)

    # ── PHASE 1: ARCHITECT ──────────────────────────────────────
    progress(0.05, desc="Phase 1/3 — Architecting hardware spec...")
    spec = call_llm(
        client,
        system=(
            "You are a Senior Hardware Architect specialising in Arduino, ESP32, and embedded systems. "
            "Write a concise technical specification: list the microcontroller, sensors, actuators, "
            "power requirements, and key logic. Use bullet points. Be specific, not verbose."
        ),
        user=idea,
        # OPTIMIZATION 4: Architect phase only needs a short spec — cap at 400 tokens.
        max_tokens=400,
    )

    # ── PHASE 2: ENGINEER ───────────────────────────────────────
    progress(0.40, desc="Phase 2/3 — Writing firmware code...")
    code_raw = call_llm(
        client,
        system=(
            "You are a Lead Firmware Engineer. Based on the hardware spec provided, write complete, "
            "compilable Arduino (.ino) code. Include all necessary #include statements, pin definitions, "
            "setup(), and loop(). Output ONLY raw code — no explanation, no markdown fences."
        ),
        user=f"Hardware Spec:\n{spec}",
        # OPTIMIZATION 5: Code is the only phase needing a large token budget.
        # 800 tokens ≈ 60–80 lines of C++ — enough for most Arduino projects.
        max_tokens=800,
    )
    code = clean_code(code_raw)

    # ── PHASE 3: DESIGNER ───────────────────────────────────────
    progress(0.75, desc="Phase 3/3 — Generating wiring guide...")
    wiring = call_llm(
        client,
        system=(
            "You are a Hardware Designer. Given Arduino firmware, produce a clear point-to-point "
            "wiring table in Markdown format with columns: Component | Component Pin | Arduino Pin | Notes. "
            "Also list any power rail connections (VCC/GND). Be precise."
        ),
        user=f"Firmware:\n{code}",
        # OPTIMIZATION 6: Wiring guide is structured text — 350 tokens is plenty.
        max_tokens=350,
    )

    # Save .ino file for download
    progress(0.95, desc="Packaging files...")
    path = os.path.join(tempfile.gettempdir(), "project_sketch.ino")
    with open(path, "w") as f:
        f.write(code)

    progress(1.0, desc="Done!")
    return spec, code, wiring, path


# ─────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────
theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate").set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_500",
)

with gr.Blocks(title="AI Hardware Lab", theme=theme) as demo:
    gr.Markdown("""
    # 🤖 AI Hardware Agent
    ### Agentic Arduino & ESP32 firmware generation — powered by HF Inference API
    > 3-phase pipeline: Architect → Engineer → Wiring Designer
    """)

    with gr.Row():
        with gr.Column(scale=2):
            idea_input = gr.Textbox(
                label="Describe your project",
                placeholder="e.g. A PID-controlled balancing robot using MPU6050 and two DC motors...",
                lines=3,
            )
            model_sel = gr.Dropdown(
                choices=list(MODELS.values()),
                value=MODELS["Qwen2.5-Coder-7B   (Fast + strong)"],
                label="Model",
            )
            build_btn = gr.Button("🚀 Build Hardware Package", variant="primary", size="lg")

            gr.Examples(
                examples=[
                    ["An automated hydroponics controller using ESP32, DHT22, and a 12V water pump relay."],
                    ["A MIDI controller for guitar using VL53L1X distance sensors over I2C."],
                    ["A PID-controlled humanoid leg using 4 MG996R servos and an MPU6050."],
                    ["A LoRa-based weather station that transmits temperature and humidity to a base station."],
                ],
                inputs=idea_input,
            )

        with gr.Column(scale=1):
            gr.Markdown("### Export")
            download_btn = gr.DownloadButton("📥 Download .ino", visible=False)
            gr.Markdown(
                "**Tip:** Qwen2.5-Coder-7B is the best starting point. "
                "Switch to Qwen2.5-Coder-32B or Llama-3.3-70B for complex projects."
            )

    with gr.Tabs():
        with gr.TabItem("🛠️ Firmware Code"):
            code_out = gr.Code(language="cpp", show_label=False)
        with gr.TabItem("🔌 Wiring Guide"):
            wiring_out = gr.Markdown()
        with gr.TabItem("📋 Technical Spec"):
            spec_out = gr.Markdown()

    build_btn.click(
        fn=agentic_workflow,
        inputs=[idea_input, model_sel],
        outputs=[spec_out, code_out, wiring_out, download_btn],
        api_name="generate",
    )

demo.launch(ssr_mode=False)