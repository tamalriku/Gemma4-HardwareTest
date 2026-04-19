---
title: Gemma4 Hardware Lab
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# 🤖 AI Hardware Lab
### *Agentic Firmware & Circuit Design powered by Gemma 4*

The **OneCore AI Hardware Lab** is an advanced, multi-agent system designed to bridge the gap between a project idea and a working hardware prototype. By leveraging the multimodal and reasoning capabilities of **Gemma 4 (31B Dense)**, this tool generates production-ready Arduino code, precise wiring guides, and step-by-step Tinkercad simulation blueprints.

---

## 🌟 Key Features

* **Multi-Agent Reasoning:** Uses a "Think-Then-Act" pipeline involving three specialized AI agents:
    * **The Architect:** Optimizes project logic and adds failsafes.
    * **The Firmware Engineer:** Writes non-blocking, memory-efficient Arduino C++.
    * **The Tinkercad Designer:** Maps components to a virtual simulation environment.
* **ZeroGPU Optimized:** Native support for Hugging Face ZeroGPU (H200), allowing high-performance 31B/26B inference without constant cost.
* **Tinkercad Integration:** Generates color-coded wiring tables specifically for the Tinkercad library.
* **One-Click Export:** Instantly download `.ino` files ready for the Arduino IDE.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Model** | Google Gemma 4 (31B-IT / 26B-MOE) |
| **Host** | Hugging Face Spaces (Pro Tier) |
| **Compute** | NVIDIA H200 (ZeroGPU) |
| **Interface** | Gradio 4.x |
| **Library** | Transformers + Accelerate + BitsAndBytes (4-bit) |

---

## 🚀 Quick Start (Deployment)

1.  **Model Access:** Ensure you have accepted the license agreement for Gemma 4 on the [Hugging Face Model Hub](https://huggingface.co/google/gemma-4-31b-it).
2.  **Environment Variables:** To run this in a Space, add your access token to the **Settings > Variables and secrets** tab:
    * `HF_TOKEN`: Your Hugging Face "Read" token.
3.  **Sync to Spaces:** * Create a new **Gradio** Space on Hugging Face.
    * Choose **ZeroGPU [NVIDIA H200]** as the hardware avaiable on Huggingface Pro.
    * Connect this GitHub repository to the Space.

---

## 📖 Usage Guide

For the best results, describe your project with specific sensors or goals.

**Example Prompts:**
* *"A solar-powered weather station using a BME280 sensor that sends data over ESP32 Deep Sleep every 30 minutes."*
* *"An ultrasonic distance sensor that controls a servo motor to open a trash can lid, with an emergency stop button."*
* *"A 4-channel MIDI controller for a bass guitar pedalboard using analog potentiometers."*

---

## 🏗️ The Agentic Workflow Logic

Unlike standard LLM prompts, this lab uses a sequential logic chain:
1.  **Spec Generation:** The Architect creates a `requirements.md`.
2.  **Code Drafting:** The Engineer references the requirements to ensure no pins are skipped.
3.  **Circuit Validation:** The Designer checks the code to see which pins are actually used and creates the wiring table.

---

## 🤝 Contributing
* Feel free to open issues or pull requests for:
* Adding support for more microcontrollers (STM32, Raspberry Pi Pico).
* Implementing Vision-Language-Action (VLA) models for physical robot arm control.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
