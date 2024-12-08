

---

# DungeonMaster: An LLM-Based D&D Game Master

## Overview
**DungeonMaster** is an AI-powered Dungeon & Dragons game master that uses Large Language Models (LLMs) to create immersive and dynamic tabletop RPG experiences. This project automates storytelling, model memory, and world-building while preserving the creativity and unpredictability of traditional D&D gameplay.

This repository contains the final project for **Northeastern University, CS 5220**, authored by **Xiaoyang Fei** and **Maalolan Bharaniraj**.

---

## Project Resources
- **Final Report Document**: [View Report](https://docs.google.com/document/d/1Ppuy3OfIk9oOIg2agufuECZvXLcEnVt2aBuh7ULXEq0/edit?usp=sharing)
- **Gradio Gameplay Panel (Google Colab)**: [Access Colab](https://colab.research.google.com/drive/1NX0eBYsWHNgxKJ9F2k9xkopDXpCkglrJ?usp=sharing)

---

## How to Run

### Option 1: Use Our Pretrained Model
1. Download the **model_output_files.zip** from this repository or shared link.
2. Upload the zip file to the [Google Colab Notebook](https://colab.research.google.com/drive/1NX0eBYsWHNgxKJ9F2k9xkopDXpCkglrJ?usp=sharing).
3. Unzip the file in Colab.
4. Request your own **Hugging Face credential token**:
   - Visit [Hugging Face's website](https://huggingface.co) and create an account (or log in if you already have one).
   - Go to your account settings, find the "Access Tokens" section, and generate a new token with the required permissions.
   - Copy the generated token.
5. Add your token to Colab:
   - In Colab, create a variable named `HF_TOKEN` by running the following code in a cell:
     ```python
     import os
     os.environ["HF_TOKEN"] = "your_hugging_face_token_here"
     ```
   - Replace `"your_hugging_face_token_here"` with the token you generated.
6. Execute the Colab cells step by step to initiate gameplay.

### Option 2: Train Your Own Model
1. Clone the repository to your local machine or desired cluster (ensure GPU access is available).
2. Modify the `discovery.command` file:
   - Replace Xiaoyang's Northeastern Discovery Cluster credentials with your own.
3. Follow these steps on your cluster:
   - Run the commands in `discovery.command` one by one.
   - Execute `train_gpu.py` to train your model.
4. Transfer the trained model output to Google Colab.
5. Follow steps 4–6 of **Option 1** to enjoy gameplay with your trained model.

---

## Acknowledgments
We extend our gratitude to our professor and teaching assistants for their support throughout this project.  
Happy Adventuring! 🐉

---