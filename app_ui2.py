import streamlit as st
from PIL import Image
import os
import tempfile
from ultralytics import YOLO
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and dish-to-ingredient map
model = YOLO("runs/detect/train/weights/best.pt")

with open("dish2ingredients.json") as f:
    dish_map = json.load(f)


# === Load Mistral-7B-Instruct model locally ===
@st.cache_resource
def load_mistral_model():
    model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

tokenizer, mistral_model = load_mistral_model()


# === Step 2: Recipe Generator ===
def generate_recipe_steps(dish, ingredients):
    prompt = f"Give me a clear at least 5-step recipe for making {dish} using the following ingredients: {', '.join(ingredients)}"
    inputs = tokenizer(prompt, return_tensors="pt").to(mistral_model.device)
    outputs = mistral_model.generate(**inputs, max_new_tokens=300)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# === Streamlit UI ===
st.set_page_config(page_title="CookVision", layout="centered")
st.title("🍳 CookVision: AI Cooking Assistant")
st.markdown("Upload a food image to detect the dish and get a recipe with likely ingredients.")

upload_dir = "data/uploads"
os.makedirs(upload_dir, exist_ok=True)

uploaded_file = st.file_uploader("📷 Upload a food image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, dir=upload_dir, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        image_path = tmp.name

    image = Image.open(image_path)
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)

    # === Detection Phase ===
    with st.spinner("Detecting dish..."):
        results = model(image_path)
        detected_dishes = set()

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                detected_dishes.add(class_name)

    if detected_dishes:
        for dish in detected_dishes:
            st.subheader(f"🍽️ Detected Dish: {dish.capitalize()}")

            # Get ingredients
            ingredients = dish_map.get(dish, ["❓ Ingredients not found"])
            st.markdown("**🧾 Inferred Ingredients:** " + ", ".join(ingredients))

            # Get recipe steps
            with st.spinner("🧠 Generating recipe..."):
                recipe = generate_recipe_steps(dish, ingredients)

            st.markdown("**👩‍🍳 Suggested Recipe Steps:**")
            st.markdown(recipe)
    else:
        st.warning("⚠️ No recognizable dish detected.")
else:
    st.info("Upload a food photo to get started!")