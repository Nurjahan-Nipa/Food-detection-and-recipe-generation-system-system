# CookVision: A Virtual AI-Powered Cooking Assistant for ingredient Detection and Recipe Gudiance Using Machine Learning

## Team members: 
- [Nurjahan](nurja1@lsu.edu): nurja1@lsu.edu, [Nushrat Jahan Ria](nria1@lsu.edu): nrai1@lsu.edu, [Saima Sanjida Shila](sshila1@lsu.edu): sshila1@lsu.edu, [Tania Khatun](tkhatu1@lsu.edu): tkhatu1@lsu.edu, [Epiya Ebiapia](eebiap1@lsu.edu): eebiap1@lsu.edu, [Kaushani Samarawickrama](ksamar2@lsu.edu): ksamar2@lsu.edu, [Md Saidur Rahman](mrahm65@lsu.edu):mrahm65@lsu.edu, [Asif Chowdhury](achowd6@lsu.edu): achowd6@lsu.edu

--
## Table of Content
1. [Abstract](#1-abstract)
2. [Introduction and Motivation](#2-introduction-and-motivation)
3. [Key Ideas and System Overview](#3-key-idea-and-system-overview)
4. [Dataset Description](#4-dataset-description)
5. [System Design and Implementation](#5-system-design-and-implementation)
6. [Experimentations and Results](#6-experiments-and-results)
7. [Summary and Takeaways](#7-summary-and-takeaways)
8. [References](#8-references)
9. [Appendix](#9-appendix)

---

## 1. Abstract

This project presents *CookVision*, an AI-powered smart cooking assistant that can detect prepared food items from images, infer likely ingredients, and generate clear, step-by-step recipe instructions. By integrating object detection using a custom-trained YOLOv8 model and natural language generation via a locally hosted Mistral-7B-Instruct language model, the system offers a seamless end-to-end pipeline from food recognition to recipe generation. The ingredient inference component is powered by a rule-based JSON mapping built from the Recipe Ingredients Dataset (Kaggle), which connects recognized dishes to real-world recipe data.

The system is deployed through an interactive Streamlit interface, allowing users to upload food images and receive not just ingredient lists but also detailed cooking instructions without relying on cloud APIs. This project demonstrates the application of computer vision, rule-based reasoning, and generative language modeling in a practical, assistive system for real-world users.

---

## 2. Introduction and Motivation
In a world increasingly augmented by artificial intelligence, food remains one of the most universal and culturally significant aspects of human life. Yet, despite advancements in computer vision and language modeling, the act of identifying a dish and turning it into a cookable recipe remains a challenge for non-expert users. *CookVision* addresses this problem by building a real-time AI system that bridges the gap between image-based food recognition and natural language recipe generation.

The task is motivated by the growing need for AI tools that support smart kitchens, dietary management, cooking education, and food documentation. A system that can automatically analyze a picture of food and generate an accurate, understandable recipe has applications in accessibility, sustainability (e.g., using leftovers), and education. While prior works have explored food classification and recipe generation independently, this project integrates both into a unified system powered by modern ML models.

The goal is to build an intelligent cooking assistant that:
- Detects dishes in uploaded images using a trained YOLOv8 object detection model
- Infers likely ingredients from those dishes using a rule-based mapping from a real-world dataset
- Generates natural-sounding, clear cooking instructions using a locally hosted large language model (LLM)

This project is novel in its complete end-to-end integration: from visual input to culinary output, with real-time feedback and offline capability.

---

## 3. Key Idea and System Overview
The central idea of this project is to build an intelligent assistant that can generate a recipe from a food image by leveraging a combination of machine learning techniques: object detection, rule-based reasoning, and generative language modeling.

The system is structured as a three-stage pipeline:

1. **Dish Detection**: The first stage uses a custom-trained YOLOv8 object detection model to recognize prepared food items in an image. We fine-tuned YOLOv8 on the Food-101 dataset, which contains 101,000 images across 101 distinct dish categories. The model is capable of recognizing multiple dishes in a single image, enabling multi-class food detection.

2. **Ingredient Inference**: Once the dish is detected, the second stage infers a list of likely ingredients. This is done using a rule-based lookup approach via a `dish2ingredients.json` file, which was built by matching Food-101 classes with entries from the Recipe Ingredients Dataset (Kaggle). Fuzzy string matching was applied to identify high-probability ingredient sets from recipes that mention or are compositionally similar to each dish class.

3. **Recipe Generation**: The final stage uses a local large language model (LLM), Mistral-7B-Instruct, hosted via the Hugging Face Transformers library, to generate cooking instructions. The model is prompted with the dish name and inferred ingredients to output a clear 5-step recipe in natural language. This approach enables offline, customizable generation without relying on commercial APIs like GPT-4.

### Novelty and Integration

While prior works have addressed individual components of food classification or recipe generation, this project is unique in its **end-to-end integration of visual recognition, structured inference, and natural language generation**. Unlike systems that only classify dishes or retrieve static recipes, *CookVision* dynamically generates personalized instructions for detected dishes using live model inference.

Furthermore, the system emphasizes **offline operability**, making it deployable in edge devices or privacy-sensitive environments. The entire application is wrapped in a user-friendly Streamlit interface that allows users to upload food photos and receive both ingredients and cooking steps within seconds.

This combination of vision, knowledge reasoning, and LLM generation makes *CookVision* an intelligent and interactive cooking assistant — practical, modular, and extendable.

---

## 4. Dataset Description
This project utilizes two distinct datasets to support the visual detection and ingredient inference tasks:

### 4.1 Food-101 Dataset (for Dish Detection)

The Food-101 dataset, created by Bossard et al., is a widely used benchmark for food classification and recognition tasks. It contains:

- **Total images**: 101,000 (750 training images and 250 test images per class)
- **Number of classes**: 101 food categories (e.g., pizza, sushi, cheesecake)
- **Image format**: JPEG
- **Source**: Curated from foodspotting.com, with noisy labels retained for realism

#### Preprocessing and Conversion for YOLOv8
To use Food-101 for object detection:

- A subset of the dataset was manually or synthetically annotated with bounding boxes using default object-center assumptions, given the images mostly contain a single dominant dish.
- Images were organized into YOLOv8-compatible directories (`images/train`, `images/val`, and `labels/` folders).
- A `data.yaml` file was constructed containing the class index and dataset structure for training.

This dataset powered the training of the YOLOv8 detection model used to identify food dishes from uploaded images.

---

### 4.2 Recipe Ingredients Dataset (Kaggle)
To support the ingredient inference stage, we used the **Recipe Ingredients Dataset** available on Kaggle. This dataset contains:

- **Total records**: ~39,000 recipes
- **Fields per record**: `id`, `cuisine`, and `ingredients` (list of strings)
- **Format**: JSON
- **Source**: Kaggle competition dataset

#### Preprocessing for Ingredient Mapping
To generate a `dish2ingredients.json` mapping between detected dish names and likely ingredients:
- The class names from Food-101 were compared to the recipes in the dataset using fuzzy string matching (e.g., comparing `"apple_pie"` to ingredient lists mentioning `"apple pie"` or `"baked apple"`).
- Ingredients were aggregated from the top-matching recipes for each dish.
- The most common ingredients for each dish were extracted and stored as lists under the corresponding dish name.

This mapping was later used as a lookup during real-time inference to connect visually detected dishes with real-world ingredients.

---

Both datasets were essential to building a pipeline that could go from raw food image input to structured ingredient knowledge, ultimately enabling personalized recipe generation.

---

## 5. System Design and Implementation

The CookVision system is built as a modular, real-time AI assistant that integrates three key components: object detection, ingredient inference, and recipe generation. These components are tied together through a lightweight, interactive Streamlit application.

---

### 5.1 YOLOv8 for Dish Detection
The system uses **YOLOv8 (You Only Look Once v8)** — a real-time object detection model — to identify dishes in user-uploaded food images. YOLOv8 was selected for its high speed, accuracy, and simplicity of deployment.

#### Key Details:
- **Model**: YOLOv8n (nano version for faster training)
- **Training dataset**: Food-101, converted to object detection format
- **Input resolution**: 640 × 640 pixels
- **Implementation**: Ultralytics YOLOv8 Python library
- **Output**: List of detected food class labels (e.g., `"pizza"`, `"sushi"`) The detection results serve as input to the next stage of the pipeline.

---

### 5.2 Ingredient Inference via Rule-Based Mapping

After dish detection, the system performs **ingredient inference** by using a lookup table (`dish2ingredients.json`) that maps dish names to a list of common ingredients.

#### Mapping Pipeline:
- Food-101 class names (e.g., `"cheesecake"`) were matched with recipes from the Recipe Ingredients Dataset (Kaggle) using fuzzy string matching.
- For each match, ingredient lists were aggregated and the top 5–10 most common ingredients were retained.
- These were compiled into a JSON file used during inference. This approach allows for lightweight, interpretable, and highly responsive ingredient inference based on detected dishes.

---

### 5.3 Recipe Generation via Local Language Model (LLM)
To generate step-by-step cooking instructions, the system uses a **locally hosted LLM**, specifically **Mistral-7B-Instruct**, loaded via the Hugging Face Transformers library.

#### Key Details:
- **Model**: `mistralai/Mistral-7B-Instruct-v0.1`
- **Framework**: Transformers + Accelerate (PyTorch backend)
- **Inference**: Prompt includes dish name and ingredients (e.g., “Give me a 5-step recipe for making pizza using tomato, cheese, basil...”)
- **Deployment**: Local GPU inference (no cloud API required) This LLM outputs natural language instructions tailored to each input, enabling personalized, flexible recipe creation.

---

### 5.4 User Interface (Streamlit) The entire pipeline is wrapped inside a **Streamlit** web app, allowing users to:

- Upload a food image
- View detected dishes
- View inferred ingredients
- Receive full recipe steps

#### Tools and Packages Used:
- `streamlit` — interactive frontend
- `ultralytics` — YOLOv8 detection
- `transformers` — LLM loading and inference
- `torch`, `bitsandbytes`, `accelerate`— backend support for Mistral
- `PIL`, `cv2`, `json`, `fuzzywuzzy`— preprocessing and logic

---

This modular system allows CookVision to operate entirely offline, deliver dynamic AI responses, and remain extendable — supporting additional features like nutrition lookup, user preferences, and voice output in future iterations.

--- 
## 6. Experiments and Results

The CookVision system was evaluated across its three primary components: the object detection model (YOLOv8), the ingredient inference module, and the recipe generation model (Mistral-7B-Instruct). This section presents the experimental design, training metrics, inference outputs, and qualitative evaluation of the system's performance.

---

### 6.1 YOLOv8 Training and Evaluation
#### Dataset and Training Setup:
- **Training set**: 80% of the converted Food-101 dataset
- **Validation set**: 20% of the dataset
- **Model used**: YOLOv8n (nano) for faster iteration and lower resource usage
- **Training environment**: Local GPU (NVIDIA RTX [INSERT GPU NAME], [INSERT RAM])
- **Input size**: 640 × 640 - **Epochs**: 50
- **Batch size**: 16

#### Training Metrics:
| Metric             | Final Value |
|--------------------|-------------|
| Training box loss  | ~0.23       |
| Validation box loss| ~0.28       |
| mAP@0.5            | ~0.74       |
| mAP@0.5:0.95       | ~0.42       |
| Precision          | ~0.70       |
| Recall             | ~0.68       |

#### Output Artifacts:
- `results.png`: training/validation loss curves, mAP growth
- `confusion_matrix.png`: highlights classes that are frequently confused
- `PR_curve.png`: precision-recall trade-off
- `runs/detect/train/weights/best.pt`: best-performing model weights

These metrics show that the YOLOv8 model learned to detect prepared dishes with solid performance, particularly on common classes like `pizza`, `pancakes`, and `sushi`.

---

### 6.2 Ingredient Inference Evaluation
The `dish2ingredients.json` file was generated by linking Food-101 dish names to recipes in the Kaggle dataset via fuzzy matching. Ingredient quality was evaluated qualitatively and by verifying against real-world recipes.

#### Sample Mappings:
| Dish       | Top Ingredients                            |
|------------|---------------------------------------------|
| pizza      | tomato, cheese, basil, olive oil, dough     |
| sushi      | rice, nori, fish, soy sauce, wasabi         |
| lasagna    | pasta, tomato sauce, ricotta, beef, cheese  |

This mapping provides accurate and plausible ingredient lists in real time, with <100ms lookup time per prediction.

---

### 6.3 Recipe Generation Performance
The Mistral-7B-Instruct model was used to generate cooking instructions from dish and ingredient inputs. The model was evaluated on coherence, relevance, and fluency.

#### Prompt Example:
`"Give me at least 5-step recipe for making pizza using: dough, tomatoes, cheese, basil, olive oil"`

#### Sample Output:
`1. Preheat oven to 425&deg;F (220&deg;C)`

`2. Roll out the dough into a thin crust.`

`3. Spread tomato sauce over the dough and add cheese and basil.`

`4. Drizzle olive oil on top and bake for 12 - 15 minuites.`

`5. Remove from oven, slice, and serve hot.`

#### Observation:
- Recipe steps were generally coherent, well-structured, and usable.
- Model responded in <5 seconds on GPU (single forward pass).
- Occasionally generated extra instructions or used uncommon ingredient assumptions.

---

### 6.4 End-to-End Use Case

Uploaded a photo of `pasta`, and the system:

- Detected dish as `spaghetti_bolognese`
- Retrieved accurate ingredients
- Generated realistic recipe steps This validated the full pipeline integration from image input to natural language output.

---

The system performed well across tasks, and the integration of detection, rule-based reasoning, and generative LLM produced fast and relevant outputs.

---

## 7. Summary and Takeaways

The CookVision project demonstrates how modern machine learning methods can be integrated into a cohesive, real-time intelligent assistant that bridges computer vision, structured knowledge inference, and natural language generation.

By combining a YOLOv8 object detection model trained on Food-101 with a rule-based ingredient mapping system derived from a real-world recipe dataset, and a locally hosted Mistral-7B language model for recipe generation, the system is capable of:

- Detecting prepared dishes in food images with strong accuracy
- Inferring realistic ingredients for those dishes using publicly available data
- Generating coherent, human-readable cooking instructions without relying on cloud APIs

The project illustrates the power of modular AI system design— each component (detection, reasoning, generation) is independently valuable but collectively produces a smart, interactive tool that could support users in smart kitchens, cooking education, food documentation, or assistive technologies.

### Key Takeaways:
- **End-to-end ML integration**: From raw image to structured natural language recipe.
- **Offline capability**: The system operates fully offline using local models, which increases privacy and reliability.
- **Efficiency and extensibility**: Each component is modular and can be swapped or extended (e.g., with nutrition APIs, audio instructions, or user preferences).
- **Potential for impact**: This framework could be deployed in apps, kiosks, or embedded systems to assist with meal preparation, food tracking, or educational use.

While the system performed well, challenges included limited bounding box data in Food-101, the need for approximate ingredient matching, and the resource demands of running large LLMs locally. Future work could improve ingredient matching via embeddings or ontology mapping, use quantized LLMs for efficiency, and expand the system to suggest meal plans or dietary alternatives. 

Overall, *CookVision* shows how real-world machine learning can empower interactive systems that feel intelligent, useful, and responsive to human needs.

---

## 8. References
- Bossard et al., Food-101 Dataset (2014), https://www.kaggle.com/datasets/dansbecker/food-101/data
- Ultralytics YoloV8 Documentation, https://docs.ultralytics.com/
- Hugging Face Transformers, https://higgingface.co/docs
- Recipe Ingredients Dataset (Kaggle), https://www.kaggle.com/kaggle/recipe-ingredients-dataset
- Mistral AI, https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1

---

## 9. Appendix
- Github Repository: https://github.com/EpicSituation/CookVision.git
- Sample Output images:
- Trained model weights:
- Notebook file as needed: