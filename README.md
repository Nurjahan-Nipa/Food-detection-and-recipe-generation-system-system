# CookVision: A Virtual AI-Powered Cooking Assistant for ingredient Detection and Recipe Gudiance Using Machine Learning

## 🔍 Abstract
**CookVision** is a virtual prototype of an AI-powered cooking assistant that uses machine learning and image detection to support the food preparation process. The system simulates a smart kitchen environment by using pre-recorded images and videos to detect ingredients, cooking tools, and food states. It integrates a pre-trained object detection model (YOLOv8) with a rule-based recipe recommendation engine to suggest context-aware recipts based on detected inputs.

The system is designed and tested entirely in a virtual space, enabling team members to develop and evaluate the core functionalities without physical hardware. A simple web-based interface displays detection results and guides users through recipe steps. This prototype serves as a foundation for future deployment on edge devices, demostrating the feasibility of combining computer vision and machine learning in the domain of intelligent cooking assistant.

## Dataset to be used
### Food Image Detection
Image to help train or test the object detection system (e.g, identifying tomatoes, onions, etc.):
- [Food-101](https://www.kaggle.com/datasets/dansbecker/food-101/data)
- [UECFOOD-256](https://www.kaggle.com/datasets/rkuo2000/uecfood256)
- YOLOV8 (Optionally train our model ourselves as needed)

### Cooking Activity & Steps Detection
- [EPIC-KITCHENS Dataset](https://github.com/epic-kitchens)
- [YouCook2](https://huggingface.co/datasets/morpheushoc/youcook2)

### Recipe and Ingredient Matching.
- [Kaggle's Recipe Ingredients Dataset](https://www.kaggle.com/datasets/kaggle/recipe-ingredients-dataset)
- Create our own clean, curated dataset - small, understandable, and grouped by couisine

## 🧩 Key Features
- Image-based ingredient and tool detection using YOLOv8
- AI-suggested recipes according to selected cuisine
- Simulated video/image pipeline in place of physical sensors
- Lightweight web UI for interaction and visualization
- Modular architecture to allow team collaboration and extension

## 🧪 Technology Used
- Python 3
- YOLOv8 (Ultralytics)
- OpenCV
- Streamlit or Flask (for UI)
- JSON / CSV (for recipe storage)
- Github & Notion (for collaboration)
- e.t.c.

## 👥 Team & Roles

| Name          | Responsibility                        |
|---------------|---------------------------------------|
| Member A      | Object Detection & Preprocessing      |
| Member B      | Recipe Engine & Ingredient Mapping    |
| Member C      | Cooking Step Detection                |
| Member D      | Api & Backend Development             |
| Member E      | UI Development                        |
| Member F      | Testing & Integration                 |
| Member G      | Documentation & Presentation          |

## 📂 Project Structure
```planttext
Cookvision/
|--- dectecton/         # YOLOv8 script, image processing
|--- recepe_engine/     # Ingredient-to-recepi logic and suggestion algorithms
|--- ui/                #Streamlit or frontend interface code
|--- api/               #Backend services using Flask or FastAPI
|--- data/              #Test images, video samples, recipe datasets
|___ README.md          #Project documenentation
```
