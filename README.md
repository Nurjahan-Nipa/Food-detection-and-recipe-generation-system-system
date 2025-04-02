# CookVision: A Virtual AI-Powered Cooking Assistant for ingredient Detection and Recipe Gudiance Using Machine Learning

## 🔍 Abstract
**CookVision** is a virtual prototype of an AI-powered cooking assistant that uses machine learning and image detection to support the food preparation process. The system simulates a smart kitchen enviornment by using pre-recorded images and videos to detect ingredients, cooking tools, and food states. It integrates a pre-trained object detection model (YOLOv8) with a rule-based recipe recommendation engine to suggest context-aware recipts based on detected inputs.

The system is designed and tested entirely in a virtual space, enabling team members to develop and evaluate the core functionalities without physical hardware. A simple web-based interface displays detection results and guides users through recipe steps. This prototype serves as a foundation for future deployment on edge devices, demostrating the feasibility of combining computer vision and machine learning in the domain of intelligent cooking assistant.

## 🧩 Key Features
- Image-based ingredient and tool detection using YOLOv8
- Rule-based recipe suggestion engine
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