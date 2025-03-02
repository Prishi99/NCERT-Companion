# 📚 NCERT-Companion 📚

![ncert logo](https://github.com/user-attachments/assets/51b7d1cc-af1c-47a4-ac8c-41fbcb8bf040)


NCERT-Companion is an AI-powered learning tool designed to help students and educators explore NCERT subjects like Biology and Chemistry in an interactive and engaging way. It leverages advanced natural language processing (NLP) and machine learning techniques to provide instant answers to questions, offer relevant follow-up suggestions, and present knowledge sourced from official NCERT PDF textbooks.


![image](https://github.com/user-attachments/assets/c2996ec9-4ae6-411a-b294-32bb2f434ebf)
![image](https://github.com/user-attachments/assets/8d09970f-db26-4a37-b7fb-84e0eda9afdc)
![image](https://github.com/user-attachments/assets/9562de15-02cd-4d1f-8e35-c804bfb2b48b)



**Features**


✅ AI-Driven Answers: Get accurate and contextually relevant answers to your questions using OpenAI's GPT-3.5 architecture.

✅ PDF-Based Knowledge Base: Process and retrieve information directly from NCERT PDF textbooks, ensuring curriculum-aligned content.

✅ Streaming Responses: Watch answers generate in real-time, enhancing the interactive experience.

✅ Suggested Questions: Explore topics further with AI-generated follow-up questions that deepen understanding.

✅ Source Citations: At-a-glance source references help trace the origin of information.

✅ GPU Acceleration: Vector embeddings and similarity searches accelerated by GPU (when available) for faster performance.

**Prerequisites**

To run this application, you need:
Python 3.8+ installed.
An OpenAI API Key .


**Setup**

Clone this repository and follow the steps to get started:

```bash
git clone https://github.com/Prishi99/NCERT-Companion
cd NCERT-Companion
```

Install dependencies via pip:

```bash
pip install -r requirements.txt
```

Create a .env file to add your OpenAI API key:
OPENAI_API_KEY=your_openai_api_key_here

**Start Streamlit App**
```bash
streamlit run app.py
```

In the app interface:

--> Select a subject (Biology/Chemistry).
--> Click Pre-load Knowledge base
--> Ask any subject-related question.
--> Explore suggested follow-up questions.
--> Toggle GPU acceleration (if available) in the settings.


**Directory Structure**

```.
├── app.py                 # Main Streamlit application code
├── ncert_dataset/         # Directory for NCERT PDF textbooks
│   ├── Biology/           # Biology subject PDFs
│   ├── Chemistry/         # Chemistry subject PDFs
├── README.md              # This file
├── requirements.txt       # Python dependencies
└── .env.example           # Template for environment variables
```

**License**
This project is licensed under the MIT License
