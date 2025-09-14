import gradio as gr
from PIL import Image
import numpy as np
import cv2
import pytesseract
import requests
import json

# ===== Конфигурация OpenRouter =====
OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Функция распознавания текста и извлечения полей
# OCR + GPT обработка
def ocr_extract_all_fields(image, lang_choice="Все вместе"):
    # Преобразуем PIL -> OpenCV
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # OCR через Tesseract
    lang_options = {
        "Казахский": "kaz",
        "Русский": "rus",
        "Английский": "eng",
        "Все вместе": "kaz+rus+eng"
    }
    lang_code = lang_options.get(lang_choice, "kaz+rus+eng")
    ocr_text = pytesseract.image_to_string(thresh, lang=lang_code)

    # Формируем prompt для GPT: просим вернуть все найденные значения
    prompt = (
        f"Извлеки все значения следующих полей из документа и верни JSON:\n"
        f"Поля: ФИО, Дата, Номер договора, Сумма, ИНН, Адрес.\n"
        f"Если поле встречается несколько раз, укажи все значения в виде списка.\n\n"
        f"Документ:\n{ocr_text}"
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
        result_text = response.json()['choices'][0]['message']['content']
        try:
            result_json = json.loads(result_text)
        except json.JSONDecodeError:
            result_json = {"raw_text": result_text}

    except Exception as e:
        result_json = {"error": str(e)}

    return ocr_text, result_json


# Gradio интерфейс
with gr.Blocks() as demo:
    gr.Markdown("## OCR + OpenRouter GPT: извлечение всех ключевых полей")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Загрузите изображение", type="pil")
            lang_choice = gr.Radio(
                choices=["Казахский", "Русский", "Английский", "Все вместе"],
                value="Все вместе",
                label="Язык OCR"
            )
            run_btn = gr.Button("Распознать и извлечь поля")
        with gr.Column():
            ocr_output = gr.Textbox(label="Текст OCR", lines=20, max_lines=100)
            json_output = gr.JSON(label="Извлечённые поля GPT")

    run_btn.click(
        fn=ocr_extract_all_fields,
        inputs=[input_image, lang_choice],
        outputs=[ocr_output, json_output]
    )

# Запуск
demo.launch(inbrowser=True, share=False)