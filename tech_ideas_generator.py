"""
Генератор технических описаний для AI-проектов

Этот скрипт реализует сравнительное исследование трёх LLM-моделей 
на задаче генерации технических описаний для AI-проектов.
"""

# =================================
# 1. Импорт библиотек и конфигурация
# =================================

import requests
import json
import pandas as pd
from typing import Dict, List, Tuple
import time
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

# Конфигурация API OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Базовый URL и заголовки для OpenRouter API
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/singl3focus/gen_desctibe_tech_ideas",
    "X-Title": "Tech Ideas Generator"
}

# Три модели для сравнения (использую бесплатные версии)
MODELS = [
    "qwen/qwen3-30b-a3b:free",
    "google/gemma-3-27b-it:free",
    "deepseek/deepseek-r1:free"
]

# Параметры генерации (фиксированы для всех моделей для честного сравнения)
GENERATION_PARAMS = {
    "temperature": 0.7,
    "max_tokens": 2000,
    "top_p": 0.9
}

print("✓ Библиотеки импортированы")
print(f"✓ Количество моделей для тестирования: {len(MODELS)}")

# =================================
# 2. Генерация идей AI-проектов
# =================================

# Создаём набор из 10 тестовых идей AI-проектов различной сложности и направленности.

# Набор тестовых идей AI-проектов (8-10 штук)
AI_PROJECT_IDEAS = [
    "Система автоматического обнаружения лесных пожаров с помощью дронов и компьютерного зрения. Дроны патрулируют лесные массивы, анализируют видеопоток в реальном времени и отправляют сигнал тревоги при обнаружении дыма или огня.",
    
    "Умный помощник для медицинской диагностики на основе анализа симптомов и истории болезни. Система собирает данные о симптомах пациента, сравнивает с базой знаний и предлагает возможные диагнозы для консультации с врачом.",
    
    "Персонализированная рекомендательная система для онлайн-образования. Анализирует прогресс студента, стиль обучения и текущие знания, чтобы предлагать оптимальные курсы и учебные материалы.",
    
    "Система автоматического саммаризации и анализа тональности отзывов клиентов. Обрабатывает тысячи отзывов, выделяет ключевые темы, определяет настроение клиентов и генерирует краткие аналитические отчёты для менеджеров.",
    
    "Чат-бот для технической поддержки с возможностью обучения на базе знаний компании. Отвечает на вопросы клиентов, решает типовые проблемы и передаёт сложные случаи живым операторам.",
    
    "Система прогнозирования спроса для розничной торговли на основе исторических данных продаж и внешних факторов. Учитывает сезонность, погоду, праздники и другие факторы для оптимизации закупок и складских запасов.",
    
    "Приложение для распознавания и классификации растений по фотографиям с телефона. Пользователь делает снимок растения, а система определяет вид, даёт рекомендации по уходу и предупреждает о ядовитости.",
    
    "Система мониторинга качества производства на основе компьютерного зрения. Анализирует изделия на конвейере в реальном времени, выявляет дефекты и отклонения от стандартов качества.",
    
    "Голосовой ассистент для управления умным домом с поддержкой естественного языка. Понимает команды на русском языке, управляет освещением, климатом, бытовой техникой и учится предпочтениям пользователя.",
    
    "Система автоматического перевода технической документации с сохранением терминологии и форматирования. Переводит руководства, спецификации и инструкции, сохраняя структуру документа и специфические термины."
]

print(f"\n✓ Создано {len(AI_PROJECT_IDEAS)} идей AI-проектов")
print("\nПримеры идей:")
for i, idea in enumerate(AI_PROJECT_IDEAS[:3], 1):
    print(f"{i}. {idea[:100]}...")

# =================================
# 3. Шаблон промпта для моделей
# =================================

# Единый промпт для всех трёх моделей обеспечивает честное сравнение.

# Шаблон промпта (одинаковый для всех моделей)
PROMPT_TEMPLATE = """Ты — технический эксперт в области AI и машинного обучения. 

Дана следующая идея AI-проекта:
{idea}

Пожалуйста, предоставь детальный технический анализ этого проекта в следующем формате:

1. ТЕХНИЧЕСКОЕ ОПИСАНИЕ:
[Развёрнутое техническое описание решения на 3-5 предложений]

2. ТЕХНОЛОГИИ И БИБЛИОТЕКИ:
[Конкретный список фреймворков, библиотек, баз данных и других технологий]

3. ЭТАПЫ РЕАЛИЗАЦИИ:
[3-5 основных этапов разработки проекта]

4. ОЦЕНКА СЛОЖНОСТИ:
[Одно слово: Легко / Средне / Сложно]

Будь конкретным и практичным в своих рекомендациях."""

print("\n✓ Шаблон промпта создан")
print("\nПример промпта:")
print(PROMPT_TEMPLATE.format(idea="[Пример идеи проекта]")[:300] + "...")

# =================================
# 4. Функции работы с OpenRouter API
# =================================

def call_openrouter_model(model_name: str, prompt: str, max_retries: int = 3) -> str:
    """
    Вызывает конкретную модель через OpenRouter API.
    
    Args:
        model_name: Название модели (например, "qwen/qwen3-30b-a3b:free")
        prompt: Текст промпта для модели
        max_retries: Максимальное количество попыток при ошибке
    
    Returns:
        Текст ответа модели или сообщение об ошибке
    """
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        **GENERATION_PARAMS
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                OPENROUTER_BASE_URL,
                headers=HEADERS,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                return f"[ОШИБКА: {str(e)}]"
            time.sleep(2 ** attempt)
        except (KeyError, IndexError) as e:
            return f"[ОШИБКА ПАРСИНГА ОТВЕТА: {str(e)}]"
    
    return "[ОШИБКА: Превышено количество попыток]"


def parse_model_response(response: str) -> Dict[str, str]:
    """
    Парсит ответ модели и извлекает структурированную информацию.
    
    Args:
        response: Текст ответа от модели
    
    Returns:
        Словарь с извлечёнными секциями
    """
    sections = {
        'technical_description': '',
        'tech_stack': '',
        'implementation_steps': '',
        'difficulty': ''
    }
    
    lines = response.split('\n')
    current_section = None
    
    for line in lines:
        line_upper = line.upper()
        
        if 'ТЕХНИЧЕСКОЕ ОПИСАНИЕ' in line_upper or 'TECHNICAL DESCRIPTION' in line_upper:
            current_section = 'technical_description'
        elif 'ТЕХНОЛОГИ' in line_upper or ('TECH' in line_upper and 'STACK' in line_upper):
            current_section = 'tech_stack'
        elif 'ЭТАП' in line_upper or 'IMPLEMENTATION' in line_upper or 'STEPS' in line_upper:
            current_section = 'implementation_steps'
        elif 'СЛОЖНОСТ' in line_upper or 'DIFFICULTY' in line_upper:
            current_section = 'difficulty'
        elif current_section and line.strip():
            sections[current_section] += line + '\n'
    
    for key in sections:
        sections[key] = sections[key].strip()
    
    return sections


print("\n✓ Функции для работы с API созданы")

# =================================
# 5. Основной эксперимент: сбор данных
# =================================

# Запрашиваем все три модели для каждой идеи и собираем результаты.

print("\n✓ Начинаем основной эксперимент по сбору данных")

# Структура для хранения результатов
results = []

# Общее количество вызовов API
total_calls = len(AI_PROJECT_IDEAS) * len(MODELS)

print(f"Начинаем эксперимент: {len(AI_PROJECT_IDEAS)} идей × {len(MODELS)} моделей = {total_calls} вызовов API")
print("Это может занять несколько минут...\n")

# Основной цикл эксперимента
for idea_idx, idea in enumerate(tqdm(AI_PROJECT_IDEAS, desc="Обработка идей"), 1):
    prompt = PROMPT_TEMPLATE.format(idea=idea)
    
    for model in MODELS:
        # Запрос к модели
        response = call_openrouter_model(model, prompt)
        
        # Парсинг ответа
        parsed = parse_model_response(response)
        
        # Сохранение результата
        results.append({
            'idea_id': idea_idx,
            'idea': idea,
            'model': model,
            'raw_response': response,
            'technical_description': parsed['technical_description'],
            'tech_stack': parsed['tech_stack'],
            'implementation_steps': parsed['implementation_steps'],
            'difficulty': parsed['difficulty']
        })
        
        # Небольшая задержка между запросами
        time.sleep(1)

# Создание DataFrame
df_results = pd.DataFrame(results)

print(f"\n✓ Эксперимент завершён!")
print(f"✓ Собрано {len(df_results)} результатов")
print(f"\nПервые несколько записей:")
print(df_results[['idea_id', 'model', 'difficulty']].head(10))

# =================================
# 6. Просмотр примеров ответов
# =================================

# Выводим несколько примеров полных ответов для анализа
sample_idea_id = 1

print(f"Сравнение ответов трёх моделей для идеи #{sample_idea_id}:")
print(f"\nИдея: {AI_PROJECT_IDEAS[sample_idea_id - 1]}")
print("\n" + "="*80 + "\n")

for model in MODELS:
    model_data = df_results[(df_results['idea_id'] == sample_idea_id) & (df_results['model'] == model)].iloc[0]
    
    print(f"МОДЕЛЬ: {model}")
    print("-" * 80)
    print(f"Техническое описание:\n{model_data['technical_description'][:300]}...\n")
    print(f"Технологии:\n{model_data['tech_stack'][:200]}...\n")
    print(f"Сложность: {model_data['difficulty']}")
    print("\n" + "="*80 + "\n")

# =================================
# 7. Анализ и оценка качества моделей
# =================================

# Оцениваем каждую модель по трём критериям.

def evaluate_response(row: pd.Series) -> Tuple[str, str, str, str]:
    """
    Эвристическая оценка качества ответа модели.
    
    Критерии:
    - technical_quality: наличие конкретных технологий и технических деталей
    - completeness: полнота всех секций ответа
    - practicality: практичность и реалистичность рекомендаций
    """
    # Техническая грамотность
    tech_keywords = ['python', 'pytorch', 'tensorflow', 'fastapi', 'flask', 'django', 
                     'postgresql', 'mongodb', 'docker', 'kubernetes', 'opencv', 'yolo',
                     'bert', 'transformer', 'api', 'rest', 'websocket', 'nginx']
    
    tech_stack_lower = row['tech_stack'].lower()
    tech_count = sum(1 for kw in tech_keywords if kw in tech_stack_lower)
    
    if tech_count >= 5:
        technical_quality = "Высокая"
    elif tech_count >= 3:
        technical_quality = "Средняя"
    else:
        technical_quality = "Низкая"
    
    # Полнота описания
    desc_len = len(row['technical_description'])
    stack_len = len(row['tech_stack'])
    steps_len = len(row['implementation_steps'])
    total_length = desc_len + stack_len + steps_len
    
    if total_length > 800 and desc_len > 200 and stack_len > 100 and steps_len > 200:
        completeness = "Высокая"
    elif total_length > 400:
        completeness = "Средняя"
    else:
        completeness = "Низкая"
    
    # Практичность
    steps_lines = [line.strip() for line in row['implementation_steps'].split('\n') if line.strip()]
    has_numbers = any(c.isdigit() for c in row['implementation_steps'][:50])
    
    if len(steps_lines) >= 3 and has_numbers:
        practicality = "Высокая"
    elif len(steps_lines) >= 2:
        practicality = "Средняя"
    else:
        practicality = "Низкая"
    
    comment = f"Упомянуто {tech_count} технологий, длина ответа {total_length} символов, {len(steps_lines)} этапов"
    
    return technical_quality, completeness, practicality, comment


print("Оцениваем качество ответов...\n")

evaluations = df_results.apply(evaluate_response, axis=1, result_type='expand')
evaluations.columns = ['technical_quality', 'completeness', 'practicality', 'comment']

df_results = pd.concat([df_results, evaluations], axis=1)

print("✓ Оценка завершена")
print("\nПример оценок:")
print(df_results[['model', 'technical_quality', 'completeness', 'practicality']].head(9))

# =================================
# 8. Сводная таблица сравнения моделей
# =================================

def quality_to_score(quality: str) -> int:
    mapping = {'Высокая': 3, 'Средняя': 2, 'Низкая': 1}
    return mapping.get(quality, 0)

df_results['tech_score'] = df_results['technical_quality'].apply(quality_to_score)
df_results['completeness_score'] = df_results['completeness'].apply(quality_to_score)
df_results['practicality_score'] = df_results['practicality'].apply(quality_to_score)

model_comparison = df_results.groupby('model').agg({
    'tech_score': 'mean',
    'completeness_score': 'mean',
    'practicality_score': 'mean'
}).round(2)

def score_to_quality(score: float) -> str:
    if score >= 2.5:
        return f"Высокая ({score:.2f})"
    elif score >= 1.5:
        return f"Средняя ({score:.2f})"
    else:
        return f"Низкая ({score:.2f})"

model_comparison['Техническая грамотность'] = model_comparison['tech_score'].apply(score_to_quality)
model_comparison['Полнота описания'] = model_comparison['completeness_score'].apply(score_to_quality)
model_comparison['Практичность'] = model_comparison['practicality_score'].apply(score_to_quality)

model_comparison['Общий балл'] = (model_comparison['tech_score'] + 
                                   model_comparison['completeness_score'] + 
                                   model_comparison['practicality_score']).round(2)

final_comparison = model_comparison[['Техническая грамотность', 'Полнота описания', 
                                     'Практичность', 'Общий балл']]

print("СВОДНАЯ ТАБЛИЦА СРАВНЕНИЯ МОДЕЛЕЙ")
print("="*100)
print(final_comparison.to_string())
print("\n✓ Анализ завершён")

# =================================
# 9. Экспорт результатов для отчёта
# =================================

# Сохраняем результаты в CSV для дальнейшего анализа
df_results.to_csv('experiment_results.csv', index=False, encoding='utf-8-sig')
final_comparison.to_csv('model_comparison.csv', encoding='utf-8-sig')

print("✓ Результаты сохранены в файлы:")
print("  - experiment_results.csv")
print("  - model_comparison.csv")

print("\n\n✓ Эксперимент полностью завершён!")
