# Simple RAG API

Простой API для сохранения документов и ответов на вопросы с использованием RAG (Retrieval-Augmented Generation) и Google Gemini AI.

## 🚀 Особенности

- **Простая архитектура** - все в одном файле `api.py`
- **Google Gemini AI** - для эмбеддингов и генерации ответов
- **Chroma DB** - векторная база данных для хранения документов
- **FastAPI** - современный веб-фреймворк
- **Автоматическое разбиение** документов на чанки
- **Фильтрация** по метаданным документов

## 📁 Структура проекта

```
rag-study-main/
├── api.py              # Основной API файл
├── requirements.txt    # Зависимости Python
├── .env               # Конфигурация (создать самостоятельно)
├── chroma_db/         # База данных Chroma
└── README.md          # Этот файл
```

## ⚙️ Установка

### 1. Клонирование и настройка

```bash
git clone <repository-url>
cd rag-study-main
```

### 2. Создание виртуального окружения

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# или
.venv\Scripts\activate     # Windows
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Создание .env файла

Создайте файл `.env` в корне проекта:

```bash
# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Server Configuration
HOST=0.0.0.0
PORT=8000

# Database Configuration
VECTOR_DB_DIR=./chroma_db
COLLECTION_NAME=documents

# Model Configuration
EMBEDDING_MODEL=models/embedding-001
LLM_MODEL=gemini-1.5-flash

# Chunking Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## 🚀 Запуск

```bash
source .venv/bin/activate  # Активируйте виртуальное окружение
python api.py
```

API будет доступен по адресу: `http://localhost:8000`

## 📖 API Endpoints

### 1. Сохранение документа

```bash
POST /save-document
```

**Request Body:**

```json
{
  "content": "Текст документа...",
  "doc_id": "unique_id",
  "name": "Название документа",
  "organization": "Организация"
}
```

**Response:**

```json
{
  "doc_id": "unique_id",
  "name": "Название документа",
  "content": "Первые 200 символов...",
  "organization": "Организация"
}
```

### 2. Задать вопрос

```bash
POST /ask-question
```

**Request Body:**

```json
{
  "question": "Ваш вопрос?",
  "doc_id": "unique_id",        # Опционально - фильтр по ID
  "name": "название",           # Опционально - фильтр по названию
  "organization": "организация" # Опционально - фильтр по организации
}
```

**Response:**

```json
{
  "question": "Ваш вопрос?",
  "answer": "Ответ на основе документов",
  "documents_used": [
    {
      "doc_id": "unique_id",
      "name": "Название документа",
      "organization": "Организация"
    }
  ]
}
```

## 🔧 Конфигурация

Все настройки берутся из переменных окружения в файле `.env`:

- **`GEMINI_API_KEY`** - API ключ Google Gemini
- **`HOST`** - хост сервера (по умолчанию 0.0.0.0)
- **`PORT`** - порт сервера (по умолчанию 8000)
- **`VECTOR_DB_DIR`** - путь к базе данных Chroma
- **`COLLECTION_NAME`** - название коллекции документов
- **`EMBEDDING_MODEL`** - модель для эмбеддингов
- **`LLM_MODEL`** - модель для генерации ответов
- **`CHUNK_SIZE`** - размер чанка документа
- **`CHUNK_OVERLAP`** - перекрытие между чанками

## 📚 Документация API

После запуска сервера документация доступна по адресу:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🎯 Примеры использования

### Сохранение документа

```bash
curl -X POST "http://localhost:8000/save-document" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "JavaScript был создан в 1995 году...",
    "doc_id": "js_history",
    "name": "История JavaScript",
    "organization": "Web Development Lab"
  }'
```

### Задать вопрос без фильтров

```bash
curl -X POST "http://localhost:8000/ask-question" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Когда был создан JavaScript?"
  }'
```

### Задать вопрос с фильтром

```bash
curl -X POST "http://localhost:8000/ask-question" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Что такое JavaScript?",
    "doc_id": "js_history"
  }'
```

## 🛠️ Технологии

- **FastAPI** - веб-фреймворк
- **LangChain** - фреймворк для RAG
- **Google Gemini AI** - AI модели
- **Chroma DB** - векторная база данных
- **Pydantic** - валидация данных
- **Uvicorn** - ASGI сервер

## 📝 Примечания

- Документы автоматически разбиваются на чанки для лучшего поиска
- Фильтрация работает по одному полю за раз (doc_id, name или organization)
- В ответе показываются только уникальные документы (без дублирования чанков)
- Все настройки конфигурируются через переменные окружения
