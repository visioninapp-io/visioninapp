# FastAPI Backend

## Setup

1. Create virtual environment:
```bash
python -m venv venv
```

2. Activate virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file:
```bash
cp .env.example .env
```

5. Run the application:
```bash
uvicorn main:app --reload
```

## API Documentation

After running the application, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
BE/
├── app/
│   ├── api/
│   │   └── v1/          # API endpoints
│   ├── core/            # Core configuration
│   ├── models/          # Database models
│   ├── schemas/         # Pydantic schemas
│   ├── services/        # Business logic
│   └── utils/           # Utility functions
├── main.py              # Application entry point
├── requirements.txt     # Dependencies
└── .env                 # Environment variables
```
