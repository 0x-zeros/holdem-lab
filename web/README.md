# Equity Calculator Web Application

A web-based Texas Hold'em equity calculator built with FastAPI and React.

## Architecture

```
web/
├── backend/          # FastAPI backend
│   ├── app/
│   │   ├── api/      # API routes
│   │   ├── schemas/  # Pydantic models
│   │   └── services/ # Business logic (wraps holdem_lab)
│   └── tests/
├── frontend/         # React frontend
│   ├── src/
│   │   ├── api/      # API client
│   │   ├── components/
│   │   └── store/    # Zustand state
│   └── ...
└── docker-compose.yml
```

## Quick Start

### Development (Local)

1. **Start Backend**
   ```bash
   cd web/backend
   uv pip install -e "../../holdem-core" -e ".[dev]"
   uv run uvicorn app.main:app --reload --port 8000
   ```

2. **Start Frontend**
   ```bash
   cd web/frontend
   npm install
   npm run dev
   ```

3. Open http://localhost:5173

### Development (Docker)

```bash
cd web
docker compose -f docker-compose.dev.yml up --build
```

### Production (Docker)

```bash
cd web
docker compose up --build
```

Access at http://localhost

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/canonical` | GET | Get 169 canonical hands |
| `/api/parse-cards` | POST | Parse card notation |
| `/api/equity` | POST | Calculate equity |
| `/api/draws` | POST | Analyze draws |

## Tech Stack

**Backend:**
- Python 3.12
- FastAPI
- Pydantic v2
- holdem-lab (core library)

**Frontend:**
- React 18
- TypeScript
- Tailwind CSS 4
- Zustand (state)
- TanStack Query (server state)

## Testing

```bash
# Backend tests
cd web/backend
uv run pytest

# Frontend type check
cd web/frontend
npm run typecheck
```
