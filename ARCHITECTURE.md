# DeepEval Lab — Architecture Overview

Короткий огляд того, **як влаштований проект** і **як його шари спілкуються**.
Без коду і деталей реалізації — лише ментальна карта.

---

## Що це за проект

Навчальна лабораторія, де на одному домені (іспанська нерухомість) показано
повний RAG-сценарій + оцінка відповідей через DeepEval. Кінцева мета — мати
єдине середовище, де можна:

- задавати моделям питання з прив'язаною базою знань,
- порівнювати моделі поряд (arena),
- ганяти метрики DeepEval по матриці моделей × суддів × сюїт,
- бачити результати в дашборді.

---

## Загальна карта

```
                    ┌───────────────────────────────────┐
                    │         FRONTEND (Next.js)        │
                    │   /chat     /arena     /eval      │
                    └────────────────┬──────────────────┘
                                     │  HTTP + SSE
                                     ▼
                    ┌───────────────────────────────────┐
                    │        BACKEND (FastAPI)          │
                    │ /chat  /arena  /models  /eval     │
                    └─────┬────────────┬────────────┬───┘
                          │            │            │
                  ┌───────▼───┐  ┌─────▼────┐  ┌────▼─────┐
                  │   RAG     │  │   LLM    │  │   EVAL   │
                  │  layer    │  │  layer   │  │  layer   │
                  └─────┬─────┘  └─────┬────┘  └────┬─────┘
                        │              │            │
                        ▼              ▼            ▼
                  ┌──────────┐   ┌──────────┐  ┌──────────┐
                  │ ChromaDB │   │  OpenAI  │  │  SQLite  │
                  │ (vectors)│   │ Anthropic│  │  (runs)  │
                  └────▲─────┘   │  Ollama  │  └──────────┘
                       │         └──────────┘
                       │ ingest
                  ┌────┴─────┐
                  │knowledge/│
                  │ curated  │
                  │   raw    │
                  └──────────┘
```

---

## Чотири шари

### 1. Knowledge (база знань)

Сировина для RAG. Лежить на диску у `knowledge/`:
- **`curated/`** — markdown статті, написані вручну (ITP, Golden Visa, NIE, …).
- **`raw/wikipedia/`** — статті з Wikipedia, скачані скриптом.
- **`raw/pdfs/`** — офіційні PDF з BOE (іспанські закони), скачуються і
  конвертуються в markdown.

Усі файли мають **однаковий формат**: YAML frontmatter (`title`, `topic`,
`lang`, `source`, `region` …) + markdown тіло. Один loader читає їх однаково.

Цикл: запускається `make ingest` → скрипти качають дані → loader збирає
все у `Document`-и → RAG-шар ріже їх на чанки і кладе у ChromaDB.

---

### 2. RAG (пошук за смислом)

`apps/backend/rag/`. Відповідає за: "по запиту користувача знайти топ-K
найрелевантніших шматків з бази".

Pipeline:
```
docs ─► chunker ─► embeddings ─► ChromaDB
                                      ▲
query ─► embeddings ──────────────────┘
                          │
                          ▼
                   top-K chunks (з фільтрами по metadata)
```

Ключові ідеї:
- **Чанки** ~800 токенів кожен, з overlap 100; розбиваються по markdown-
  заголовках, а не посеред речення.
- **Embeddings** — три варіанти: вбудований ChromaDB (offline, default),
  Ollama, OpenAI. Перемикання через `.env`.
- **ChromaDB** — embedded vector DB, файли на диску (`.chroma/`). Окрема
  колекція на кожен провайдер ембедингів, бо вектори різних моделей не
  можна змішувати.
- **Retriever** — фронтенд над ChromaDB: top-K + фільтри (region, topic,
  source, lang). Повертає список `RetrievedChunk` зі стабільним API.

---

### 3. LLM (адаптери до моделей)

`apps/backend/llm/`. Відповідає за: "відправити повідомлення моделі і
отримати відповідь — однаково для OpenAI / Anthropic / Ollama".

```
chat / arena / eval ─► LLMAdapter (один контракт)
                            │
              ┌─────────────┼──────────────┐
              ▼             ▼              ▼
         OpenAI SDK    Anthropic SDK    Ollama SDK
```

Ключові ідеї:
- **Спільний контракт** `LLMAdapter`: `chat()`, `stream()`, `list_models()`,
  `health()`. Усі провайдери виглядають однаково для решти проекту.
- **ModelRegistry** — синглтон, який знає які адаптери сконфігуровані
  (по env-vars) і кешує `health`+`list_models` на 30 секунд.
- **Async** — усі виклики async, стрімінг через async generators.
- **DeepEvalLLM** — обгортка, яка дозволяє використовувати наші адаптери
  як **суддів** в DeepEval. Тому в eval-matrix можна "оцінити GPT через
  Claude" чи "Claude через локальну Llama" — будь-яка пара провайдерів.

---

### 4. Backend (FastAPI)

`apps/backend/`. Склеює RAG + LLM + сесії + eval і експонує HTTP API.

Основні endpoint-и:
- **`GET /models`** — які провайдери/моделі доступні зараз.
- **`POST /chat`** — один turn у чаті; режим `stream=true` → SSE,
  `stream=false` → один JSON. Якщо `use_rag=true` — спочатку retriever,
  потім модель.
- **`POST /arena`** — паралельний прогон двох моделей на одному запиті,
  з однаковим контекстом.
- **`GET /eval/...`** — список eval-прогонів, деталі по конкретному прогону.

Внутрішні шматки:
- **session store** — in-memory dict сесій з історією повідомлень, TTL 2h.
- **DI** через `Depends(...)` — реєстр і retriever ін'єктяться у endpoint-и,
  тести легко підміняють їх фейками.
- **Конфіг** — `pydantic-settings`, читає `.env`, валідує типи.

---

### 5. Frontend (Next.js)

`apps/frontend/`. Тонкий клієнт над бекендом, три сторінки:

- **`/chat`** — чат-сторінка зі стрімінгом. Підписується на SSE, рендерить
  по delta-event-ах, з боковою панеллю джерел (`Retrieved sources`).
- **`/arena`** — дві моделі поряд на одному запиті, для людської оцінки.
- **`/eval`** — heat-map результатів eval-матриці (моделі × судді × метрики).

Спілкування з бекендом — звичайний `fetch` для JSON-endpoint-ів і
`EventSource`-style SSE-reader для `/chat?stream=true`.

---

### 6. Eval (DeepEval-сюїти + matrix runner)

`apps/backend/eval/` + `scripts/run_eval_matrix.py`. Відповідає за: "запусти
сюїту метрик на наборі тестових кейсів і збережи результат".

Концепція:
- **SUT** (System Under Test) — модель, яку оцінюємо.
- **Judge** — модель, яка ставить оцінку (через DeepEval LLM-as-judge
  метрики типу `G-Eval`, `AnswerRelevancy`).
- **Suite** — набір метрик у певній категорії (single-turn, conversational,
  rag, agentic, safety, multimodal, custom).
- **Matrix runner** — декартів добуток SUT × Judge × Suite, прогін, запис
  результату в SQLite.
- **CI** — на push у main гониться `smoke`-сюїта, gate на регресії.

---

## Як один chat-запит проходить систему

```
1. Користувач набирає "What is the ITP rate in Madrid?" і клікає Send.
       │
       ▼
2. Frontend: POST /chat зі stream=true, провайдером і моделлю.
       │
       ▼
3. Backend (FastAPI):
       ├─ перевіряє адаптер у ModelRegistry
       ├─ дістає сесію зі store
       ├─ retriever.retrieve(question, filters)  ──► RAG-шар
       │      └─► ChromaDB top-K чанків з metadata
       ├─ будує messages: [system, context, ...history, user]
       └─ повертає StreamingResponse (SSE).
       │
       ▼
4. SSE-стрім frontend-у:
       event: session  (session_id)
       event: context  (retrieved chunks → sources panel)
       event: delta    (× багато)  ──► накопичується у текст відповіді
       event: done     (latency, model)
       │
       ▼
5. Frontend оновлює UI у реальному часі; сесія зберігається, наступний
   запит шле той самий session_id.
```

---

## Як один eval-прогон проходить систему

```
1. Користувач (або CI) запускає run_eval_matrix.
       │
       ▼
2. Matrix runner для кожної комбінації SUT × Judge × Suite:
       │
       ├─ створює DeepEvalLLM(adapter=Judge)
       ├─ для кожного кейсу в сюїті:
       │     ├─ викликає answer(SUT, question, use_rag=True)
       │     │     ├─► RAG retrieve   (так само як у чаті)
       │     │     └─► SUT.chat(...)  (адаптер цільової моделі)
       │     ├─ передає (input, actual_output, retrieval_context) у метрики
       │     └─ метрики DeepEval кличуть Judge.generate(prompt) для оцінки
       │
       └─ записує summary у SQLite (eval_results/).
       │
       ▼
3. Frontend /eval читає SQLite через /eval API → рендерить heat-map.
```

---

## Як це все запускається

| Команда | Що робить |
|---|---|
| `make ingest` | Підкачує Wikipedia + PDF, ріже на чанки, кладе в ChromaDB |
| `make dev` | Стартує FastAPI на :8000 |
| `cd apps/frontend && npm run dev` | Стартує Next.js на :3000 |
| `make test-smoke` | Швидкі дешеві тести (gpt-4o-mini як SUT+judge) |
| `make eval-matrix` | Повна матриця локально |

Решта в `Makefile` і в `README.md`.

---

## Якщо одним абзацом

**Frontend** малює UI і шле HTTP. **Backend** на FastAPI приймає запит,
іде в **RAG-шар** за контекстом з ChromaDB, потім кличе одну з моделей
через **LLM-шар** і стрімить відповідь назад. Той самий LLM-шар
переюзується **eval-шаром** як SUT і як judge — тому одна й та сама
модель може і відповідати в чаті, і оцінювати інші моделі в матриці.
