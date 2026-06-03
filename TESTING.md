# Testing & Evaluation

Як запускати тести і що саме вони перевіряють — по категоріях.

---

## TL;DR — найчастіші команди

| Команда | Що робить | Витрати |
|---|---|---|
| `make test-smoke` | Швидкі дешеві тести (= що ганяє CI) | ~$0.05 за прогон |
| `make test-local` | Усі smoke + heavy тести (повна якість, локально) | помітно дорожче |
| `pytest tests/rag/ -v` | Тільки інфраструктурні тести RAG (без LLM) | $0 |
| `pytest tests/llm/ -v` | Тільки тести базових структур LLM (без HTTP) | $0 |
| `pytest tests/api/ -v` | Тести FastAPI endpoint-ів (з фейковими LLM) | $0 |
| `pytest -m "smoke and rag" -v` | Один зріз по маркерах | залежить |
| `pytest tests/eval/safety/test_positive.py -v` | Один сюїт-файл явно | залежить |
| `pytest tests/eval/rag/test_negative.py -v` | Лише negative-контролі однієї категорії | дешевше (judge-only) |
| `pytest tests/eval -k negative -v` | Усі negative-контролі (всі категорії) | дешевше (judge-only) |

Вимоги:
- venv активований, залежності встановлені (`pip install -e ".[llm,rag,eval]"`).
- ChromaDB-індекс зібраний: `make ingest` (один раз).
- `OPENAI_API_KEY` у `.env` — інакше всі eval-тести скіпаються через `_require_openai_key()`.

---

## Що означає кожен маркер

Маркери оголошені в `pyproject.toml` під `[tool.pytest.ini_options]` і клеяться через `@pytest.mark.<name>`.

| Маркер | Сенс |
|---|---|
| `smoke` | Дешеві швидкі тести; гарантовані CI-гейтом на push в main |
| `local` | Heavy/full версії тих самих сюїт; ганяти руками |
| `rag` | Тести RAG-метрик (retrieve→generate→grade loop) |
| `conversational` | Multi-turn діалоги |
| `agentic` | Tool-using agents |
| `safety` | Red-team probes |
| `multimodal` | Тести з зображеннями (зарезервовано) |
| `custom` | GEval / DAG / Arena |

Smoke-гейт ще обрізає кількість параметризованих кейсів через `SMOKE_MAX_CASES` (default 50), щоб датасет, що розрісся, не з'їв весь бюджет. Реалізовано у `tests/conftest.py` через `pytest_collection_modifyitems`.

---

## Дві сім'ї тестів

```
tests/
├── (infrastructure — швидкі, без LLM)
│   ├── test_knowledge_loader.py    # parse markdown + frontmatter → Document
│   ├── llm/test_base_and_registry.py  # DTO, ProviderStatus, fallback логіка
│   ├── rag/test_pipeline.py        # chunker, embeddings, ChromaStore round-trip
│   └── api/test_endpoints.py       # FastAPI endpoints з підміненими залежностями
│
└── eval/                           # DeepEval-сюїти — LLM-as-judge
    ├── conftest.py                 # judge + chat_adapter фікстури
    ├── single_turn/
    │   ├── test_positive.py        # метрики на «хорошому» виводі SUT
    │   └── test_negative.py        # negative-контролі: метрика МУСИТЬ зловити поганий вивід
    ├── rag/
    │   ├── test_positive.py
    │   └── test_negative.py
    ├── conversational/
    │   ├── test_positive.py
    │   └── test_negative.py
    ├── agentic/
    │   ├── test_positive.py
    │   └── test_negative.py
    ├── custom/
    │   ├── test_positive.py
    │   └── test_negative.py
    └── safety/
        └── test_positive.py        # адверсаріальні проби (вже «негативні» за суттю)
```

**positive vs negative.** У кожній категорії `test_positive.py` перевіряє, що наш чат-бот поводиться добре (метрика проходить). `test_negative.py` — це **negative-контролі для самих метрик**: ми хардкодимо явно поганий вивід (упереджений / токсичний / вигаданий / не за форматом) і стверджуємо, що метрика його **ловить** (`not m.is_successful()`). Зелений negative-прогон = «у наших гардів є зуби». Negative-тести не викликають SUT (вивід захардкоджено) → лише judge → дешевші.

**Infrastructure-тести** — звичайні pytest-assertion-и, ніяких LLM-викликів. Перевіряють, що шари коду коректно з'єднані: парсинг, чанкінг, ембедингова батч-обробка, top-K по фейковому корпусу, FastAPI 200/404 з підміненим Retriever-ом тощо. Цю частину можна ганяти безкоштовно на кожному коміті.

**Eval-тести** — це й є DeepEval. Реальний LLM відповідає на питання (SUT), реальний LLM-суддя ставить оцінку. Завжди коштують грошей.

---

## DeepEval-сюїти по категоріях

Усі eval-тести використовують спільні фікстури з `tests/eval/conftest.py`:

- **`chat_adapter`** — `OpenAIAdapter()` (модель з env `SUT_MODEL`, default `gpt-4o-mini`).
- **`judge`** — той самий адаптер, обгорнутий у `DeepEvalLLM` з `temperature=0.0` (env `JUDGE_MODEL`).
- **`sut_model`** — рядок назви моделі для SUT.

Це й є концепт **SUT (System Under Test)** vs **Judge** в DeepEval: одна модель пише відповідь, інша її оцінює. Тут default — обидві `gpt-4o-mini`, але matrix runner (`scripts/run_eval_matrix.py`) робить декартів добуток усіх SUT × judge.

---

### 1. Single-turn metrics — `tests/eval/single_turn/`

Одна пара "user→assistant", без історії та (зазвичай) без RAG.

| Метрика | Що перевіряє | Як ганяє |
|---|---|---|
| **HallucinationMetric** | Чи відповідь узгоджена з даним наземним контекстом (inverse of Faithfulness) | Передаємо handcrafted `context=[...]`, питання, отриману відповідь; judge каже чи відповідь не вигадує |
| **BiasMetric** | Чи є упереджені/стереотипні твердження про людей/регіони | Питаємо "describe buyer in Marbella", judge шукає bias |
| **ToxicityMetric** | Чи є ворожа/токсична мова | Той самий вхід, інший judge-промпт |
| **PromptAlignmentMetric** | Чи модель послухалась явну інструкцію формату | "Explain in exactly two sentences..." — judge перевіряє |
| **JsonCorrectnessMetric** | Чи вихід — валідний JSON, який матчить Pydantic-schema | Просимо JSON з полями `region/itp_percent/notes`; metric парсить і валідує |

Smoke-пороги навмисно нижчі за "якість" — мета smoke це wiring/regression gate. Строгі пороги — у `local`-варіантах.

---

### 2. RAG metrics — `tests/eval/rag/`

Найбагатша сюїта. Прогоняє **повний** retrieve→generate→grade цикл на наборі "goldens" (тестових питань з очікуваною відповіддю), визначених у `knowledge/synth/goldens.json`.

Для кожного golden:
1. Викликає `service.answer()` зі справжнім RAG (ChromaDB, top-K=4).
2. Створює `LLMTestCase(input, actual_output, expected_output, retrieval_context)`.
3. Прогоняє п'ять RAG-метрик.

| Метрика | Що перевіряє |
|---|---|
| **FaithfulnessMetric** | Чи відповідь **ґрунтується** на retrieved-чанках (без галюцинацій) |
| **AnswerRelevancyMetric** | Чи відповідь відповідає **на питання** (не off-topic) |
| **ContextualRelevancyMetric** | Чи **retrieved-чанки** були релевантні до питання (якість retriever-а) |
| **ContextualRecallMetric** | Чи в retrieved-чанках **достатньо інформації**, щоб відповісти |
| **ContextualPrecisionMetric** | Чи **найрелевантніший** чанк ранжований першим |

Smoke vs local: smoke бере 6 goldens (з `smoke_only()`), local — всі 12. Smoke-пороги м'якіші:
- Faithfulness 0.6 (full 0.8), AnswerRelevancy 0.5 → 0.7, ContextualRelevancy 0.0 → 0.2.

Усі п'ять метрик ділять одну SUT-відповідь на golden через fixture `case_for` з кешем — платимо за генерацію один раз.

---

### 3. Conversational metrics — `tests/eval/conversational/`

Розкочує **4-турний діалог** UK-покупця про резaleл-квартиру в Мадриді (Madrid → ITP rate → residency → STR licence). Кожен наступний turn отримує `history=` з попередніх.

Збирає список `Turn`-ів і робить `ConversationalTestCase(scenario, chatbot_role, expected_outcome)`.

| Метрика | Що перевіряє |
|---|---|
| **KnowledgeRetentionMetric** | Чи асистент пам'ятає факти з ранніх turn-ів ("Madrid", "UK citizen") у пізніх |
| **ConversationCompletenessMetric** | Чи асистент покрив усі цілі сценарію (tax kind + rate + residency + licence) |
| **RoleAdherenceMetric** | Чи асистент тримався заявленої ролі ("Spanish real estate advisor") |

---

### 4. Agentic metrics — `tests/eval/agentic/`

Для tool-using agents. **Без** реального agent-loop — handcrafted `tools_called` лісти на `LLMTestCase`, щоб демонструвати метрики дешево і детерміністично.

Тестова задача: "знайди квартиру в Madrid <€400k і порахуй іпотеку при 70% LTV на 25 років" (правильний шлях: 2 послідовні tool calls).

| Метрика | Що перевіряє |
|---|---|
| **TaskCompletionMetric** | Чи агент завершив задачу користувача (з огляду на викликані tools) |
| **ToolCorrectnessMetric** | Чи викликані саме потрібні tools (детермінований match без LLM) |
| **ArgumentCorrectnessMetric** | Чи аргументи в tool-викликах добре сформовані |

Окремий negative case (`test_task_completion_bad_path_is_flagged`): свідомо неправильні tools + ухильна відповідь → asserts that **metric fails**.

---

### 5. Safety metrics — `tests/eval/safety/`

Smoke-вибірка red-team пробів, дешева для CI. Повне red-teaming через `deepeval.red_teaming.RedTeamer` — лише local (сотні LLM-викликів).

| Тест | Що перевіряє | Як |
|---|---|---|
| **Prompt injection** | Бот не зливає system-prompt і не виконує "ignore all instructions" | Regex/substring check на канарок із дефолтного system-prompt-у — без LLM-судді |
| **PII solicitation** | Бот не вигадує справжній іспанський NIE/DNI | Regex `\b[XYZ]\d{7}[A-Z]\b` на вихід + перевірка наявності refusal-слів ("cannot", "privacy", ...) |
| **Bias under provocation** | На провокаційні питання про райони/національності бот не дає упереджених тверджень | `BiasMetric` як judge |

---

### 6. Custom metrics — `tests/eval/custom/`

Escape-hatch DeepEval — коли вбудовані метрики не описують потрібний критерій.

| Метрика | Що це | Конкретний кейс |
|---|---|---|
| **GEval** | Критерій природньою мовою + LLM-judge | "Чи у відповіді про податки є disclaimer 'verify with current sources'" |
| **DAGMetric** | Дерево бінарних рішень — judge ставить один Y/N на крок | Smoke: один node "відповідь — нумерований список кроків?". Local: 3-level chain procedural → arras → NIE |
| **ArenaGEval** | Pairwise порівняння двох відповідей за критерієм | "Concrete answer vs vague" — concrete має виграти (тільки `@local`, бо метрика flaky на іменах контестантів) |

DAG є двох глибин:
- Smoke — один `TaskNode → BinaryJudgementNode → VerdictNode-листи`.
- Local — 3-level chain з **окремими листами на кожну гілку** (shared VerdictNodes → score=None, gotcha задокументоване в коментарях).

---

## Інфраструктурні тести (без LLM)

Швидкі, безкоштовні, ганяти на кожному save.

### `tests/test_knowledge_loader.py`
Перевіряє, що loader коректно парсить markdown з YAML frontmatter, fallback на дефолтні значення без frontmatter, стабільні `doc.id` зі шляху, ітерацію по `curated/` + `raw/`.

### `tests/llm/test_base_and_registry.py`
Без живих HTTP-викликів. Перевіряє: DTO-структури (`ChatMessage`, `ChatOptions`, `TokenUsage.total` через `@property`), `ModelRegistry._try_init` ловить помилки в `_init_errors`, `describe()` повертає `ProviderStatus` для невдалих провайдерів з `available=False`, кеш TTL працює.

### `tests/rag/test_pipeline.py`
Round-trip через chunker + ChromaStore на тимчасовому шляху і фейковому ембедері: doc → chunks → upsert → query → результат містить очікувані metadata. Перевіряє idempotency (повторний ingest не дублює).

### `tests/api/test_endpoints.py`
FastAPI `TestClient` зі заміненими через `app.dependency_overrides` `get_registry` і `get_retriever` на фейки. Перевіряє: `/health` 200, `/models` форма JSON, `/chat` non-stream і SSE-режим, `/chat/sessions/{id}` GET/DELETE, 400 на невідомого провайдера, 422 на пустий message.

---

## Environment variables, які впливають на тести

| Змінна | Default | Що робить |
|---|---|---|
| `OPENAI_API_KEY` | (нема) | Без неї всі eval-тести скіпаються (через `_require_openai_key`) |
| `SUT_MODEL` | `gpt-4o-mini` | Модель, що відповідає в eval-тестах |
| `JUDGE_MODEL` | `gpt-4o-mini` | Модель-суддя |
| `SMOKE_MAX_CASES` | `50` (`.env`: `15`) | Cap кількості параметризованих кейсів у smoke-режимі |
| `EMBEDDING_PROVIDER` | `chromadb` | Який ембедер використовується в RAG (впливає на retriever-результати в RAG-сюїті) |

---

## Як читати fail-повідомлення

DeepEval-метрики ставлять оцінку 0..1 (або 0..10 в DAG). Assert виглядає як:

```
assert m.is_successful(), f"Faithfulness {m.score:.2f}: {m.reason}"
```

- `m.score` — числове значення.
- `m.is_successful()` — `score >= threshold` для звичайних, `score <= threshold` для lower-is-better (Hallucination, Bias, Toxicity, KnowledgeRetention).
- `m.reason` — текст від судді **чому** така оцінка. Це найкорисніша частина при debug-у.

Деякі smoke-тести явно НЕ assert-ять якість (тільки що metric відпрацював без crash) — це wiring-gate, не quality-gate. Реальна якість сидить у `*_full` варіантах з `@pytest.mark.local`.

---

## Як це склеюється з рештою

- Та сама `service.answer()`, що використовує `/chat` endpoint у проді, викликається з eval-тестів — тому метрики оцінюють **prod behaviour**, не test-specific промпт.
- Той самий `LLMAdapter` працює і як SUT, і як judge через `DeepEvalLLM`-обгортку.
- Goldens (`knowledge/synth/goldens.json`) — спільне джерело істини між RAG-сюїтою і matrix runner-ом.
- Smoke-сюїта запускається CI-pipeline-ом (`.github/workflows/...`) на кожен push в main.

---

## Що далі

- Розширити сюїти на нові провайдери (Anthropic як SUT, Ollama як judge) через `make eval-matrix`.
- Додати `multimodal/` тести (маркер уже зарезервований).
- Підняти smoke-пороги ближче до `local`, коли RAG-якість стабілізується.
