# Разработка и запуск

## Локальное окружение (любая ОС, CPU)

Нужен [uv](https://docs.astral.sh/uv/). Python 3.12 uv скачает сам.

```bash
uv venv --python 3.12 .venv
uv pip sync requirements/dev.txt
```

Активация: `.venv\Scripts\activate` (Windows) или `source .venv/bin/activate` (Linux/macOS).

Без GPU локально запускаются только тесты на крошечных моделях. FlexAttention на CPU
не поддерживает backward, поэтому на CPU внимание диффузионного прохода считается
через ту же маску в плотном виде и SDPA (тест сверяет его с FlexAttention).

```bash
python -m pytest            # ~3 секунды: модель, функция потерь, данные, конфиг, обучение
ruff check . && ruff format --check .
```

Сквозной тест (`tests/test_train.py`) обучает крошечную Qwen3 через `orthrus.train` с локальным
хранилищем, прерывает и продолжает обучение и проверяет, что веса совпадают бит в бит, а экспорт
соответствует официальному формату.

## Зависимости

Верхнеуровневые зависимости — в `requirements/*.in`, закреплённые версии — в `requirements/*.txt`:

| Файл | Где используется |
|---|---|
| `train.txt` | обучение и оценка на GPU (DataSphere, любой Linux-сервер с CUDA) |
| `datagen.txt` | генерация ответов учителя через vLLM (Linux + CUDA) |
| `dev.txt` | локальная разработка: CPU, тесты, линтер, DataSphere CLI |

torch 2.13.0 + transformers 5.17.0 выбраны как общий стек с vLLM 0.31. На A100 в DataSphere стоит
драйвер NVIDIA 535 (CUDA 12.2), а torch 2.13 с PyPI собран под CUDA 13 (нужен драйвер ≥580),
поэтому для GPU torch берётся из индекса PyTorch cu129 (совместим с драйвером 535).

После правки `.in` пересоберите lock-файлы (нужен uv):

```bash
python scripts/lock.py            # все; или: python scripts/lock.py train
```

`train.txt` и `datagen.txt` пишутся без комментариев и без `--index-url`: DataSphere CLI отвергает
их в файле зависимостей.

На своём GPU-сервере: `pip install -r requirements/train.txt`.

## DataSphere Jobs

Что нужно один раз:

1. [Yandex Cloud CLI](https://yandex.cloud/ru/docs/cli/quickstart) с авторизацией (`yc init`) под аккаунтом,
   у которого есть роль Developer в проекте DataSphere.
2. `cp .env.example .env` и заполнить `DS_PROJECT_ID` (ID проекта — в URL страницы проекта) и
   `ORTHRUS_BUCKET` (приватный HF-бакет). `scripts/ds.py` передаёт их в задания вместе с
   `ORTHRUS_GIT_COMMIT` — версией кода (с пометкой `-dirty`, если есть незакоммиченные правки:
   `ds.py` об этом предупреждает). Метрики смотрятся локально: `python -m orthrus.report`.
3. В проекте DataSphere создать секрет `HF_TOKEN` с правом записи в бакет: секреты проекта
   попадают в задания как переменные окружения.

Запуск (из корня репозитория, в активированном `.venv`):

```bash
python scripts/ds.py run jobs/smoke.yaml --max-minutes 60   # запустить, смотреть логи,
                                                          # отменить задание через 60 минут
python scripts/ds.py run jobs/train.yaml --retry-minutes 120   # ждать свободную A100 до 2 часов
python scripts/ds.py attach <job_id>              # переподключиться к идущему заданию
python scripts/ds.py list                         # задания проекта
python scripts/ds.py cancel <job_id>
python scripts/ds.py download <job_id>            # скачать outputs (не больше 1 ГБ)
```

Задания описаны в `jobs/*.yaml` (только ASCII: на Windows CLI читает YAML в системной кодировке).
Пути в них указаны относительно корня репозитория:

- код передаётся через `local-paths` (DataSphere кладёт его в `PYTHONPATH`, а не в `/job`) и
  запускается как `python3 -m <модуль>`; `local-paths: []` ломает datasphere 0.10.0;
- `inputs` — для данных и конфигов, они попадают в `/job` с тем же относительным путём;
- окружение Python задаётся вручную (`version`, `requirements-file`, `local-paths`), поэтому
  DataSphere не анализирует локальную машину;
- `cmd` оборачивается в `timeout`, а `ds.py run --max-minutes` отменяет задание снаружи:
  ограничения длительности на стороне DataSphere нет.

Логи CLI сохраняются в `.ds_logs/<время>-<задание>/`, результаты — в `outputs/` (не в git).

Поток логов CLI иногда останавливается (замечено после обновления токена `yc`), а завершение
задания CLI может не заметить: статус — `python scripts/ds.py get <id>`, вывод задания —
`.ds_logs/.../stdout.txt` или `ds.py attach <id>`; метрики обучения в бакете обновляются сами
(`python -m orthrus.report`). `cancel` снимает задание за ~20 с, не дожидаясь `graceful-shutdown`
(проверено `jobs/probe-signal.yaml`), поэтому для остановки с сохранением используйте
`python scripts/ds.py stop runs/<run>` (или `data/<dataset>`): задание само сохранится и выйдет.

Замеры скорости: `jobs/bench.yaml` (шаг обучения, `orthrus/bench.py`) и `jobs/genbench.yaml`
(генерация данных, `orthrus/genbench.py`); результаты печатаются строками `BENCH {...}` и
сохраняются в `bench/` в бакете.

### Проверенная среда g2.1 (пробы `jobs/probe-*.yaml`, 2026-10-08)

| Что | Результат |
|---|---|
| ВМ | A100-SXM4-80GB, драйвер 535.261.03 (CUDA 12.2), AMD EPYC 28 vCPU, 116 ГБ RAM |
| Образ | Ubuntu 22.04, `python3.12` есть, gcc 11.4, системный nvcc 11.8, conda нет |
| Диск `/job` | ~34 ГБ свободно, запись ~60 МБ/с |
| Сеть | PyPI, PyTorch, GitHub, HF доступны; HF ~40–50 МБ/с |
| Обучение (`train.txt`) | torch 2.13.0+cu129 работает; bf16 GEMM 268 TFLOPS; FlexAttention с маской Orthrus компилируется (~30 с), fwd+bwd 4.6 мс |
| Генерация (`datagen.txt`) | vLLM 0.31.0+cu129 работает с `VLLM_USE_FLASHINFER_SAMPLER=0`: JIT-сборка сэмплера FlashInfer системным nvcc 11.8 падает; Qwen3-0.6B даёт ~5 тыс. токенов/с даже в eager-режиме |

Окружение для генерации ставится так: `datagen.txt`, затем
`pip install --no-deps -r requirements/datagen-urls.txt` (колесо vLLM cu129 из GitHub-релиза).

Ограничения, которые влияют на код:

- на одно задание до 10 ГБ данных, один файл до 5 ГБ, через CLI скачивается до 1 ГБ результатов,
  поэтому чекпоинты и датасеты хранятся вне `outputs`;
- хранилище проекта подключается к заданию только для чтения;
- для данных и чекпоинтов в `/job` мало места, нужен `working-storage` (от 100 ГБ, оплачивается);
- данные задания (кеш, логи) хранятся 14 дней, файл лога до 100 МБ;
- установленное окружение не кешируется: каждое задание заново ставит зависимости (`train.txt`
  ~5 минут, `datagen.txt` ~8 минут), поэтому короткие эксперименты лучше объединять;
- g2.1 (1×A100 80 ГБ) стоит ≈543 ₽/ч, тарификация посекундная.
- свободной A100 может не оказаться («Unable to find available VM with spec [g2.1]»): задание
  падает до старта и не тарифицируется; задачам без GPU указывайте `c1.8`/`c1.4`.

## Хранилище данных и чекпоинтов

Всё, что должно пережить задание (сгенерированный датасет, чекпоинты, логи обучения), хранится в
приватном [HF Storage Bucket](https://huggingface.co/docs/hub/storage-buckets)
`<пользователь>/orthrus-training`: это изменяемое хранилище без git-истории (перезапись и удаление
на месте, `sync`, дедупликация Xet), бесплатно в пределах 100 ГБ приватного места. В коде:

```python
from huggingface_hub import sync_bucket

sync_bucket(
    "checkpoints/run1", "hf://buckets/<пользователь>/orthrus-training/runs/run1"
)  # выгрузить
sync_bucket("hf://buckets/<пользователь>/orthrus-training/runs/run1", "checkpoints/run1")  # скачать
```

Задания берут токен из секрета проекта `HF_TOKEN`, и ему нужны права на запись: fine-grained токен
с записью в бакет (и будущие репозитории проекта) или отдельный write-токен только для DataSphere.
Локально хватает read-токена. Проверка прав и скорости: `python scripts/ds.py run
jobs/probe-storage.yaml --max-minutes 60` (CPU-ВМ, 1–2 минуты). Замер 2026-10-08 из DataSphere:
выгрузка 1 ГиБ ~68 МБ/с, скачивание ~24 МБ/с, повторная выгрузка с половиной изменённых данных
передаёт только изменения (дедупликация Xet). Бакет должен быть приватным. В заданиях ставьте
`HF_HUB_DISABLE_PROGRESS_BARS=1`: прогресс-бары раздувают логи (лимит 100 МБ на файл).

Почему не другое: внутри DataSphere задание может писать только в `outputs` (скачать через CLI можно
1 ГБ) и в `output-datasets` (создаются лишь при успешном завершении, монтируются только на чтение,
квота 10 датасетов на сообщество); S3-коннектору нужен бакет в облаке, а прав там у команды нет;
Яндекс Диск на бесплатном тарифе ограничивает файл 1 ГБ и месячный трафик загрузки; Google Drive
требует Workspace (у сервисных аккаунтов нет квоты) или хрупкого OAuth личного аккаунта.

## Git

`.gitattributes` держит переводы строк LF и на Windows: иначе bash-скрипты, загруженные в
задание, падают на `\r`.
