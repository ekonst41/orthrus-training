# Orthrus для Qwen3

Подготовка ответов базовой модели, обучение diffusion attention на замороженной Qwen3,
оценка скорости и совпадения с обычной AR-генерацией. Один trainer работает через `python`
и через `torchrun` на нескольких GPU одного сервера.

## Установка

Python 3.12+, Linux и подходящие под драйвер CUDA-сборки PyTorch/vLLM для GPU-запусков.
Версии PyTorch и Transformers сохранены из исходного проекта; CUDA-сборку PyTorch
при необходимости установи перед установкой проекта.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .                       # обучение и оценка
pip install -e '.[datagen,notebooks,dev]'  # генерация данных, ноутбуки, проверки
cp .env.example .env
```

В `.env` задай `HF_TOKEN` с доступом к данным и записью в bucket и `ORTHRUS_BUCKET`.
Скрипты читают `.env` автоматически. При прямом вызове Python экспортируй переменные:
`set -a; source .env; set +a`. Для Nemotron требуется принять условия доступа на HF.

vLLM нужен только для подготовки ответов. Его бинарная сборка должна быть совместима
с PyTorch и драйвером на сервере. Исходный проект использовал сборки `cu129`, включая
специальное колесо vLLM; стандартное `pip install` может выбрать другую CUDA-сборку.
GPU-окружение проверяется коротким запуском `configs/smoke.yaml` перед полным обучением.

## Запуск

Команды из корня проекта:

Если датасет уже находится в прежнем bucket, пропусти `prepare.sh`: обучение скачает
его готовые шарды. Для новой генерации выбери новое имя, например добавь к командам
`data.dataset=nemotron-v2-qwen3-0.6b-minimal`, чтобы сохранить прежний набор.

```bash
bash scripts/prepare.sh configs/qwen3-0.6b.yaml
bash scripts/train.sh configs/qwen3-0.6b.yaml
bash scripts/benchmark.sh configs/qwen3-0.6b.yaml
```

Для нескольких GPU используется тот же конфиг и тот же цикл обучения:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 NPROC=3 bash scripts/train.sh configs/qwen3-0.6b.yaml \
    run_name=qwen3-0.6b-gamma8-3gpu train.global_batch_size=120 train.loss_gamma=8
```

Глобальный батч = число GPU × `micro_batch_size` × накопление градиентов.
В примере: 3 × 4 × 10 = 120. Он должен делиться на число GPU × micro-batch.
На каждой GPU находятся полная модель и свои активации; шардирования весов нет.

Прямые вызовы без shell-скриптов:

```bash
python -m src.dataset.prepare --config configs/qwen3-0.6b.yaml
python -m src.train.train --config configs/qwen3-0.6b.yaml
# То же обучение на четырёх GPU:
torchrun --standalone --nproc-per-node=4 --module src.train.train --config configs/qwen3-0.6b.yaml
python -m src.test.benchmark --config configs/qwen3-0.6b.yaml
python -m src.test.generate --config configs/qwen3-0.6b.yaml --prompt 'Solve 2x + 3 = 11' --compare-ar
```

Любой параметр можно переопределить как `section.key=value`. Блок включает anchor:
`model.block_size=16` означает 15 предложений диффузии после anchor. Новый размер блока,
другой loss или другой рецепт обучения требуют нового `run_name`.

## Данные и веса

Локальные пути:

```text
data/<dataset>/
  prompts.parquet            отобранные промпты с train/eval-разбиением
  eval_prompts.jsonl         отложенные промпты для генерации
  raw/shard-*.parquet        prompt_ids, response_ids и метаданные
  manifest.json             ревизии, параметры генерации, контрольные суммы
  packed/                   локальный кэш input_ids и assistant_mask
outputs/<run_name>/
  config.yaml               конфиг с фактическими значениями
  run.json                  данные запуска и версии библиотек
  train_metrics.jsonl        loss, agreement, LR, скорость, валидация и TPF
  checkpoints/step-*/        diffusion-веса, optimizer/scheduler и позиция обучения
  final/                    полный HF-экспорт: конфиг, токенизатор, веса и код модели
  eval-*.json               benchmark-результаты
```

Подготовка: Nemotron math/code/chat, равные доли, ответы самой базовой модели,
greedy, thinking выключен. Если в домене не хватает примеров, все домены обрезаются
до общего доступного количества. Промпты и ответы сохраняются как token IDs.
Диалоги перемешиваются и склеиваются в строки по `seq_len`; маска выделяет ответы ассистента.
Это сохраняет способ упаковки исходного проекта, включая переходы между диалогами внутри строки.

На HF используется прежний **Storage Bucket** с путями `data/<dataset>/` и `runs/<run_name>/`.
Готовые шарды, чекпоинты и финальная модель отправляются автоматически, синхронно.
Пишет только rank 0. При ошибке HF процесс сообщает об ошибке; уже сохранённый
локальный чекпоинт остаётся на диске. Для повторной отправки:

```bash
bash scripts/upload.sh configs/qwen3-0.6b.yaml --kind data
bash scripts/upload.sh configs/qwen3-0.6b.yaml --kind run
```

`hf.upload=false` отключает отправку, сохраняя возможность скачивания.
`hf.bucket=` полностью отключает HF Bucket. Локальный packed-кэш не загружается.
`keep_checkpoints` ограничивает только локальные копии; удалённая история не удаляется.

## Продолжение и остановка

Тот же `run_name` автоматически продолжает последний полный чекпоинт.
Сохраняются веса, optimizer, scheduler, шаг, fingerprint данных и рецепт. Порядок батчей
и якорей восстанавливается из seed, эпохи и глобального номера micro-batch.
Число GPU, данные, батчи и параметры обучения при продолжении должны совпадать.

`Ctrl+C`/`SIGTERM` запрашивают сохранение после текущего шага. В фоне можно создать
`outputs/<run_name>/STOP`; после сохранения файл удаляется. Жёсткое завершение процесса
сохраняет только последний ранее записанный чекпоинт.

Старые `raw/*.parquet` и полные экспорты модели поддерживаются. Чтобы взять прежний запуск
из bucket для оценки, укажи его `run_name`. Для инициализации нового обучения полным
экспортом укажи его локальную папку в `model.base` и соответствующий `block_size`.
Прямое продолжение optimizer-state старого trainer автоматически не конвертируется:
для него сохрани исходный архив. Существующий legacy-датасет можно читать для обучения,
а для новой генерации с новым форматом manifest нужно выбрать новое `data.dataset`.

## Что оценивает benchmark

GSM8K, MATH-500, HumanEval, MBPP и отложенные Nemotron-промпты. Считаются:

- `tpf = generated_tokens / (1 + 2 * cycles)` — с учётом prefill;
- `acceptance_length` — принятые draft-токены на цикл, без бонусного AR-токена;
- `tokens_per_second` и `speedup = AR_seconds / Orthrus_seconds`;
- `ar_match` — доля ответов, совпавших с greedy AR.

Правильность решения задач и pass@1 не оцениваются. Времена измеряются после прогрева
с CUDA-синхронизацией. Для сравнения скорости bf16 задай `eval.dtype=bfloat16`.
Для строгой проверки AR-parity используй float32; в bf16 близкие логиты могут менять argmax.
Во время обучения пишутся KL/agreement по позициям блока и TPF на отложенных промптах.
`eval.prompts=0` отключает генерацию во время обучения, а в benchmark означает весь набор.

## Код и ноутбуки

- `src/dataset/` — отбор, генерация, упаковка и единый загрузчик батчей.
- `src/train/` — trainer, KL-loss, checkpoint и распределённые операции.
- `src/test/` — benchmark, генерация и запись блоков для анализа.
- `src/model.py` — самодостаточная модель, копируется в HF-экспорт.
- `src/hub.py`, `src/config.py` — работа с bucket и YAML.
- `notebooks/check_pipeline.ipynb` — данные, заморозка весов, backward и AR-parity.
- `notebooks/token_analysis.ipynb` — блоки, top-4, вероятности отказов и признаки ошибок.
- `notebooks/compare_runs.ipynb` — кривые обучения и сравнение benchmark-результатов.
- `deprecated/` — исследования третьего прохода, перебор стратегий и сравнение KV-cache.

Веса loss: `loss_gamma=0` — равные; `loss_gamma>0` — `exp(-position/gamma)`.
Модель обучает только `q/k/v/o_proj_diff` и `q/k_norm_diff`.
Сохранены FlexAttention, компиляция функциональных слоёв, переиспользование целей учителя
и fused/chunked KL. Референсный модульный путь вынесен в тесты.

## Проверки

```bash
pytest -q
pytest deprecated/decoding/tests -q
ruff check src tests
```

CPU-тесты проверяют loss/градиенты против независимого autograd-расчёта, attention-маски,
заморозку AR, генерацию, упаковку, продолжение обучения, HF-экспорт и benchmark.
Интеграционный тест двухпроцессного обучения запускает настоящий `torchrun`/Gloo,
сравнивает его с одной CPU и проверяет resume. При запрете Gloo-сокетов он явно пропускается.

Проверка CUDA/NCCL, компиляции GPU-ядер и vLLM требует GPU-сервера.
Для первой проверки используй `configs/smoke.yaml`; для независимого повторного smoke
задай новые `run_name` и `data.dataset`.
