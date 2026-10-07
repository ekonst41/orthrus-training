# Разработка и запуск

## Локальное окружение (любая ОС, CPU)

Нужен [uv](https://docs.astral.sh/uv/). Python 3.12 uv скачает сам.

```bash
uv venv --python 3.12 .venv
uv pip sync requirements/dev.txt
```

Активация: `.venv\Scripts\activate` (Windows) или `source .venv/bin/activate` (Linux/macOS).

Без GPU локально запускаются только тесты на крошечных моделях. FlexAttention на CPU
не поддерживает backward, поэтому в тестах внимание диффузионного прохода считается
через плотную маску и SDPA (эталон, с которым сверяется flex).

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
2. `cp .env.example .env` и заполнить `DS_PROJECT_ID` (ID проекта — в URL страницы проекта).
3. В проекте DataSphere создать секрет `HF_TOKEN`: секреты проекта попадают в задания как переменные окружения.

Запуск (из корня репозитория, в активированном `.venv`):

```bash
python scripts/ds.py run jobs/probe-system.yaml --max-minutes 60  # запустить, смотреть логи,
                                                  # отменить задание через 60 минут
python scripts/ds.py attach <job_id>              # переподключиться к идущему заданию
python scripts/ds.py list                         # задания проекта
python scripts/ds.py cancel <job_id>
python scripts/ds.py download <job_id>            # скачать outputs (не больше 1 ГБ)
```

Задания описаны в `jobs/*.yaml`. Пути в них указаны относительно корня репозитория. В этих файлах только ASCII: на Windows CLI читает YAML в системной кодировке, и кириллица ломает разбор. Окружение
Python задаётся вручную (Python 3.12 и `requirements/train.txt`), поэтому DataSphere не анализирует
локальную машину. Логи CLI сохраняются в `.ds_logs/`.

Ограничения, которые влияют на код:

- на одно задание до 10 ГБ данных, один файл до 5 ГБ, через CLI скачивается до 1 ГБ результатов,
  поэтому чекпоинты и датасеты хранятся вне `outputs`;
- хранилище проекта подключается к заданию только для чтения;
- рабочий каталог задания `/job` — общий диск с ~34 ГБ свободного места (запись ~60 МБ/с),
  поэтому данным и чекпоинтам нужен `working-storage` (от 100 ГБ, оплачивается);
- данные задания (кеш, логи) хранятся 14 дней, файл лога до 100 МБ;
- g2.1 (1×A100 80 ГБ) стоит ≈543 ₽/ч, тарификация посекундная.

## Git

`.gitattributes` держит переводы строк LF и на Windows: иначе bash-скрипты, загруженные в
задание, падают на `\r`.
