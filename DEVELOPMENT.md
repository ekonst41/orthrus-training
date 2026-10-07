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

torch 2.13.0 + transformers 5.17.0 выбраны как общий стек с vLLM 0.31.
После правки `.in` пересоберите lock-файлы:

```bash
uv pip compile requirements/dev.in --universal --python-version 3.12 -o requirements/dev.txt
uv pip compile requirements/train.in --python-version 3.12 --python-platform x86_64-manylinux_2_28 --no-header --no-annotate -o requirements/train.txt
uv pip compile requirements/datagen.in --python-version 3.12 --python-platform x86_64-manylinux_2_28 --no-header --no-annotate -o requirements/datagen.txt
```

`--no-header --no-annotate` обязательны для `train.txt` и `datagen.txt`: DataSphere CLI отвергает
файл зависимостей с комментариями и маркерами окружения.

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
- данные задания (кеш, логи) хранятся 14 дней, файл лога до 100 МБ;
- g2.1 (1×A100 80 ГБ) стоит ≈543 ₽/ч, тарификация посекундная.

## Git

`.gitattributes` держит переводы строк LF и на Windows: иначе bash-скрипты, загруженные в
задание, падают на `\r`.
