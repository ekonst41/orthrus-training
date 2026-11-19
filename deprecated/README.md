# Архив исследовательских экспериментов

Эти инструменты сохранены отдельно от основного pipeline подготовки данных, обучения
и стандартной оценки скорости. Основной код `src/` от этой папки не зависит.

| Путь | Что сохранено |
|---|---|
| `decoding/decoding.py` | Третий проход по хвосту блока, сдвиги и reuse в следующей проверке |
| `decoding/decode_exp.py` | Реальные декодирования, таблицы черновиков и проверка воспроизведения |
| `decoding/decode_report.py` | Воспроизведение на CPU, перебор политик, bootstrap-отчёт |
| `decoding/configs/` | Исходные конфиги экспериментов |
| `decoding/EXPERIMENTS.md` | Постановка и прежние результаты |
| `decoding/tests/` | Проверки стратегий и воспроизведения |
| `kv_cache/KV_cache_Qwen3_8B.ipynb` | Сравнение памяти KV-cache с внешними реализациями драфтеров |

## Декодирование

Запуск из корня проекта, после экспорта переменных `.env`:

```bash
python -m deprecated.decoding.decode_exp --config deprecated/decoding/configs/tail-v1.yaml --build-prompts
python -m deprecated.decoding.decode_exp --config deprecated/decoding/configs/tail-v1.yaml
python -m deprecated.decoding.decode_report --config deprecated/decoding/configs/tail-v1.yaml --grid small
```

Конфиги и ревизии моделей сохранены. Старый layout `storage.local_dir/data/` и
`storage.local_dir/experiments/` действует только внутри архивного runner.
По умолчанию `storage.local_dir=.`. Отправка идёт в тот же ORTHRUS_BUCKET.
Файл prompt set закреплён по sha256; после изменения промптов задай новый набор и его хэш.

Импорты обновлены на `src/`; runner и отчёт используют локальные записи вместо
облачных jobs. Численные алгоритмы стратегий сохранены. Исторический `EXPERIMENTS.md`
содержит сведения о старых запусках, а не результаты нового выполнения.

## KV-cache

Ноутбук сохранён как самостоятельное исследование с очищенными outputs. Он использует
собственное окружение, внешние модели и код DeepSpec. Эти зависимости не добавлены
в основной `pyproject.toml`. Повторный запуск сравнения требует настройки путей внутри ноутбука.
