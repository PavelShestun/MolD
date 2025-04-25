# MolD

Проект по разработке улучшенной эквивариантной диффузионной модели и RL-фреймворка для структурно-ориентированного дизайна и оптимизации лекарств.

## Установка

1. Клонируйте репозиторий:
   ```
   git clone https://github.com/PavelShestun/MolD.git
   ```
2. Установите зависимости:
   ```
   pip install -r requirements.txt
   ```

## Использование

### Обучение УЭДМ
```bash
python scripts/train_uedm.py
```

### Генерация молекул
```bash
python scripts/generate.py
```

### Обучение RL-агента
```bash
python scripts/train_rl.py
```

### Оптимизация молекул
```bash
python scripts/optimize.py
```
