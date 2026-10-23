# 🧠 MBTI Content Recommendation System

> ML-система персонализации контента на основе психоанализа текстов и изображений. 
> Решает проблему холодного старта (+211% MRR) и автоматизирует профилирование пользователей.


[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

## 💡 Ключевые бизнес-результаты
- **+211% MRR** рекомендательной системы по сравнению с базовыми методами
- **48% Accuracy** классификации 16 типов личности (SOTA уровень для русскоязычных текстов)
- **64% Accuracy** визуального анализа профилей (+8 п.п. к существующим решениям)
- Внедрено в **ООО «Оллкьюб»** для повышения пользовательской вовлеченности

## 🛠️ Стек технологий
**ML/DL:** PyTorch, Transformers (DistilBERT, CLIP), PEFT (LoRA/DoRA), XGBoost, LightGBM  
**Data:** Pandas, NumPy, Scikit-learn, HuggingFace Datasets  
**Инфраструктура:** Accelerate, AMP (FP16), WeightedRandomSampler  
**Визуализация:** Matplotlib, Seaborn, t-SNE  

## 📊 Основные метрики
| Компонент | Метрика | Значение | Улучшение |
|-----------|---------|----------|-----------|
| Рекомендации | MRR | 0.513 | +211% vs baseline |
| Текст (16 классов) | Macro-F1 | 0.432 | SOTA |
| Визуал (4 дихотомии) | Avg Accuracy | 64% | +8 p.p. vs SOTA |
| Cold-Start | Soft P@10 | 0.361 | +28% vs Global Mean |


## 📚 Публикации

По результатам работы опубликовано 2 научные статьи:

1. **Гибридная функция потерь для иерархической классификации типов личности MBTI по текстовым данным**  
   *Пантелеев А. С., Афанасьева В. В.*  
   Описывает двухголовую архитектуру и комбинацию Focal Loss + MixUp для решения проблемы дисбаланса классов.  
   📄 [Современные проблемы математики и информатики, ТулГУ, 2026, С. 229–233](https://www.elibrary.ru/item.asp?id=91725310) | EDN: REGZGP

2. **От 90% до 21%: деконструкция иллюзии высокой точности в задачах автоматизированного психотипирования**  
   *Пантелеев А. С.*  
   Посвящена проблеме утечки данных (data leakage) в датасетах MBTI и методологии валидной оценки моделей.  
   📄 [Цифровое общество: образование, экономика, технологии, 2026, С. 372–377](https://www.elibrary.ru/item.asp?id=89175102) | EDN: KHKTTW


## Структура проекта

    mbti-content-recommendation/
    ├── README.md              # Главный экран: результаты, стек, бизнес-ценность
    ├── requirements.txt       # Зависимости
    ├── .gitignore
    ├── LICENSE                # MIT
    ├── configs/
    │   └── default.yaml       # Конфигурация
    ├── src/                   # Модульный продакшн-код
    │   ├── text_classifier.py
    │   ├── visual_classifier.py
    │   ├── recommendation.py
    │   ├── data_processing.py
    │   ├── synthetic_filter.py
    │   └── leakage_analysis.py
    ├── scripts/               # Точки входа
    │   ├── train_text.py
    │   ├── train_visual.py
    │   ├── evaluate_recommendations.py
    │   └── llm_generation_stub.py
    └── docs/
        ├── ARCHITECTURE.md    # Схема системы
        └── images/
