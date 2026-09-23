"""Deterministic summary of validated comparison results; no inference."""
from collections import Counter
from src.models import ComparisonResult

DISCLAIMER = "Выводы носят рекомендательный характер и требуют проверки ответственным сотрудником по исходным документам."


def build_conclusion(result: ComparisonResult) -> list[tuple[str, str]]:
    structure = (f"В документах до реорганизации определено {result.before_units_total} структурных подразделений, "
                 f"после реорганизации — {result.after_units_total}.")
    if not result.structure_complete:
        structure += " Сопоставление организационной структуры выполнено частично; часть подразделений требует дополнительной проверки."
    if result.unit_changes:
        counts = Counter(c.status for c in result.unit_changes)
        structure += (f" По принятым сопоставлениям: создано: {counts['CREATED']}; сохранено: {counts['PRESERVED']}; "
                      f"преобразовано: {counts['TRANSFORMED']}; удалено: {counts['DELETED']}.")
    duties = Counter(c.status for c in result.responsibility_changes)
    functions = (f"В принятых результатах: без изменений: {duties['SAME']}; изменены: {duties['MODIFIED']}; "
                 f"перераспределены: {duties['MOVED']}; возможные потери: {duties['POTENTIALLY_LOST']}.")
    risks = Counter(f.type for f in result.findings)
    risk_text = (f"Гипотезы для проверки: возможное дублирование — {risks['DUPLICATION']}; "
                 f"потенциальные конфликты интересов — {risks['CONFLICT_OF_INTEREST']}; "
                 f"возможная потеря функции — {risks['FUNCTION_LOSS']}.")
    if not duties['POTENTIALLY_LOST'] and not risks['FUNCTION_LOSS']:
        functions += " В рамках выполненного анализа потенциальные потери не выявлены. Это не доказывает отсутствия потерь."
    recommendations = list(dict.fromkeys(f.recommendation.strip() for f in result.findings
                                        if f.recommendation and f.recommendation.strip()))
    limits = "Анализ завершён не полностью. " if not result.completed else ""
    if result.warnings:
        limits += " ".join(dict.fromkeys(result.warnings)) + " "
    limits += "Наличие ссылки подтверждает источник, но не гарантирует правильность интерпретации. " + DISCLAIMER
    return [("Организационная структура", structure), ("Изменения функций", functions),
            ("Выявленные риски", risk_text),
            ("Рекомендации", "\n\n".join(recommendations) if recommendations else
             "В принятых результатах рекомендации отсутствуют."), ("Ограничения анализа", limits)]
