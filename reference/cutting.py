import pulp
from collections import Counter
import time
import sys
import re

def solve_bounded_knapsack_optimized(values, weights, counts, capacity):
    """
    Решение ограниченной задачи о рюкзаке с использованием бинарного представления предметов.
    Если точная вместимость недостижима, возвращает лучшее возможное решение.

    Args:
        values: список значений предметов
        weights: список весов предметов
        counts: список с количеством доступных предметов каждого типа
        capacity: вместимость рюкзака

    Returns:
        Кортеж из двух элементов:
        - Максимальная достигнутая ценность
        - Список с выбранным количеством каждого предмета
    """
    n = len(values)

    # Преобразуем ограниченную задачу в 0-1 задачу о рюкзаке с помощью бинарного представления
    binary_values = []
    binary_weights = []
    item_mapping = []  # Для отслеживания оригинальных предметов

    for i in range(n):
        # Представляем каждый предмет в бинарном виде (1, 2, 4, 8, ...)
        j = 1
        remaining = counts[i]

        while j <= remaining:
            binary_values.append(values[i] * j)
            binary_weights.append(weights[i] * j)
            item_mapping.append((i, j))  # Запоминаем оригинальный индекс и количество
            remaining -= j
            j *= 2

        # Добавляем оставшиеся предметы
        if remaining > 0:
            binary_values.append(values[i] * remaining)
            binary_weights.append(weights[i] * remaining)
            item_mapping.append((i, remaining))

    # Решаем 0-1 задачу о рюкзаке
    binary_n = len(binary_values)
    dp = [0] * (capacity + 1)
    chosen = [[] for _ in range(capacity + 1)]  # Будем хранить выбранные "бинарные" предметы

    # Используем для отслеживания максимального достигнутого веса
    max_weight_achieved = 0

    for w in range(1, capacity + 1):
        for i in range(binary_n):
            if binary_weights[i] <= w and dp[w - binary_weights[i]] + binary_values[i] > dp[w]:
                dp[w] = dp[w - binary_weights[i]] + binary_values[i]
                chosen[w] = chosen[w - binary_weights[i]] + [i]  # Добавляем выбранный предмет

                # Обновляем максимальный достигнутый вес
                if dp[w] > 0 and w > max_weight_achieved:
                    max_weight_achieved = w

    # Если точная вместимость не может быть достигнута, берем лучшее возможное решение
    target_weight = capacity
    if dp[capacity] == 0:
        target_weight = max_weight_achieved

    # Восстанавливаем решение в терминах исходных предметов
    result_counts = [0] * n
    for item_idx in chosen[target_weight]:
        orig_idx, count = item_mapping[item_idx]
        result_counts[orig_idx] += count

    return dp[target_weight], result_counts


def min_material_cutting(requirements, stock, padding=5, show_paddings=True, verbose=True):
    """
    Оптимальный раскрой при нескольких типах исходных досок с ограниченными запасами.

    Минимизируется суммарная длина использованного материала (эквивалентно минимуму
    отходов, так как суммарная полезная длина фиксирована спросом). Используется
    целочисленное линейное программирование с генерацией столбцов (Gilmore-Gomory),
    обобщённой на несколько длин исходных досок.

    Args:
        requirements: список пар (длина, количество) — требуемые куски
        stock: список пар (длина, количество) — доступные на складе доски.
               «Неограниченный» тип задаётся просто большим количеством.
        padding: зазор (пропил) между соседними кусками
        show_paddings: печатать зазоры в детальном плане
        verbose: подробный вывод

    Returns:
        Кортеж (суммарная длина материала, план раскроя). При недопустимости — (-1, []).
    """
    TOLERANCE = 1e-6

    # --- Склад: агрегируем одинаковые длины, сортируем по убыванию длины ---
    stock_counter = Counter()
    for length, supply in stock:
        stock_counter[length] += supply
    stock_types = sorted(stock_counter.items(), reverse=True)  # [(длина, запас), ...]
    if not stock_types:
        return 0, []

    stock_len = [length for length, _ in stock_types]
    stock_sup = [supply for _, supply in stock_types]
    cap = [length + padding for length in stock_len]  # вместимость с учётом зазора
    n_stock = len(stock_types)
    max_stock_length = stock_len[0]

    # --- Требования: фильтруем то, что не влезает даже в самую длинную доску ---
    valid_requirements = [(l, c) for l, c in requirements if l <= max_stock_length]
    valid_requirements.sort(reverse=True)

    dropped = [(l, c) for l, c in requirements if l > max_stock_length]
    if dropped and verbose:
        print(f"Внимание: {len(dropped)} требований длиннее максимальной доски "
              f"{max_stock_length} мм — отброшены: {dropped}", file=sys.stderr)

    if not valid_requirements:
        return 0, []

    n_req = len(valid_requirements)
    req_len = [l for l, _ in valid_requirements]
    req_cnt = [c for _, c in valid_requirements]
    req_w = [l + padding for l in req_len]  # вес куска с учётом зазора

    # --- Паттерны (выровненные списки) ---
    patterns = []    # вектор количеств кусков
    pat_stock = []   # индекс типа исходной доски
    existing = set()

    def add_pattern(k, vec):
        key = (k, tuple(vec))
        if key in existing:
            return False
        existing.add(key)
        patterns.append(vec)
        pat_stock.append(k)
        return True

    # Начальные "жадные" одно-типовые паттерны для каждой пары (доска k, кусок i)
    for k in range(n_stock):
        for i in range(n_req):
            if req_w[i] <= cap[k]:
                vec = [0] * n_req
                vec[i] = cap[k] // req_w[i]
                add_pattern(k, vec)

    # --- Генерация столбцов ---
    max_iterations = 1000
    for iteration in range(max_iterations):
        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}, patterns: {len(patterns)}", file=sys.stderr)

        lp = pulp.LpProblem("Cutting_Master", pulp.LpMinimize)
        x = [pulp.LpVariable(f"x_{j}", lowBound=0) for j in range(len(patterns))]

        # Цель: минимизировать суммарную длину материала
        lp += pulp.lpSum(stock_len[pat_stock[j]] * x[j] for j in range(len(patterns)))

        # Спрос по каждой требуемой длине
        for i in range(n_req):
            lp += pulp.lpSum(patterns[j][i] * x[j] for j in range(len(patterns))) >= req_cnt[i], f"demand_{i}"

        # Ограничение запаса по каждому типу исходной доски
        for k in range(n_stock):
            lp += pulp.lpSum(x[j] for j in range(len(patterns)) if pat_stock[j] == k) <= stock_sup[k], f"supply_{k}"

        lp.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=30))

        if lp.status != pulp.LpStatusOptimal:
            if verbose:
                print("Релаксированная задача недопустима/неоптимальна — "
                      "вероятно, запасов склада не хватает под спрос", file=sys.stderr)
            break

        # Двойственные цены: y — спрос (>=0), z — запас (<=0)
        y = [lp.constraints[f"demand_{i}"].pi or 0.0 for i in range(n_req)]
        z = [lp.constraints[f"supply_{k}"].pi or 0.0 for k in range(n_stock)]

        # Прайсинг по каждому типу доски: рюкзак с её вместимостью
        added = False
        for k in range(n_stock):
            best_value, best_vec = solve_bounded_knapsack_optimized(y, req_w, req_cnt, cap[k])
            # reduced cost = c_j - y·a_j - z_k, где c_j = длина доски k, coeff запаса = 1
            reduced_cost = stock_len[k] - best_value - z[k]
            if reduced_cost < -TOLERANCE:
                if add_pattern(k, best_vec):
                    added = True

        if not added:
            if verbose:
                print(f"Останов: нет улучшающих паттернов (итерация {iteration})", file=sys.stderr)
            break

    # --- Целочисленная задача на найденном наборе паттернов ---
    ilp = pulp.LpProblem("Cutting_Integer", pulp.LpMinimize)
    yv = [pulp.LpVariable(f"y_{j}", lowBound=0, cat=pulp.LpInteger) for j in range(len(patterns))]

    ilp += pulp.lpSum(stock_len[pat_stock[j]] * yv[j] for j in range(len(patterns)))
    for i in range(n_req):
        ilp += pulp.lpSum(patterns[j][i] * yv[j] for j in range(len(patterns))) >= req_cnt[i]
    for k in range(n_stock):
        ilp += pulp.lpSum(yv[j] for j in range(len(patterns)) if pat_stock[j] == k) <= stock_sup[k]

    ilp.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=60))

    if ilp.status != pulp.LpStatusOptimal:
        if verbose:
            print("Целочисленная задача недопустима/неоптимальна", file=sys.stderr)
        return -1, []

    sol = [int(round(pulp.value(yv[j]) or 0)) for j in range(len(patterns))]

    # Проверка удовлетворения спроса
    for i in range(n_req):
        fulfilled = sum(patterns[j][i] * sol[j] for j in range(len(patterns)))
        if fulfilled < req_cnt[i] and verbose:
            print(f"Ошибка: требование {req_len[i]} мм не удовлетворено: "
                  f"{fulfilled} < {req_cnt[i]}", file=sys.stderr)

    # --- Формируем детальный план раскроя ---
    cutting_plan = []          # список (длина_доски, board_plan)
    boards_used = 0
    total_material = 0
    boards_by_stock = Counter()

    for j in range(len(patterns)):
        usage = sol[j]
        if usage <= 0:
            continue

        k = pat_stock[j]
        L = stock_len[k]
        boards_used += usage
        boards_by_stock[L] += usage
        total_material += usage * L

        for _ in range(usage):
            board_plan = []
            position = 0

            for i in range(n_req):
                for _ in range(patterns[j][i]):
                    board_plan.append({
                        "length": req_len[i],
                        "start": position,
                        "end": position + req_len[i],
                    })
                    position += req_len[i]

                    if padding > 0:
                        board_plan.append({
                            "length": padding,
                            "start": position,
                            "end": position + padding,
                            "padding": True,
                        })
                        position += padding

            # Убираем хвостовой зазор
            if padding > 0 and board_plan and board_plan[-1].get("padding"):
                board_plan.pop()
                position -= padding

            waste = L - position
            if waste > 0:
                board_plan.append({
                    "length": waste,
                    "start": position,
                    "end": L,
                    "waste": True,
                })

            cutting_plan.append((L, board_plan))

    # --- Отчёт ---
    if verbose:
        required_sum = sum(l * c for l, c in valid_requirements)
        print(f"\nСуммарная требуемая длина: {required_sum} мм")
        print(f"\nДетальный план раскроя ({boards_used} досок, {total_material} мм материала):")

        for idx, (L, board) in enumerate(cutting_plan):
            waste = 0
            pieces = []
            for piece in board:
                prefix = 'отступ ' if 'padding' in piece else ''
                if piece.get("waste", False):
                    waste = piece["length"]
                elif show_paddings or 'padding' not in piece:
                    pieces.append(f"{prefix}{piece['length']} мм ({piece['start']}-{piece['end']})")

            print(f"\nДоска #{idx+1} [{L} мм]:")
            print(f"  Куски: {', '.join(pieces)}")
            print(f"  Отходы: {waste} мм")

        print("\nИспользовано досок по типам:")
        for L in sorted(boards_by_stock, reverse=True):
            print(f"  {L} мм: {boards_by_stock[L]} шт")

        produced = Counter()
        for _, board in cutting_plan:
            for piece in board:
                if 'waste' not in piece and 'padding' not in piece:
                    produced[piece["length"]] += 1

        print("\nПроверка выполнения требований:")
        for length, count in valid_requirements:
            status = "V" if produced[length] >= count else "X"
            print(f"  {status} Требуется: {count} × {length} мм, произведено: {produced[length]}")

        end_waste = sum(p["length"] for _, b in cutting_plan for p in b if 'waste' in p)
        unused = total_material - required_sum  # включает пропилы и хвостовые отходы
        waste_percentage = (unused / total_material * 100) if total_material else 0
        print(f"\nХвостовые отходы: {end_waste} мм")
        print(f"Неиспользовано всего (с пропилами): {unused} мм")
        print(f"Общий процент неиспользованного: {waste_percentage:.2f}%")

    return total_material, cutting_plan


def parse_line(line):
    """Разбор строки 'длина[, количество]'. Без количества — 1."""
    items = re.split(r'[,\t]', line)
    length = int(items[0].replace(" ", ""))
    count = 1 if len(items) == 1 else int(items[1].strip())
    return length, count


def parse_input(stream):
    """Читает stdin, делит на секцию склада и секцию требований по строке '---'."""
    stock_lines = []
    req_lines = []
    section = stock_lines
    seen_sep = False

    for raw in stream:
        line = raw.strip()
        if not line:
            continue
        if line == "---":
            section = req_lines
            seen_sep = True
            continue
        section.append(line)

    if not seen_sep:
        print("Ошибка: не найден разделитель '---' между секцией склада и секцией требований",
              file=sys.stderr)
        sys.exit(1)

    return stock_lines, req_lines


# --- Точка входа ---
def main():
    stock_raw, req_raw = parse_input(sys.stdin)

    stock = [parse_line(l) for l in stock_raw]

    req_counter = Counter()
    for length, count in (parse_line(l) for l in req_raw):
        req_counter[length] += count
    requirements = list(req_counter.items())

    if not stock:
        print("Ошибка: пустая секция склада", file=sys.stderr)
        sys.exit(1)

    print(f"Типов досок на складе: {len(stock)}; всего досок: {sum(c for _, c in stock)}", file=sys.stderr)
    print(f"Различных требуемых длин: {len(requirements)}; всего кусков: {sum(c for _, c in requirements)}", file=sys.stderr)
    print("Запускаем оптимальный алгоритм для получения детального плана раскроя...", file=sys.stderr)

    start_time = time.time()
    total_material, cutting_plan = min_material_cutting(requirements, stock, show_paddings=False, verbose=True)
    end_time = time.time()

    print(f"\nВремя выполнения: {end_time - start_time:.2f} секунд", file=sys.stderr)


if __name__ == "__main__":
    main()
