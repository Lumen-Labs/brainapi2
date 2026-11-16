# Policy conflict gold production

The scored corpus is built from **GitHub Docs** (CC-BY-4.0 help center) and the **GSA-TTS handbook** (U.S. government work / CC0 policy).

1. Extract short paragraphs that contain a number+unit or required/optional polarity.
2. **Seeded pairs (majority):** duplicate the paragraph with a single documented mutation (number change or required↔optional) stored as a `stale copy`. Gold pair = original statement_id vs `*-seed`.
3. **Natural pairs:** two unmodified paragraphs from the same source whose titles share a topic token and whose numeric facts are disjoint.
4. **Near-misses:** Standard vs Enterprise plan titles with different numbers — listed on questions as `non_pairs`, **not** in `gold_pairs.jsonl`.
5. Original Northwind fixtures remain `ci_smoke`.

Every committed gold pair was produced by this script; a sample is reviewed in this file.

## Human review (sample)

- `stmt-gi-5e084203e9` vs seed — **pass**. GitHub Docs budget paragraph `$100` SKU budget vs seeded `$150`. Same audience, not plan-scoped.
- `stmt-gi-dd73555c97` vs seed — **pass**. India payment reminder "7 and 14 days later" vs seeded 21 days.
- `stmt-gi-b608972242` vs seed — **pass**. "After 14 days, paid features … will be locked" vs seeded 21 days.
- Northwind smoke pairs (`stmt-refund-14`/`30`, retention, password, 2FA) — **pass** (original hand-seeded fixtures).
- Plan-scoped near-misses titled `(Standard plan)` / `(Enterprise plan)` are listed as `non_pairs` only.

Seeded pairs are labeled `seeded: true`. Natural pairs (`seeded: false`) are unmodified GitHub/TTS paragraphs with disjoint numeric facts and overlapping title tokens.

Builder meta: `{"n_original_statements": 260, "n_seeded_twins": 175, "n_natural_pairs": 20, "n_gold_pairs": 195, "n_github_files": 105, "n_tts_files": 87}`

## Spot-check sample

### 1. `pc-pair-stmt-gi-5e084203e9-stmt-gi-5e084203e9-seed`

- split: `test`
- gold method: `seeded`
- question: Do statements stmt-gi-5e084203e9 and stmt-gi-5e084203e9-seed contradict each other for the same audience and time? If yes, list the statement ids.
- gold: `stmt-gi-5e084203e9 vs stmt-gi-5e084203e9-seed`
- must_include: `['stmt-gi-5e084203e9', 'stmt-gi-5e084203e9-seed']`
- gold_pairs: `[['stmt-gi-5e084203e9', 'stmt-gi-5e084203e9-seed']]`

### 2. `pc-pair-stmt-gi-449139d78a-stmt-gi-449139d78a-seed`

- split: `test`
- gold method: `seeded`
- question: Do statements stmt-gi-449139d78a and stmt-gi-449139d78a-seed contradict each other for the same audience and time? If yes, list the statement ids.
- gold: `stmt-gi-449139d78a vs stmt-gi-449139d78a-seed`
- must_include: `['stmt-gi-449139d78a', 'stmt-gi-449139d78a-seed']`
- gold_pairs: `[['stmt-gi-449139d78a', 'stmt-gi-449139d78a-seed']]`

### 3. `pc-pair-stmt-tt-28ff74e74f-stmt-tt-28ff74e74f-seed`

- split: `test`
- gold method: `seeded`
- question: Do statements stmt-tt-28ff74e74f and stmt-tt-28ff74e74f-seed contradict each other for the same audience and time? If yes, list the statement ids.
- gold: `stmt-tt-28ff74e74f vs stmt-tt-28ff74e74f-seed`
- must_include: `['stmt-tt-28ff74e74f', 'stmt-tt-28ff74e74f-seed']`
- gold_pairs: `[['stmt-tt-28ff74e74f', 'stmt-tt-28ff74e74f-seed']]`

### 4. `pc-pair-stmt-tt-56185c5007-stmt-tt-56185c5007-seed`

- split: `test`
- gold method: `seeded`
- question: Do statements stmt-tt-56185c5007 and stmt-tt-56185c5007-seed contradict each other for the same audience and time? If yes, list the statement ids.
- gold: `stmt-tt-56185c5007 vs stmt-tt-56185c5007-seed`
- must_include: `['stmt-tt-56185c5007', 'stmt-tt-56185c5007-seed']`
- gold_pairs: `[['stmt-tt-56185c5007', 'stmt-tt-56185c5007-seed']]`

### 5. `pc-pair-stmt-tt-034b8a6658-stmt-tt-034b8a6658-seed`

- split: `test`
- gold method: `seeded`
- question: Do statements stmt-tt-034b8a6658 and stmt-tt-034b8a6658-seed contradict each other for the same audience and time? If yes, list the statement ids.
- gold: `stmt-tt-034b8a6658 vs stmt-tt-034b8a6658-seed`
- must_include: `['stmt-tt-034b8a6658', 'stmt-tt-034b8a6658-seed']`
- gold_pairs: `[['stmt-tt-034b8a6658', 'stmt-tt-034b8a6658-seed']]`

### 6. `pc-pair-stmt-tt-6bc9a3f5e6-stmt-tt-6bc9a3f5e6-seed`

- split: `test`
- gold method: `seeded`
- question: Do statements stmt-tt-6bc9a3f5e6 and stmt-tt-6bc9a3f5e6-seed contradict each other for the same audience and time? If yes, list the statement ids.
- gold: `stmt-tt-6bc9a3f5e6 vs stmt-tt-6bc9a3f5e6-seed`
- must_include: `['stmt-tt-6bc9a3f5e6', 'stmt-tt-6bc9a3f5e6-seed']`
- gold_pairs: `[['stmt-tt-6bc9a3f5e6', 'stmt-tt-6bc9a3f5e6-seed']]`

### 7. `pc-pair-stmt-tt-f6d44930dd-stmt-tt-f6d44930dd-seed`

- split: `test`
- gold method: `seeded`
- question: Do statements stmt-tt-f6d44930dd and stmt-tt-f6d44930dd-seed contradict each other for the same audience and time? If yes, list the statement ids.
- gold: `stmt-tt-f6d44930dd vs stmt-tt-f6d44930dd-seed`
- must_include: `['stmt-tt-f6d44930dd', 'stmt-tt-f6d44930dd-seed']`
- gold_pairs: `[['stmt-tt-f6d44930dd', 'stmt-tt-f6d44930dd-seed']]`

### 8. `pc-pair-stmt-tt-121999007e-stmt-tt-121999007e-seed`

- split: `test`
- gold method: `seeded`
- question: Do statements stmt-tt-121999007e and stmt-tt-121999007e-seed contradict each other for the same audience and time? If yes, list the statement ids.
- gold: `stmt-tt-121999007e vs stmt-tt-121999007e-seed`
- must_include: `['stmt-tt-121999007e', 'stmt-tt-121999007e-seed']`
- gold_pairs: `[['stmt-tt-121999007e', 'stmt-tt-121999007e-seed']]`

### 9. `pc-pair-stmt-tt-3b9c2148c2-stmt-tt-3b9c2148c2-seed`

- split: `test`
- gold method: `seeded`
- question: Do statements stmt-tt-3b9c2148c2 and stmt-tt-3b9c2148c2-seed contradict each other for the same audience and time? If yes, list the statement ids.
- gold: `stmt-tt-3b9c2148c2 vs stmt-tt-3b9c2148c2-seed`
- must_include: `['stmt-tt-3b9c2148c2', 'stmt-tt-3b9c2148c2-seed']`
- gold_pairs: `[['stmt-tt-3b9c2148c2', 'stmt-tt-3b9c2148c2-seed']]`

### 10. `pc-pair-stmt-tt-8a00cf121a-stmt-tt-8a00cf121a-seed`

- split: `test`
- gold method: `seeded`
- question: Do statements stmt-tt-8a00cf121a and stmt-tt-8a00cf121a-seed contradict each other for the same audience and time? If yes, list the statement ids.
- gold: `stmt-tt-8a00cf121a vs stmt-tt-8a00cf121a-seed`
- must_include: `['stmt-tt-8a00cf121a', 'stmt-tt-8a00cf121a-seed']`
- gold_pairs: `[['stmt-tt-8a00cf121a', 'stmt-tt-8a00cf121a-seed']]`

### 11. `pc-pair-stmt-tt-df53134947-stmt-tt-df53134947-seed`

- split: `test`
- gold method: `seeded`
- question: Do statements stmt-tt-df53134947 and stmt-tt-df53134947-seed contradict each other for the same audience and time? If yes, list the statement ids.
- gold: `stmt-tt-df53134947 vs stmt-tt-df53134947-seed`
- must_include: `['stmt-tt-df53134947', 'stmt-tt-df53134947-seed']`
- gold_pairs: `[['stmt-tt-df53134947', 'stmt-tt-df53134947-seed']]`

### 12. `pc-pair-stmt-gi-1068387d8c-stmt-gi-0ef8ec786f`

- split: `test`
- gold method: `natural_numeric_clash`
- question: Do statements stmt-gi-1068387d8c and stmt-gi-0ef8ec786f contradict each other for the same audience and time? If yes, list the statement ids.
- gold: `stmt-gi-1068387d8c vs stmt-gi-0ef8ec786f`
- must_include: `['stmt-gi-1068387d8c', 'stmt-gi-0ef8ec786f']`
- gold_pairs: `[['stmt-gi-1068387d8c', 'stmt-gi-0ef8ec786f']]`
