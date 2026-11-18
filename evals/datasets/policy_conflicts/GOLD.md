# Policy conflict gold production

The scored corpus is built from **GitHub Docs** (CC-BY-4.0 help center) and the **GSA-TTS handbook** (U.S. government work / CC0 policy).

1. Extract short paragraphs that contain a number+unit or required/optional polarity.
2. **Seeded pairs (majority):** duplicate the paragraph with a single documented mutation (number change or required↔optional) stored as a `stale copy`. Gold pair = original statement_id vs `*-seed`.
3. **Natural pairs:** two unmodified paragraphs from the same source whose titles share a *specific* topic token (generic words like “about” or “plans” do not count) and whose numeric facts are disjoint **but in the same unit family** (days vs days, money vs money). Pairs that only share a generic title word, or that mix unrelated units (e.g. 2FA days vs refund days, $1 hold vs 10 business days), are not gold.
4. **Near-misses:** Standard vs Enterprise plan titles with different numbers — listed on questions as `non_pairs`, **not** in `gold_pairs.jsonl`.
5. Original Northwind fixtures remain `ci_smoke`.

Every committed gold pair was produced by this script; a sample is reviewed in this file.

False-positive natural pairs called out in review (2FA vs refunds; $1 hold vs 10 business days; mixed-unit plan prices) were removed. Seeded `CC0`→`5` mutations are rejected.

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

### 9. `pc-pair-stmt-tt-b16b253d49-stmt-tt-b16b253d49-seed`

- split: `test`
- gold method: `seeded`
- question: Do statements stmt-tt-b16b253d49 and stmt-tt-b16b253d49-seed contradict each other for the same audience and time? If yes, list the statement ids.
- gold: `stmt-tt-b16b253d49 vs stmt-tt-b16b253d49-seed`
- must_include: `['stmt-tt-b16b253d49', 'stmt-tt-b16b253d49-seed']`
- gold_pairs: `[['stmt-tt-b16b253d49', 'stmt-tt-b16b253d49-seed']]`

### 10. `pc-pair-stmt-tt-1e1a87f547-stmt-tt-1e1a87f547-seed`

- split: `test`
- gold method: `seeded`
- question: Do statements stmt-tt-1e1a87f547 and stmt-tt-1e1a87f547-seed contradict each other for the same audience and time? If yes, list the statement ids.
- gold: `stmt-tt-1e1a87f547 vs stmt-tt-1e1a87f547-seed`
- must_include: `['stmt-tt-1e1a87f547', 'stmt-tt-1e1a87f547-seed']`
- gold_pairs: `[['stmt-tt-1e1a87f547', 'stmt-tt-1e1a87f547-seed']]`

### 11. `pc-pair-stmt-tt-45e29dbe89-stmt-tt-45e29dbe89-seed`

- split: `test`
- gold method: `seeded`
- question: Do statements stmt-tt-45e29dbe89 and stmt-tt-45e29dbe89-seed contradict each other for the same audience and time? If yes, list the statement ids.
- gold: `stmt-tt-45e29dbe89 vs stmt-tt-45e29dbe89-seed`
- must_include: `['stmt-tt-45e29dbe89', 'stmt-tt-45e29dbe89-seed']`
- gold_pairs: `[['stmt-tt-45e29dbe89', 'stmt-tt-45e29dbe89-seed']]`

### 12. `pc-pair-stmt-gi-e6f1105773-stmt-gi-9662dfbe64`

- split: `test`
- gold method: `natural_numeric_clash`
- question: Do statements stmt-gi-e6f1105773 and stmt-gi-9662dfbe64 contradict each other for the same audience and time? If yes, list the statement ids.
- gold: `stmt-gi-e6f1105773 vs stmt-gi-9662dfbe64`
- must_include: `['stmt-gi-e6f1105773', 'stmt-gi-9662dfbe64']`
- gold_pairs: `[['stmt-gi-e6f1105773', 'stmt-gi-9662dfbe64']]`
