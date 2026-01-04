"""
File: /logtable.py
Created Date: Sunday December 28th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday December 28th 2025 12:01:04 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from datetime import datetime
import os
from rich.table import Table
from rich.console import Console


def logtable(table: list[dict], title: str = ""):
    if not table:
        return

    all_keys = set()
    for row in table:
        all_keys.update(row.keys())

    headers = sorted(all_keys)

    rich_table = Table(title=title)
    for header in headers:
        rich_table.add_column(header)

    for row in table:
        rich_table.add_row(*[str(row.get(key, "")) for key in headers])

    tmp_folder = os.getcwd() + "/tmp/logtable"
    os.makedirs(tmp_folder, exist_ok=True)
    now_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    tmp_file = os.path.join(tmp_folder, f"{title}-{now_timestamp}.txt")
    with open(tmp_file, "w") as f:
        console = Console(file=f, force_terminal=False)
        console.print(rich_table)
