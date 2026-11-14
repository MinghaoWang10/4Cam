"""让测试在仓库根目录或 simulation 目录下运行都能找到 `sim` 包。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
