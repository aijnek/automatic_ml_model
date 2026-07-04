"""ollama 呼び出しの共通部。

- designer: `ollama launch claude --model ... -- -p "<prompt>"` を subprocess で実行
- VLM: ollama Python ライブラリの chat API（structured outputs 使用）
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import ollama

DESIGNER_TIMEOUT_SEC = 3600  # ローカル35Bモデルは遅いので余裕を持つ


class DesignerLLMError(RuntimeError):
    pass


def run_designer_llm(prompt: str, command: list[str], timeout: int = DESIGNER_TIMEOUT_SEC) -> str:
    """特徴量設計LLM（ローカルClaude Code CLI）を1問1答で実行し、stdoutを返す。"""
    result = subprocess.run(
        command + [prompt],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise DesignerLLMError(
            f"designer LLM が失敗しました (exit {result.returncode}): {result.stderr[-2000:]}"
        )
    if not result.stdout.strip():
        raise DesignerLLMError("designer LLM の出力が空です")
    return result.stdout


def vlm_chat(
    model: str,
    prompt: str,
    image_path: Path,
    format_schema: dict,
    timeout: float = 120.0,
) -> str:
    """画像1枚 + プロンプトを VLM に投げ、structured output（JSON文字列）を返す。

    - think=False は必須: thinking モデル（qwen3.5等）は一部の画像で思考が
      暴走し、num_predict を思考だけで使い切って content が空になる
    - num_predict 上限も必須: 上限なしだとコンテキスト長まで数分間ハングする
    """
    client = ollama.Client(timeout=timeout)
    kwargs = dict(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(image_path)],
            }
        ],
        format=format_schema,
        options={"temperature": 0, "num_predict": 1024},
    )
    try:
        response = client.chat(think=False, **kwargs)
    except ollama.ResponseError:  # thinking 非対応モデル
        response = client.chat(**kwargs)
    return response["message"]["content"]
