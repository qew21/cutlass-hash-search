#!/usr/bin/env python3
# scripts/pr_review.py

import os
import subprocess
from openai import OpenAI
import requests

def main():
    # --- 从环境变量读取必要信息 ---
    repo        = os.getenv("GITHUB_REPOSITORY")                     # e.g. "owner/repo"
    pr_number   = os.getenv("GITHUB_EVENT_PULL_REQUEST_NUMBER")      # e.g. "42"
    base_ref    = os.getenv("GITHUB_BASE_REF")                       # PR 的目标分支名
    sha         = os.getenv("GITHUB_SHA")                            # 本次提交的 SHA

    # --- 拉取 base 分支以便对比 ---
    subprocess.run(["git", "fetch", "origin", base_ref], check=True)
    # --- 生成 diff ---
    diff = subprocess.check_output(
        ["git", "diff", f"origin/{base_ref}...{sha}"],
        text=True
    )

    # --- 配置 OpenAI API ---
    api_key  = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    model = os.getenv("OPENAI_MODEL")
    client = OpenAI(api_key=api_key, base_url=api_base)
    
    # --- 组织 Prompt 并调用 LLM ---
    prompt = (
        "请针对以下 Pull Request 的 diff 提出代码质量、可读性、潜在 Bug 等反馈和优化建议：\n\n"
        "```diff\n" + diff + "\n```"
    )
    resp = client.chat.completions.create(
        model=model,    
        messages=[{"role":"user", "content": prompt}],
        stream=False,
    )
    feedback = resp.choices[0].message.content.strip()

    # --- 发布到 GitHub PR Comment ---
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"token {os.getenv('GITHUB_TOKEN')}",
        "Content-Type": "application/json"
    }
    data = {"body": feedback}
    r = requests.post(url, headers=headers, json=data)
    if not r.ok:
        print(f"❌ Comment 发布失败: {r.status_code}\n{r.text}")
        exit(1)

if __name__ == "__main__":
    main()
