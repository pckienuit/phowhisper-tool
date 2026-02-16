# Hướng Dẫn Update Dependencies

## 🔄 Update Gemini API Package

Package `google-generativeai` đã deprecated. Cần chuyển sang package mới:

```bash
# Gỡ package cũ
pip uninstall google-generativeai

# Cài package mới
pip install google-genai
```

**Lưu ý**: Code đã được update để tương thích với cả 2 packages. Nếu chưa cài `google-genai`, code vẫn chạy được với package cũ (nhưng sẽ có warning).

## 📦 Full Requirements Update

Nếu muốn cài tất cả dependencies mới nhất:

```bash
pip install --upgrade -r requirements.txt
```

## ✅ Verification

Sau khi update, chạy lại tool:

```bash
python phowhisper.py
```

Không nên thấy warning nào về `google.generativeai` nữa.
