# Qwen2-from-scratch从零实现QWen2

## 1.加载tokenizer

```python
from transformers import AutoTokenizer

model_path=""
tokenizer=AutoTokenizer.from_pretrained(model_path)
```

测试一下tokenizer是否正确加载

```python
encode_result=tokenizer.encode("hello world")
print(encode_result)
```

本实验中使用的qwen2-7B-Instruct，打印出来的结果是[14990,1879]，可以看到tokenizer将两个单词变成了两个token

tokenizer也可以自行实现，可以参考https://github.com/karpathy/minbpe

