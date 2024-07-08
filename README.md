# MM-Lab

2024.07.08
### 1. BPC:
(1) Colab with 110 items 
!python /content/main.py\
    --model_name meta-llama/Llama-2-7b-hf\
    --task_name python\
    --block_size 400\
    --stride 64\
    --batch_size 4

    ---------- Result ----------
Total loss: 22700.653057813644
Character num: 84551
BPC: 0.38734159964339426
finished

<br>
(2) RTX 6000 with  110 items

```python
--model_name meta-llama/Llama-2-7b-hf \
--task_name python \
--block_size 1900 \
--stride 512 \
--batch_size 8
```

### 2. Sparse AutoEncoder
