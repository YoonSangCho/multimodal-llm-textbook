# multimodal-llm-textbook
1. Multimodal Learning LLM Fundamentals  
2. Transformer Architecture  
3. Retrieval-Augmented Generation (RAG)  
4. Fine-tuning (LoRA)  
5. Multimodal Learning  
6. Engineering & Deployment  
7. References  

## 핵심 수식 예시

### Language Model

$$
P(y) = \prod_{t=1}^{T} P(y_t \mid y_{<t})
$$

각 항의 의미:
- $y_t$: t번째 토큰  
- $y_{<t}$: 이전 토큰 시퀀스  
- $P(y_t | y_{<t})$: 다음 단어 확률  

---

## Reference Papers

- Vaswani et al., *Attention is All You Need*, 2017  
- Brown et al., *GPT-3*, 2020  
- Radford et al., *CLIP*, 2021  
- Hu et al., *LoRA*, 2021  
