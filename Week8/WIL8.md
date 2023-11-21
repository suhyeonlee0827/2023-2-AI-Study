Machine Reading Comprehension

문제점과 해결법
- 단어들의 구성이 유사하지는 않지만 동일한 의미의 문장을 이해
-> 컨텍스트 기반의 언어 이해를 하는 모델 GPT, BERT 사용

- 주어진 지문에서 질문에 대한 답을 찾을 수 없는 경우
-> 'no answer', 'unknown' 카테고리 학습

- multi-hop reasoning
-> 그래프 신경망 (GNN)

토큰화(tokenization): 주어진 코퍼스(corpus)에서 토큰(token)이라 불리는 단위로 나누는 작업. 보통 의미있는 단위로 토큰을 정의한다. 
- word tokenization, sentence tokenization

From Hugging Face docutmentation
- Byte-Pair Encoding (BPE) was initially developed as an algorithm to compress texts, and then used by OpenAI for tokenization when pretraining the GPT model. It’s used by a lot of Transformer models, including GPT, GPT-2, RoBERTa, BART, and DeBERTa.
