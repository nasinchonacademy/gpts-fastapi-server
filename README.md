
# 🧠 GPTs + RAG 예제 프로젝트 (무료 버전)

이 프로젝트는 OpenAI의 Custom GPTs 기능과 무료 오픈소스 임베딩 모델 (`InstructorXL`)을 이용해,
**대용량 문서(3GB 이상)를 기반으로 검색하고 답변하는 구조 (RAG)**를 구현하는 예제입니다.

---

## 📁 프로젝트 구조

```
gpts_rag_example/
├── embed_and_index.py       # 텍스트 임베딩 + 벡터 DB 생성
├── main.py                  # FastAPI 검색 API 서버
├── data/
│   └── extracted/           # ZIP에서 추출된 텍스트 샘플
├── index/
│   ├── vec.index            # FAISS 벡터 인덱스
│   └── chunks.json          # 문단 매핑 정보
```

---

## 🧰 1. 설치 방법

Python 3.8 이상 환경에서 아래 패키지를 설치하세요.

```bash
pip install sentence-transformers faiss-cpu fastapi uvicorn
```

---

## ⚙️ 2. 임베딩 및 인덱스 생성

```bash
python embed_and_index.py
```

---

## 🚀 3. 검색 API 서버 실행

```bash
uvicorn main:app --reload
```

---

## 🤖 4. Custom GPT에 API 도구 연결

ChatGPT (https://chat.openai.com/gpts)에서 GPT를 생성하고 다음과 같이 설정하세요:

1. GPT 만들기 → **Create GPT**
2. **Instructions**:
   ```
   너는 검색 도구를 통해 문서 내용을 찾아 설명해주는 도우미야.
   ```
3. **Tools > Add Action**:
   - Name: `SearchDocs`
   - URL: `http://localhost:8000/query`
   - Method: POST
   - Request Body:
     ```json
     {
       "query": "{{user_input}}"
     }
     ```
   - Response:
     ```json
     {
       "results": ["문단1", "문단2", "문단3"]
     }
     ```
