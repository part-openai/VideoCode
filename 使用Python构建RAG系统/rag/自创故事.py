from typing import List

# 文件名
file_name = "自创故事.md"
# 用户提问
query = "故事中谁是大坏蛋？干过什么坏事？"
##################### 切分文件中的文本到chunks列表 ##################
def split_into_chunks(doc_file: str) -> List[str]:
    # 读取文件
    with open(doc_file, 'r') as f:
        content = f.read()
    return [chunk for chunk in content.split("\n\n")]


chunks = split_into_chunks(file_name)
for i, chunk in enumerate(chunks):
    print(f'[{i}] {chunk}\n')

############## 定义文本转向量的函数,通过Embedding模型转换 ##################
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")
def embed_chunk(chunk: str) -> List[float]:
    embedding = embedding_model.encode(chunk)
    return embedding.tolist()

# # 测试
# test_embedding = embed_chunk("测试内容")
# # 向量维度
# print(len(test_embedding))
# # 向量内容
# print(test_embedding)

##################################### 将片段内容通过Embedding模型转成向量 ####################################
embeddings = [embed_chunk(chunk) for chunk in chunks]
print(f"向量列表长度:{len(embeddings)}")


################ 将向量列表和片段内容，存到向量数据库 ################
import chromadb
# 创建内存类型的client，不会写入磁盘
chromadb_client = chromadb.EphemeralClient()
chromadb_collection = chromadb_client.get_or_create_collection(name="default")
# 保存到向量db
def save_embeddings(chunks: List[str], embeddings: List[List[float]]) -> None:
    ids = [str(i) for i in range(len(chunks))]
    chromadb_collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

save_embeddings(chunks, embeddings)
print("保存向量成功！")

# 将用户提问通过Embedding模型转成向量，并从向量数据库中进行检索
######################### 1.召回 ##############################
def retrieve(query: str, top_k: int) -> List[str]:
    # 将问题转成向量
    query_embedding = embed_chunk(query)
    results = chromadb_collection.query(
        query_embeddings = [query_embedding],
        n_results = top_k
    )
    return results['documents'][0]

# 问题
# query = "孙悟空被谁封印在了五指山，封贴是什么？"

# 召回
retrieved_chunks = retrieve(query, 5)
# 打印
for i, chunk in enumerate(retrieved_chunks):
    print(f'[{i}] {chunk}\n')

#################### 2.重排 ################
from sentence_transformers import CrossEncoder
def rerank(query: str, retrieved_chunks: List[str], top_k: int) -> List[str]:
    # 初始化Cross模型（多语言支持）
    cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)
    # 将片段文本和分数组成元组列表
    chunk_with_score_list = [(chunk, score)
                             for chunk, score in zip(retrieved_chunks, scores)]
    # 将列表按照分数高低进行倒序排序
    chunk_with_score_list.sort(key=lambda pair: pair[1], reverse=True)

    # 过滤分数，取前k个文本片段
    return [chunk for chunk, _ in chunk_with_score_list[:top_k]]

# 调用
rerank_chunks = rerank(query, retrieved_chunks, 3)
for i, chunk in enumerate(rerank_chunks):
    print(f'[{i}] {chunk}\n')

###################### 将重排后的结果和用户提问发送给大模型进行输出 ################
from dotenv import load_dotenv
from google import genai
# 加载环境变量，读取本目录下的.env文件
load_dotenv()
google_client = genai.Client()
def generate(query: str, chunks: List[str]) -> str:
    prompt = f"""你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。
    
用户问题: {query}
    
相关片段：
{"\n\n".join(chunks)}
    
请基于上述内容作答，不用瞎编乱造"""

    # 打印promot
    print(f"{prompt}\n\n---\n")
    llm = "gemini-2.5-flash"
    # 回答
    response = google_client.models.generate_content(
        model = llm,
        contents=prompt
    )
    return response.text

# 调用
answer = generate(query, rerank_chunks)
print(answer)

