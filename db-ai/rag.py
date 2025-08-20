#! https://github.com/timescale/private-rag-example/blob/main/local-rag-app-llama3.2.ipynb

import psycopg2

# the knowledge base
dummy_data = [
    {"title": "Seoul Tower", "content": "Seoul Tower is a communication and observation tower located on Namsan Mountain in central Seoul, South Korea."},
    {"title": "Gwanghwamun Gate", "content": "Gwanghwamun is the main and largest gate of Gyeongbokgung Palace, in Jongno-gu, Seoul, South Korea."},
    {"title": "Bukchon Hanok Village", "content": "Bukchon Hanok Village is a Korean traditional village in Seoul with a long history."},
    {"title": "Myeong-dong Shopping Street", "content": "Myeong-dong is one of the primary shopping districts in Seoul, South Korea."},
    {"title": "Dongdaemun Design Plaza", "content": "The Dongdaemun Design Plaza is a major urban development landmark in Seoul, South Korea."}
]

def connect_db():
    return psycopg2.connect( # use the credentials of your postgresql database 
        host = 'localhost',
        database = 'postgres',
        user = 'postgres',
        password = 'password',
        port = '5432'
    )

conn = connect_db()
cur = conn.cursor()
cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            title TEXT,
            content TEXT,
            embedding VECTOR(768)
        );
    """)
conn.commit()
cur.close()
conn.close()

conn = connect_db()
cur = conn.cursor()

# use the port at which your ollama service is running.
for doc in dummy_data:
    cur.execute("""
        INSERT INTO documents (title, content, embedding)
        VALUES (
            %(title)s,
            %(content)s,
            ollama_embed('nomic-embed-text', concat(%(title)s, ' - ', %(content)s), _host=>'http://ollama:11434')
        )
    """, doc)

conn.commit()
cur.close()
conn.close()

conn = connect_db()
cur = conn.cursor()
    
cur.execute("""
    SELECT title, content, vector_dims(embedding) 
    FROM documents;
""")

rows = cur.fetchall()
for row in rows:
    print(f"Title: {row[0]}, Content: {row[1]}, Embedding Dimensions: {row[2]}")

cur.close()
conn.close()

query = "Tell me about gates in South Korea."

conn = connect_db()
cur = conn.cursor()
    
# Embed the query using the ollama_embed function
cur.execute("""
    SELECT ollama_embed('nomic-embed-text', %s, _host=>'http://ollama:11434');
""", (query,))
query_embedding = cur.fetchone()[0]

# Retrieve relevant documents based on cosine distance
cur.execute("""
    SELECT title, content, 1 - (embedding <=> %s) AS similarity
    FROM documents
    ORDER BY similarity DESC
    LIMIT 3;
""", (query_embedding,))

rows = cur.fetchall()
    
# Prepare the context for generating the response
context = "\n\n".join([f"Title: {row[0]}\nContent: {row[1]}" for row in rows])
print(context)

cur.close()
conn.close()

conn = connect_db()
cur = conn.cursor()

# Generate the response using the ollama_generate function
cur.execute("""
    SELECT ollama_generate('llama3.2', %s, _host=>'http://ollama:11434');
""", (f"Query: {query}\nContext: {context}",))
    
model_response = cur.fetchone()[0]
print(model_response['response'])
    
cur.close()
conn.close()

