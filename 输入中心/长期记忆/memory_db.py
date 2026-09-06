# memory_db.py — 记忆存储层
# SQLite 存记忆，embedding 预存 BLOB，检索按标签过滤 + 向量精筛

import sqlite3
import numpy as np

class MemoryDB:
    def __init__(self, db_path, embed_model=None):
        """
        db_path: SQLite 文件路径
        embed_model: bge-base-zh-v1.5 模型（SentenceTransformer），None 时不做向量操作
        """
        self.db_path = db_path
        self.model = embed_model
        # check_same_thread=False：允许 HTTP 请求线程（子线程）访问主线程创建的连接
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_tables()

    def _init_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                domain TEXT NOT NULL,
                scale TEXT,
                time TEXT,
                eval TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_domain ON memories(domain)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_scale ON memories(scale)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_time ON memories(time)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_eval ON memories(eval)")
        # 防重复：同 text + 同 domain 只允许一条
        self.conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_mem ON memories(text, domain)")
        self.conn.commit()

    def add_memory(self, text, domain, scale=None, time=None, eval=None):
        """新增一条记忆，当场算 embedding 存进去；已存在（同 text+domain）则跳过。
        返回 True 表示真实插入，False 表示重复跳过。"""
        try:
            vec = None
            if self.model is not None:
                vec = self.model.encode([text])[0].astype(np.float32).tobytes()
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO memories (text, domain, scale, time, eval, embedding) VALUES (?,?,?,?,?,?)",
                (text, domain, scale, time, eval, vec)
            )
            self.conn.commit()
            return cur.rowcount > 0
        except sqlite3.IntegrityError:
            # UNIQUE(text, domain) 冲突：已存在，跳过
            return False

    def delete_memory(self, memory_id):
        self.conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        self.conn.commit()

    def search(self, query, domain=None, scale=None, time=None, eval=None, top_k=1):
        """
        检索：先 SQL 按标签过滤出候选集，再对候选集算余弦相似度
        query: 新对话文本（算 embedding 用）
        返回 [(id, text, cos), ...]
        """
        if self.model is None:
            raise ValueError("需要 embed_model 才能做向量检索")

        # 1. SQL 过滤
        sql = "SELECT id, text, embedding FROM memories WHERE embedding IS NOT NULL"
        params = []
        if domain:
            sql += " AND domain=?"
            params.append(domain)
        if scale:
            sql += " AND scale=?"
            params.append(scale)
        if time:
            sql += " AND time=?"
            params.append(time)
        if eval:
            sql += " AND eval=?"
            params.append(eval)

        rows = self.conn.execute(sql, params).fetchall()

        # 2. 向量精筛
        q_vec = self.model.encode([query])[0].astype(np.float32)
        q_norm = np.linalg.norm(q_vec)
        results = []
        for mem_id, text, emb_blob in rows:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            cos = float(np.dot(q_vec, emb) / (q_norm * np.linalg.norm(emb) + 1e-8))
            results.append((mem_id, text, cos))

        results.sort(key=lambda x: -x[2])
        return results[:top_k]

    def count(self):
        return self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def list_by_domain(self, domain, limit=100):
        rows = self.conn.execute(
            "SELECT id, text, scale, time, eval FROM memories WHERE domain=? LIMIT ?",
            (domain, limit)
        ).fetchall()
        return rows

    def close(self):
        self.conn.close()
