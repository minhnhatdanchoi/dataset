from py2neo import Graph
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch_geometric.nn import GraphSAGE
from torch_geometric.data import Data
import re

# 🔗 Kết nối đến Neo4j
NEO4J_URI = "neo4j+ssc://fa2fd127.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "k6y0bLBbHmLw5g-lopuQFKvIsEvjyTig7Y2r-p7aPOc"

try:
    graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    print("✅ Kết nối Neo4j thành công!")
except Exception as e:
    print(f"❌ Lỗi kết nối Neo4j: {e}")
    exit()

# 📌 Trích xuất dữ liệu từ Neo4j
query = """
MATCH (e:Employee)-[:HAS_ABOUT]->(a:About)
RETURN id(e) AS source, id(a) AS target, e.name AS employee_name, a.type AS about_text
"""
result = graph.run(query).to_data_frame()

if result.empty:
    print("⚠ Không tìm thấy dữ liệu quan hệ Employee-About")
    exit()
else:
    print(result)

# 📌 Chuyển đổi dữ liệu sang tensor
edge_index = torch.tensor(result[['source', 'target']].values, dtype=torch.long).t().contiguous()
n_nodes = max(edge_index.flatten()).item() + 1
x = torch.randn(n_nodes, 16)  # Random feature vector cho mỗi node
data = Data(x=x, edge_index=edge_index)

# 📌 Tạo từ điển ánh xạ ID -> Tên thực thể
id_to_name = {row['source']: row['employee_name'] for _, row in result.iterrows()}
id_to_name.update({row['target']: row['about_text'] for _, row in result.iterrows()})

# 📌 Sinh negative samples
positive_edges = edge_index.t().tolist()
all_nodes = set(range(n_nodes))
negative_edges = []

while len(negative_edges) < len(positive_edges):
    src, tgt = random.sample(sorted(all_nodes), 2)
    if [src, tgt] not in positive_edges:
        negative_edges.append([src, tgt])

y = torch.cat([torch.ones(len(positive_edges)), torch.zeros(len(negative_edges))])
edge_index = torch.tensor(positive_edges + negative_edges, dtype=torch.long).t().contiguous()
data.edge_index = edge_index

# 📌 Định nghĩa mô hình GraphSAGE
class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.sage = GraphSAGE(in_channels, hidden_channels, num_layers=2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, data):
        x = self.sage(data.x, data.edge_index)
        edge_embeds = torch.cat([x[data.edge_index[0]], x[data.edge_index[1]]], dim=1)
        return self.mlp(edge_embeds).squeeze()

# 📌 Huấn luyện mô hình
model = LinkPredictor(in_channels=16, hidden_channels=32)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(100):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 📌 Dự đoán quan hệ ẩn
with torch.no_grad():
    predictions = torch.sigmoid(model(data))
    predicted_edges = edge_index[:, predictions > 0.7].t().tolist()

# 📌 Hàm trích xuất thông tin từ About
def extract_entities(about_text):
    skills = []
    projects = []
    expertise = []

    # Tìm "years of experience" (HAS_SKILL)
    exp_skills = re.findall(r'(\d+ years of experience in [\w\s]+)', about_text)
    skills.extend(exp_skills)

    # Tìm "Expert in ..." hoặc "Experienced in ..." (EXPERTISE)
    expert_exp = re.findall(r'(Expert in [\w\s]+|Experienced in [\w\s]+)', about_text)
    expertise.extend(expert_exp)

    # Nếu có từ "worked on" → PROJECT
    worked_projects = re.findall(r'(worked on [\w\s]+)', about_text)
    projects.extend(worked_projects)

    return {
        "skills": skills,
        "projects": projects,
        "expertise": expertise
    }

# 📌 Hiển thị quan hệ rõ ràng hơn
print("\n🔍 Quan hệ ẩn mới được phát hiện:")
if not predicted_edges:
    print("⚠ Không tìm thấy quan hệ ẩn nào")
else:
    for src, tgt in predicted_edges:
        src_name = id_to_name.get(src, f"Node {src}")
        tgt_name = id_to_name.get(tgt, f"Node {tgt}")

        # Kiểm tra nếu About là None hoặc rỗng
        if tgt_name is None or tgt_name.strip() == "":
            print(f"⚠ {src_name} -> Không có thông tin từ About (BỎ QUA)")
            continue  # Bỏ qua quan hệ này

        # Nếu About là một đoạn text, trích xuất thực thể
        entities = extract_entities(tgt_name)
        if entities["skills"] or entities["projects"] or entities["expertise"]:
            for skill in entities["skills"]:
                print(f"🛠️ {src_name} -> {skill} (HAS_SKILL)")
            for project in entities["projects"]:
                print(f"🛠️ {src_name} -> {project} (WORKED_ON)")
            for exp in entities["expertise"]:
                print(f"🛠️ {src_name} -> {exp} (EXPERTISE)")
        else:
            print(f"⚠ {src_name} -> Không tìm thấy thực thể nào (BỎ QUA)")


# 7️⃣ Cập nhật quan hệ vào Neo4j
if predicted_edges:
    print("\n📡 Cập nhật vào Neo4j...")
    for src, tgt in predicted_edges:
        src_name = id_to_name.get(src, f"Node {src}")
        tgt_name = id_to_name.get(tgt, f"Node {tgt}")

        # Nếu About là None hoặc rỗng, bỏ qua
        if tgt_name is None or tgt_name.strip() == "":
            continue

        # Trích xuất thực thể từ About
        entities = extract_entities(tgt_name)
        relationships = []

        if entities["skills"]:
            relationships.extend([(src_name, "HAS_SKILL", skill) for skill in entities["skills"]])
        if entities["projects"]:
            relationships.extend([(src_name, "WORKED_ON", project) for project in entities["projects"]])
        if entities["expertise"]:
            relationships.extend([(src_name, "EXPERTISE", exp) for exp in entities["expertise"]])

        # Chèn vào Neo4j
        for (source, relation, target) in relationships:
            cypher_query = f"""
            MERGE (e:Employee {{name: '{source}'}})
            MERGE (t:Entity {{name: '{target}'}})
            MERGE (e)-[:{relation}]->(t)
            """
            graph.run(cypher_query)
            print(f"✅ {source} -[:{relation}]-> {target} đã được thêm vào Neo4j")

    print("🎯 Cập nhật hoàn tất!")
else:
    print("⚠ Không có quan hệ mới để cập nhật.")