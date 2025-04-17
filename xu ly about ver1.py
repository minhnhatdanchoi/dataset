from py2neo import Graph
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch_geometric.nn import GraphSAGE
from torch_geometric.data import Data
import re

# 🔗 Kết nối đến Neo4j
NEO4J_URI = "neo4j+s://fa2fd127.databases.neo4j.io"
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
x = torch.randn(n_nodes, 128)  # Random feature vector cho mỗi node
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


# 📌 Hàm tính toán độ tin cậy của quan hệ
def calculate_confidence(prediction_score, entity_text, about_text):
    # Tham số trọng số
    alpha = 0.2  # Trọng số cho điểm dự đoán từ GraphSAGE
    beta = 0.4  # Trọng số cho độ cụ thể của entity
    gamma = 0.4  # Trọng số cho ngữ cảnh

    # Đo lường độ cụ thể (specificity) của entity
    # Càng dài và chi tiết càng cụ thể
    specificity_score = min(1.0, len(entity_text.split()) / 10)

    # Đo lường chất lượng ngữ cảnh xung quanh entity
    # Tìm vị trí của entity trong about_text
    try:
        entity_pos = about_text.find(entity_text)
        if entity_pos >= 0:
            # Lấy 20 ký tự trước và sau entity để đánh giá ngữ cảnh
            start = max(0, entity_pos - 20)
            end = min(len(about_text), entity_pos + len(entity_text) + 20)
            context = about_text[start:end]

            # Đánh giá ngữ cảnh dựa trên số từ khóa quan trọng xuất hiện
            context_keywords = ["experience", "expert", "skill", "proficient",
                                "knowledge", "worked", "years", "project",
                                "expertise"]
            keyword_count = sum(1 for keyword in context_keywords if
                                keyword.lower() in context.lower())
            context_score = min(1.0, keyword_count / 3)
        else:
            context_score = 0.0
    except:
        context_score = 0.0

    # Tính tổng điểm tin cậy theo công thức
    confidence = alpha * prediction_score + beta * specificity_score + gamma * context_score

    return confidence, {
        "prediction": prediction_score,
        "specificity": specificity_score,
        "context": context_score
    }

# 📌 Định nghĩa mô hình GraphSAGE
class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.sage = GraphSAGE(in_channels, hidden_channels, num_layers=2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, data):
        x = self.sage(data.x, data.edge_index)
        edge_embeds = torch.cat([x[data.edge_index[0]], x[data.edge_index[1]]], dim=1)
        return self.mlp(edge_embeds).squeeze()

# 📌 Huấn luyện mô hình
model = LinkPredictor(in_channels=128, hidden_channels=256)
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
    for idx, (src, tgt) in enumerate(predicted_edges):
        src_name = id_to_name.get(src, f"Node {src}")
        tgt_name = id_to_name.get(tgt, f"Node {tgt}")

        # Lấy điểm dự đoán từ mô hình
        pred_score = predictions[idx].item()

        # Kiểm tra nếu About là None hoặc rỗng
        if tgt_name is None or tgt_name.strip() == "":
            print(f"⚠ {src_name} -> Không có thông tin từ About (BỎ QUA)")
            continue  # Bỏ qua quan hệ này

        # Nếu About là một đoạn text, trích xuất thực thể
        entities = extract_entities(tgt_name)
        if entities["skills"] or entities["projects"] or entities["expertise"]:
            for skill in entities["skills"]:
                confidence, details = calculate_confidence(pred_score, skill,
                                                           tgt_name)
                if confidence > 0.6:  # Chỉ giữ lại các quan hệ có độ tin cậy cao
                    print(
                        f"🛠️ {src_name} -> {skill} (HAS_DETAIL_EXPERIENCE) [Confidence: {confidence:.2f}]")

            for project in entities["projects"]:
                confidence, details = calculate_confidence(pred_score, project,
                                                           tgt_name)
                if confidence > 0.6:
                    print(
                        f"🛠️ {src_name} -> {project} (WORKED_ON) [Confidence: {confidence:.2f}]")

            for exp in entities["expertise"]:
                confidence, details = calculate_confidence(pred_score, exp,
                                                           tgt_name)
                if confidence > 0.6:
                    print(
                        f"🛠️ {src_name} -> {exp} (EXPERTISE) [Confidence: {confidence:.2f}]")
        else:
            print(f"⚠ {src_name} -> Không tìm thấy thực thể nào (BỎ QUA)")

# 7️⃣ Cập nhật quan hệ vào Neo4j
if predicted_edges:
    print("\n📡 Cập nhật vào Neo4j...")
    for idx, (src, tgt) in enumerate(predicted_edges):
        src_name = id_to_name.get(src, f"Node {src}")
        tgt_name = id_to_name.get(tgt, f"Node {tgt}")
        pred_score = predictions[idx].item()

        # Nếu About là None hoặc rỗng, bỏ qua
        if tgt_name is None or tgt_name.strip() == "":
            continue

        # Trích xuất thực thể từ About
        entities = extract_entities(tgt_name)
        relationships = []

        if entities["skills"]:
            for skill in entities["skills"]:
                confidence, _ = calculate_confidence(pred_score, skill,
                                                     tgt_name)
                if confidence > 0.6:
                    relationships.append(
                        (src_name, "HAS_DETAIL_EXPERIENCE", skill, confidence))

        if entities["projects"]:
            for project in entities["projects"]:
                confidence, _ = calculate_confidence(pred_score, project,
                                                     tgt_name)
                if confidence > 0.6:
                    relationships.append(
                        (src_name, "WORKED_ON", project, confidence))

        if entities["expertise"]:
            for exp in entities["expertise"]:
                confidence, _ = calculate_confidence(pred_score, exp, tgt_name)
                if confidence > 0.6:
                    relationships.append(
                        (src_name, "EXPERTISE", exp, confidence))

        # Chèn vào Neo4j
        for (source, relation, target, conf) in relationships:
            cypher_query = f"""
            MERGE (e:Employee {{name: '{source}'}})
            MERGE (t:Detail_Exp {{name: '{target}'}})
            MERGE (e)-[r:{relation}]->(t)
            SET r.confidence = {conf}
            """
            graph.run(cypher_query)
            print(
                f"✅ {source} -[:{relation} (conf:{conf:.2f})]-> {target} đã được thêm vào Neo4j")

    print("🎯 Cập nhật hoàn tất!")
else:
    print("⚠ Không có quan hệ mới để cập nhật.")